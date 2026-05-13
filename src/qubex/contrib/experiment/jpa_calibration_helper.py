"""Utilities for JPA calibration sweeps from an Experiment instance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qxpulse import FlatTop, PulseSchedule, Waveform

from qubex.third_party.ons61797 import ONS61797

ResponseMetric = dict[str, NDArray[np.float64] | NDArray[np.complex128]]
SweepKey = tuple[str, float, float, float, float]


class DCVoltageController(Protocol):
    """Protocol for the DC source used to bias the JPA."""

    def on(self, channel: int) -> Any:
        """Turn on the specified output channel."""
        ...

    def off(self, channel: int) -> Any:
        """Turn off the specified output channel."""
        ...

    def set_voltage(self, channel: int, voltage: float) -> Any:
        """Set the output voltage for the specified channel."""
        ...

    def get_voltage(self, channel: int) -> float:
        """Return the current output voltage for the specified channel."""
        ...

    def get_output_state(self, channel: int) -> int:
        """Return the output state for the specified channel."""
        ...

    def close(self) -> Any:
        """Close the controller connection."""
        ...


@dataclass(frozen=True)
class JPAParameters:
    """Selected JPA bias and pump parameters."""

    dc_voltage: float
    pump_frequency: float
    pump_amplitude: float
    readout_amplitude: float | None = None


@dataclass
class JPAPhaseSweepResult:
    """Result from sweeping DC voltage and measuring readout phase."""

    qubit: str
    dc_voltages: NDArray[np.float64]
    signal: NDArray[np.complex128]

    @property
    def phase(self) -> NDArray[np.float64]:
        """Return the unwrapped phase of the measured signal."""
        return np.unwrap(np.angle(self.signal))


@dataclass
class JPASweepResult:
    """Result from sweeping JPA DC voltage, pump frequency, and pump amplitude."""

    qubits: list[str]
    dc_voltages: NDArray[np.float64]
    pump_frequencies: NDArray[np.float64]
    pump_amplitudes: NDArray[np.float64]
    readout_amplitudes: NDArray[np.float64]
    signal: dict[SweepKey, complex] = field(default_factory=dict)
    noise: dict[SweepKey, float] = field(default_factory=dict)
    snr: dict[SweepKey, float] = field(default_factory=dict)
    flatness: dict[SweepKey, float] = field(default_factory=dict)
    raw_data: dict[SweepKey, NDArray[np.complex128]] = field(default_factory=dict)

    def key(
        self,
        qubit: str,
        pump_frequency: float,
        pump_amplitude: float,
        dc_voltage: float,
        readout_amplitude: float,
    ) -> SweepKey:
        """Return the normalized dictionary key for one sweep point."""
        return (
            qubit,
            float(pump_frequency),
            float(pump_amplitude),
            float(dc_voltage),
            float(readout_amplitude),
        )

    def average_metric(
        self,
        metric: Literal["snr", "flatness"],
        *,
        pump_amplitude: float,
        readout_amplitude: float,
    ) -> NDArray[np.float64]:
        """Return the qubit-averaged metric over pump frequency and DC voltage."""
        source = self.snr if metric == "snr" else self.flatness
        values = np.zeros(
            (len(self.pump_frequencies), len(self.dc_voltages)),
            dtype=np.float64,
        )
        for i, pump_frequency in enumerate(self.pump_frequencies):
            for j, dc_voltage in enumerate(self.dc_voltages):
                per_qubit = [
                    source[
                        self.key(
                            qubit,
                            pump_frequency,
                            pump_amplitude,
                            dc_voltage,
                            readout_amplitude,
                        )
                    ]
                    for qubit in self.qubits
                ]
                values[i, j] = float(np.mean(np.abs(per_qubit)))
        return values

    def best_parameters(
        self,
        *,
        flatness_threshold: float | None = 1.2,
        readout_amplitude: float | None = None,
    ) -> JPAParameters:
        """Return the best JPA parameters according to average SNR."""
        if readout_amplitude is None:
            if len(self.readout_amplitudes) != 1:
                raise ValueError("readout_amplitude must be specified.")
            readout_amplitude = float(self.readout_amplitudes[0])

        best_score = -np.inf
        best_params: JPAParameters | None = None

        for pump_amplitude in self.pump_amplitudes:
            score = self.average_metric(
                "snr",
                pump_amplitude=float(pump_amplitude),
                readout_amplitude=readout_amplitude,
            )
            if flatness_threshold is not None:
                flatness = self.average_metric(
                    "flatness",
                    pump_amplitude=float(pump_amplitude),
                    readout_amplitude=readout_amplitude,
                )
                score = score.copy()
                score[flatness > flatness_threshold] = 0.0

            idx = np.unravel_index(int(np.argmax(score)), score.shape)
            max_score = float(score[idx])
            if max_score > best_score:
                best_score = max_score
                best_params = JPAParameters(
                    dc_voltage=float(self.dc_voltages[idx[1]]),
                    pump_frequency=float(self.pump_frequencies[idx[0]]),
                    pump_amplitude=float(pump_amplitude),
                    readout_amplitude=readout_amplitude,
                )

        if best_params is None:
            raise ValueError("No sweep data is available.")
        return best_params


@dataclass
class JPAFrequencyResponse:
    """Frequency response measured with JPA pump off and on."""

    readout_frequencies: NDArray[np.float64]
    on: dict[str, ResponseMetric]
    off: dict[str, ResponseMetric]

    def gain(
        self,
        qubit: str,
        metric: Literal["snr", "flatness"] = "snr",
    ) -> NDArray[np.float64]:
        """Return the on/off ratio for a response metric."""
        on_values = np.asarray(self.on[qubit][metric], dtype=np.float64)
        off_values = np.asarray(self.off[qubit][metric], dtype=np.float64)
        return np.divide(
            on_values,
            off_values,
            out=np.full_like(on_values, np.nan),
            where=off_values != 0,
        )


class JPACalibrationHelper:
    """
    Helper for the JPA calibration workflow used in mux calibration notebooks.

    The helper owns only the reusable measurement choreography. Instrument setup
    stays outside qubex: pass an object compatible with ``DCVoltageController``.
    """

    def __init__(
        self,
        experiment: Any,
        *,
        voltage_controller: DCVoltageController | None = None,
        dc_port: str | None = None,
        dc_ip_address: str | None = None,
        dc_channel: int | None = None,
        mux_label: str | None = None,
        pump_duration: float = 8 * 42,
        pump_tau: float = 8 * 6,
        voltage_tolerance: float = 1e-3,
        max_voltage_set_attempts: int = 100,
    ) -> None:
        if voltage_controller is not None and (
            dc_port is not None or dc_ip_address is not None
        ):
            raise ValueError(
                "Specify either voltage_controller or dc_port/dc_ip_address, not both."
            )

        self.experiment = experiment
        self.voltage_controller = voltage_controller or self._create_voltage_controller(
            dc_port=dc_port,
            dc_ip_address=dc_ip_address,
        )
        self.dc_channel = dc_channel
        self.mux_label = mux_label or self._default_mux_label()
        self.pump_duration = pump_duration
        self.pump_tau = pump_tau
        self.voltage_tolerance = voltage_tolerance
        self.max_voltage_set_attempts = max_voltage_set_attempts

    @staticmethod
    def _create_voltage_controller(
        *,
        dc_port: str | None,
        dc_ip_address: str | None,
    ) -> ONS61797 | None:
        if dc_port is None and dc_ip_address is None:
            return None
        return ONS61797(port=dc_port, ip_address=dc_ip_address)

    def _default_mux_label(self) -> str:
        mux_labels = self.experiment.mux_labels
        if len(mux_labels) != 1:
            raise ValueError("mux_label must be specified when multiple muxes are used.")
        return mux_labels[0]

    def _default_dc_channel(self) -> int:
        if self.dc_channel is None:
            return int(self.mux_label[3:]) + 1
        return self.dc_channel

    def set_dc_voltage(self, voltage: float) -> None:
        """Set the JPA DC voltage and verify the controller state."""
        if self.voltage_controller is None:
            raise ValueError("voltage_controller is required to set DC voltage.")

        channel = self._default_dc_channel()
        for _ in range(self.max_voltage_set_attempts):
            self.voltage_controller.on(channel=channel)
            self.voltage_controller.set_voltage(channel=channel, voltage=float(voltage))
            current_voltage = self.voltage_controller.get_voltage(channel=channel)
            output_state = self.voltage_controller.get_output_state(channel=channel)
            if (
                abs(float(voltage) - float(current_voltage)) < self.voltage_tolerance
                and output_state == 1
            ):
                return
        raise RuntimeError(f"Failed to set DC voltage to {voltage} V.")

    def turn_dc_off(self, *, close: bool = False) -> None:
        """Turn off the configured DC output channel."""
        if self.voltage_controller is None:
            return
        self.voltage_controller.off(channel=self._default_dc_channel())
        if close:
            self.voltage_controller.close()

    def readout_pulses(self, amplitude: float | None = None) -> dict[str, Waveform]:
        """Return readout pulses for all selected qubits."""
        return {
            qubit: self.experiment.readout(qubit, amplitude=amplitude)
            for qubit in self.experiment.qubit_labels
        }

    def pump_pulse(self, amplitude: float) -> FlatTop:
        """Return the JPA pump pulse."""
        return FlatTop(
            duration=self.pump_duration,
            amplitude=amplitude,
            tau=self.pump_tau,
        )

    def readout_with_pump_schedule(
        self,
        *,
        pump_amplitude: float,
        readout_amplitude: float | None = None,
        qubits: Sequence[str] | None = None,
    ) -> PulseSchedule:
        """Build a simultaneous readout schedule with a pump pulse."""
        if qubits is None:
            qubits = self.experiment.qubit_labels
        readout_pulses = self.readout_pulses(readout_amplitude)
        with PulseSchedule() as schedule:
            for qubit in qubits:
                schedule.add("R" + qubit, readout_pulses[qubit])
            schedule.add(self.mux_label, self.pump_pulse(pump_amplitude))
        return schedule

    @staticmethod
    def calculate_flatness(data: ArrayLike, *, n_angles: int = 20) -> float:
        """Estimate IQ-distribution flatness by rotating the complex samples."""
        comp = np.asarray(data, dtype=np.complex128)
        if comp.size == 0:
            return np.nan

        flatness_values = []
        for angle in np.linspace(0, 2 * np.pi, n_angles):
            rotated = comp * np.exp(1j * angle)
            real_std = float(np.std(np.real(rotated)))
            imag_std = float(np.std(np.imag(rotated)))
            if imag_std == 0:
                flatness_values.append(np.inf)
            else:
                flatness_values.append(abs(real_std / imag_std))
        return float(np.max(flatness_values))

    def sweep_parameters(
        self,
        *,
        dc_voltages: ArrayLike,
        pump_frequencies: ArrayLike,
        pump_amplitudes: ArrayLike,
        readout_amplitudes: ArrayLike | float = 0.1,
        shots: int = 100,
        mode: Literal["single", "avg"] = "single",
        reset_awg_and_capunits: bool = False,
        turn_off_when_done: bool = True,
        close_when_done: bool = False,
    ) -> JPASweepResult:
        """Sweep JPA parameters and collect signal, noise, SNR, and flatness."""
        dc_voltages = np.asarray(dc_voltages, dtype=np.float64)
        pump_frequencies = np.asarray(pump_frequencies, dtype=np.float64)
        pump_amplitudes = np.asarray(pump_amplitudes, dtype=np.float64)
        readout_amplitudes = np.atleast_1d(
            np.asarray(readout_amplitudes, dtype=np.float64)
        )
        sweep_result = JPASweepResult(
            qubits=list(self.experiment.qubit_labels),
            dc_voltages=dc_voltages,
            pump_frequencies=pump_frequencies,
            pump_amplitudes=pump_amplitudes,
            readout_amplitudes=readout_amplitudes,
        )

        try:
            for dc_voltage in dc_voltages:
                self.set_dc_voltage(float(dc_voltage))
                for readout_amplitude in readout_amplitudes:
                    for pump_frequency in pump_frequencies:
                        frequencies = {self.mux_label: float(pump_frequency)}
                        with self.experiment.modified_frequencies(frequencies):
                            for pump_amplitude in pump_amplitudes:
                                schedule = self.readout_with_pump_schedule(
                                    pump_amplitude=float(pump_amplitude),
                                    readout_amplitude=float(readout_amplitude),
                                )
                                result = self.experiment.execute(
                                    schedule=schedule,
                                    mode=mode,
                                    shots=shots,
                                    reset_awg_and_capunits=reset_awg_and_capunits,
                                )
                                self._store_measurement(
                                    sweep_result,
                                    result,
                                    pump_frequency=float(pump_frequency),
                                    pump_amplitude=float(pump_amplitude),
                                    dc_voltage=float(dc_voltage),
                                    readout_amplitude=float(readout_amplitude),
                                )
        finally:
            if turn_off_when_done:
                self.turn_dc_off(close=close_when_done)

        return sweep_result

    def sweep_dc_phase(
        self,
        *,
        dc_voltages: ArrayLike,
        qubit: str | None = None,
        readout_frequency: float | None = None,
        readout_amplitude: float | None = None,
        mode: Literal["single", "avg"] = "avg",
        shots: int | None = None,
        reset_awg_and_capunits: bool = False,
        turn_off_when_done: bool = True,
        close_when_done: bool = False,
    ) -> JPAPhaseSweepResult:
        """Sweep DC voltage and measure the unpumped readout phase."""
        if qubit is None:
            qubit = self.experiment.qubit_labels[0]
        dc_voltages = np.asarray(dc_voltages, dtype=np.float64)
        signal = []

        try:
            for dc_voltage in dc_voltages:
                self.set_dc_voltage(float(dc_voltage))
                frequencies = (
                    None
                    if readout_frequency is None
                    else {"R" + qubit: float(readout_frequency)}
                )
                with self.experiment.modified_frequencies(frequencies):
                    with PulseSchedule() as schedule:
                        schedule.add(
                            "R" + qubit,
                            self.experiment.readout(
                                qubit,
                                amplitude=readout_amplitude,
                            ),
                        )
                    result = self.experiment.execute(
                        schedule,
                        mode=mode,
                        shots=shots,
                        reset_awg_and_capunits=reset_awg_and_capunits,
                    )
                    signal.append(complex(np.average(self._kerneled(result, qubit))))
        finally:
            if turn_off_when_done:
                self.turn_dc_off(close=close_when_done)

        return JPAPhaseSweepResult(
            qubit=qubit,
            dc_voltages=dc_voltages,
            signal=np.asarray(signal, dtype=np.complex128),
        )

    def measure_frequency_response(
        self,
        *,
        params: JPAParameters,
        readout_frequencies: ArrayLike,
        target_qubit: str | None = None,
        readout_amplitude: float | None = None,
        shots: int = 100,
        reset_awg_and_capunits: bool = False,
        turn_off_when_done: bool = True,
        close_when_done: bool = False,
    ) -> JPAFrequencyResponse:
        """Measure readout-frequency response with the JPA pump off and on."""
        if target_qubit is None:
            target_qubit = self.experiment.qubit_labels[0]
        if readout_amplitude is None:
            readout_amplitude = params.readout_amplitude
        if readout_amplitude is None:
            readout_amplitude = 0.1

        readout_frequencies = np.asarray(readout_frequencies, dtype=np.float64)
        response: dict[str, dict[str, list[Any]]] = {
            "off": {"signal": [], "noise": [], "snr": [], "flatness": [], "raw_data": []},
            "on": {"signal": [], "noise": [], "snr": [], "flatness": [], "raw_data": []},
        }

        try:
            for state in ("off", "on"):
                coeff = 0.0 if state == "off" else 1.0
                self.set_dc_voltage(params.dc_voltage * coeff)
                with self.experiment.modified_frequencies(
                    {self.mux_label: params.pump_frequency}
                ):
                    for readout_frequency in readout_frequencies:
                        frequencies = {"R" + target_qubit: float(readout_frequency)}
                        with self.experiment.modified_frequencies(frequencies):
                            schedule = self.readout_with_pump_schedule(
                                pump_amplitude=params.pump_amplitude * coeff,
                                readout_amplitude=readout_amplitude,
                            )
                            result = self.experiment.execute(
                                schedule=schedule,
                                mode="single",
                                shots=shots,
                                reset_awg_and_capunits=reset_awg_and_capunits,
                            )
                            raw_data = self._kerneled(result, target_qubit)
                            metrics = self._metrics(raw_data)
                            for key, value in metrics.items():
                                response[state][key].append(value)
                            response[state]["raw_data"].append(raw_data)
        finally:
            if turn_off_when_done:
                self.turn_dc_off(close=close_when_done)

        return JPAFrequencyResponse(
            readout_frequencies=readout_frequencies,
            off={target_qubit: self._array_response(response["off"])},
            on={target_qubit: self._array_response(response["on"])},
        )

    def _store_measurement(
        self,
        sweep_result: JPASweepResult,
        result: Any,
        *,
        pump_frequency: float,
        pump_amplitude: float,
        dc_voltage: float,
        readout_amplitude: float,
    ) -> None:
        for qubit in sweep_result.qubits:
            raw_data = self._kerneled(result, qubit)
            metrics = self._metrics(raw_data)
            key = sweep_result.key(
                qubit,
                pump_frequency,
                pump_amplitude,
                dc_voltage,
                readout_amplitude,
            )
            sweep_result.signal[key] = metrics["signal"]
            sweep_result.noise[key] = metrics["noise"]
            sweep_result.snr[key] = metrics["snr"]
            sweep_result.flatness[key] = metrics["flatness"]
            sweep_result.raw_data[key] = raw_data

    @staticmethod
    def _kerneled(result: Any, qubit: str) -> NDArray[np.complex128]:
        return np.asarray(result.data[qubit][0].kerneled, dtype=np.complex128)

    @classmethod
    def _metrics(cls, raw_data: NDArray[np.complex128]) -> dict[str, Any]:
        signal = complex(np.average(raw_data))
        noise = float(np.std(raw_data))
        snr = np.nan if noise == 0 else float(abs(signal) / noise)
        return {
            "signal": signal,
            "noise": noise,
            "snr": snr,
            "flatness": cls.calculate_flatness(raw_data),
        }

    @staticmethod
    def _array_response(
        values: dict[str, list[Any]],
    ) -> ResponseMetric:
        return {
            "signal": np.asarray(values["signal"], dtype=np.complex128),
            "noise": np.asarray(values["noise"], dtype=np.float64),
            "snr": np.asarray(values["snr"], dtype=np.float64),
            "flatness": np.asarray(values["flatness"], dtype=np.float64),
            "raw_data": np.asarray(values["raw_data"], dtype=np.complex128),
        }
