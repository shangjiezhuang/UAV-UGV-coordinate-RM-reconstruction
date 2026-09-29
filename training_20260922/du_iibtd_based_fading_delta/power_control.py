"""Discrete RF output power and ideal transmit energy (joules = watts * seconds)."""

from __future__ import annotations

import numpy as np


def dbm_to_watts(power_dbm: float) -> float:
    return float(10.0 ** ((float(power_dbm) - 30.0) / 10.0))


def current_tx_power_dbm(env) -> float:
    return float(getattr(env, "current_tx_power_dbm", env.config.comm.tx_power_dbm))


def validate_power_config(comm) -> None:
    choices = np.asarray(comm.tx_power_choices_dbm, dtype=float)
    if choices.ndim != 1 or not choices.size or not np.all(np.isfinite(choices)):
        raise ValueError("comm.tx_power_choices_dbm must be a nonempty finite list")
    if np.any(np.diff(choices) <= 0) or np.any(np.abs(choices) > 100):
        raise ValueError("comm.tx_power_choices_dbm must increase strictly within [-100, 100]")
    if not np.any(np.isclose(choices, float(comm.tx_power_dbm), rtol=0, atol=1e-9)):
        raise ValueError("comm.tx_power_dbm must be one of comm.tx_power_choices_dbm")
    if not isinstance(comm.tx_energy_enabled, bool):
        raise ValueError("comm.tx_energy_enabled must be a boolean")
    comm.tx_power_choices_dbm = choices.tolist()


def transmit_duration(queue_bits: float, capacity_bps: float, slot_seconds: float) -> float:
    """Stop on queue drain; a nonempty queue attempts the full slot in outage."""
    if queue_bits <= 0.0:
        return 0.0
    if capacity_bps <= 0.0:
        return float(slot_seconds)
    return float(min(slot_seconds, queue_bits / capacity_bps))


class TransmitPowerControl:
    def _init_transmit_power(self) -> None:
        validate_power_config(self.config.comm)
        self.tx_power_choices_dbm = np.asarray(self.config.comm.tx_power_choices_dbm, dtype=float)
        self.tx_power_choices_w = np.asarray([dbm_to_watts(p) for p in self.tx_power_choices_dbm])
        self.num_power_choices = len(self.tx_power_choices_dbm)
        self.default_power_choice_idx = int(np.argmin(abs(self.tx_power_choices_dbm - self.config.comm.tx_power_dbm)))
        self._reset_transmit_power()

    def _reset_transmit_power(self) -> None:
        self._set_transmit_power(self.default_power_choice_idx)
        self.last_uav_tx_duration = 0.0
        self.last_uav_tx_energy = 0.0

    def _set_transmit_power(self, choice_idx: int) -> None:
        if not 0 <= int(choice_idx) < self.num_power_choices:
            raise ValueError(f"Invalid UAV transmit-power choice: {choice_idx}")
        self.current_power_choice_idx = int(choice_idx)
        self.current_tx_power_dbm = float(self.tx_power_choices_dbm[choice_idx])
        self.current_tx_power_w = float(self.tx_power_choices_w[choice_idx])

    def _transmit_energy_bounds(self) -> np.ndarray:
        """Reserve only this step's full-slot cost, before sampling and channel realization."""
        if not self.config.comm.tx_energy_enabled:
            return np.zeros(self.num_power_choices, dtype=float)
        return self.tx_power_choices_w * float(self.config.uav.step_duration)

    def _charge_transmit_energy(self, queue_bits_before_tx: float) -> None:
        self.last_uav_tx_duration = transmit_duration(
            float(queue_bits_before_tx), float(self.ugv_channel_info.capacity_bps),
            float(self.config.uav.step_duration),
        )
        self.last_uav_tx_energy = (
            self.current_tx_power_w * self.last_uav_tx_duration
            if self.config.comm.tx_energy_enabled else 0.0
        )
        self.last_uav_step_energy += self.last_uav_tx_energy
        self.uav_energy -= self.last_uav_tx_energy


def greedy_power_choice(env, direction_idx: int, bw_idx: int, quant_idx) -> int:
    """Choose the lowest power that can clear predicted backlog, else maximum power."""
    if int(getattr(env, "num_power_choices", 1)) == 1:
        return 0
    from .PPO_AStar_2PathSupport.controller import _large_scale_capacity

    landing, _ = env._rollout_direction(
        position=env.uav_pos, direction_idx=direction_idx, step_count=env.uav_step_count,
        validator=env.scene.is_uav_position_valid, stop_at_target=True,
    )
    sensing_units = env.config.uav.sensing_units_for_ratio(env.bandwidth_ratios[bw_idx])
    bits = env.source_measurement_bits if quant_idx is None else int(env.quant_bits[quant_idx])
    produced = sensing_units * float(env.config.comm.data_per_sample) * bits / env.source_measurement_bits
    queue_bits = min(env.queue_capacity_bits, env._queue_remaining_bits() + produced)
    for index, power in enumerate(env.tx_power_choices_dbm):
        capacity = _large_scale_capacity(
            env, tuple(np.rint(landing).astype(int)), tuple(np.rint(env.ugv_pos).astype(int)),
            power_dbm=float(power), comm_units=int(env.config.uav.total_bw_num - sensing_units),
        )
        if capacity * env.config.uav.step_duration >= queue_bits:
            return index
    return env.num_power_choices - 1
