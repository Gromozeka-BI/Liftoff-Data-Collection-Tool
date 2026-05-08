"""
OnlineLocalizer — онлайн-локализация дрона по стикам пульта.

Алгоритм: Particle Filter в 1D-параметризации трассы.
Зависимости: только numpy.

Быстрый старт
-------------
    from dct.localization.online_localizer import OnlineLocalizer

    loc = OnlineLocalizer.from_file("reference.npz")

    # в цикле на каждый фрейм телеметрии:
    result = loc.update(
        sticks=[throttle, yaw, pitch, roll],
        dt=0.01,
    )
    print(result.position_xyz)   # [x, y, z] в метрах
    print(result.progress)       # 0.0 .. 1.0 (доля пройденного круга)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dct.rate_features import (
    FEATURE_BETAFLIGHT_CLASSIC_V1,
    physical_observation_matrix,
    physical_observation_row,
)


# ---------------------------------------------------------------------------
# Структура результата
# ---------------------------------------------------------------------------

@dataclass
class LocalizerResult:
    """Результат одного шага локализации."""

    position_xyz: np.ndarray
    """Оценка позиции дрона [x, y, z] в метрах (система координат трассы)."""

    s: float
    """Пройденное расстояние по дуге трассы в метрах (0 .. track_length)."""

    progress: float
    """Доля пройденного круга: 0.0 = старт, 1.0 = финиш."""

    uncertainty_m: float
    """Оценка неопределённости позиции в метрах (стандартное отклонение частиц)."""

    track_length: float
    """Полная длина эталонного круга в метрах."""


# ---------------------------------------------------------------------------
# Эталон трассы
# ---------------------------------------------------------------------------

class Reference:
    """Эталонный круг: нормализованные наблюдения + позиции вдоль дуги.

    ``sticks_norm`` stores z-scored observations (legacy: smoothed sticks;
    ``feature_kind=betaflight_classic_rpy_deg_s_v1``: smoothed
    [thr, yaw°, pitch°, roll°] per row).
    """

    def __init__(
        self,
        s: np.ndarray,
        pos: np.ndarray,
        sticks_norm: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        smooth_w: int = 5,
        *,
        feature_kind: str | None = None,
        rate_profile: dict | None = None,
    ):
        self.s = s                       # (N,) — дуговой параметр, м
        self.pos = pos                   # (N, 3) — xyz в метрах
        self.sticks_norm = sticks_norm   # (N, 4) — z-scored observations
        self.mean = mean                 # (4,) — mean before z-score
        self.std = std                   # (4,) — std before z-score
        self.smooth_w = smooth_w
        self.L = float(s[-1])            # длина круга в метрах
        self.feature_kind = feature_kind
        self.rate_profile = rate_profile

    # ------------------------------------------------------------------
    def pos_at_s(self, s_query: float | np.ndarray) -> np.ndarray:
        """Интерполировать xyz по дуговому параметру s."""
        s_query = np.asarray(s_query, dtype=float)
        s_clip = np.clip(s_query, 0.0, self.L)
        x = np.interp(s_clip, self.s, self.pos[:, 0])
        y = np.interp(s_clip, self.s, self.pos[:, 1])
        z = np.interp(s_clip, self.s, self.pos[:, 2])
        if s_query.ndim == 0:
            return np.array([x, y, z])
        return np.stack([x, y, z], axis=-1)

    # ------------------------------------------------------------------
    def normalize_sticks(self, sticks: np.ndarray) -> np.ndarray:
        """Z-score observations the same way as during :class:`Reference` build.

        For ``feature_kind=betaflight_classic_rpy_deg_s_v1`` the input is still
        raw ``[thr, yaw, pitch, roll]`` sticks in -1..1; they are mapped through
        the Betaflight curve, box-smoothed, then z-scored.
        """
        x = np.atleast_2d(sticks)
        if self.feature_kind == FEATURE_BETAFLIGHT_CLASSIC_V1:
            if not self.rate_profile:
                raise RuntimeError("rate_profile required for Betaflight feature mode")
            obs = physical_observation_matrix(x, self.rate_profile)
            sm = _smooth_box(obs, self.smooth_w)
            return (sm - self.mean) / self.std
        sm = _smooth_box(x, self.smooth_w)
        return (sm - self.mean) / self.std

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Сохранить эталон в .npz-файл."""
        payload: dict[str, np.ndarray] = {
            "s": self.s,
            "pos": self.pos,
            "sticks_norm": self.sticks_norm,
            "mean": self.mean,
            "std": self.std,
            "smooth_w": np.array(self.smooth_w),
        }
        if self.feature_kind:
            payload["feature_kind"] = np.asarray(self.feature_kind)
        if self.rate_profile:
            jb = json.dumps(self.rate_profile, ensure_ascii=False).encode("utf-8")
            payload["rate_profile_json"] = np.frombuffer(bytearray(jb), dtype=np.uint8)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "Reference":
        """Загрузить эталон из .npz-файла."""
        d = np.load(path, allow_pickle=False)
        feature_kind: str | None = None
        if "feature_kind" in d.files:
            fk = d["feature_kind"]
            s = fk.item() if fk.ndim == 0 else str(fk.flat[0])
            if isinstance(s, bytes):
                s = s.decode("utf-8", errors="replace")
            s = str(s).strip()
            if s:
                feature_kind = s
        rate_profile: dict | None = None
        if "rate_profile_json" in d.files:
            arr = d["rate_profile_json"]
            rate_profile = json.loads(arr.tobytes().decode("utf-8"))
        return cls(
            s=d["s"],
            pos=d["pos"],
            sticks_norm=d["sticks_norm"],
            mean=d["mean"],
            std=d["std"],
            smooth_w=int(d["smooth_w"]),
            feature_kind=feature_kind,
            rate_profile=rate_profile,
        )

    @classmethod
    def build(
        cls,
        t: np.ndarray,
        sticks: np.ndarray,
        pos: np.ndarray,
        smooth_w: int = 5,
    ) -> "Reference":
        """Построить эталон из массивов временного ряда одного круга.

        Parameters
        ----------
        t       : (N,) временные метки, сек
        sticks  : (N, 4) стики [throttle, yaw, pitch, roll] в диапазоне -1..1
        pos     : (N, 3) позиции [x, y, z] в метрах
        smooth_w: ширина окна сглаживания (нечётное число)
        """
        sticks_sm = _smooth_box(sticks, smooth_w)
        mean = sticks_sm.mean(axis=0)
        std = sticks_sm.std(axis=0) + 1e-6
        sticks_norm = (sticks_sm - mean) / std

        deltas = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(deltas)])

        return cls(
            s=s, pos=pos, sticks_norm=sticks_norm,
            mean=mean, std=std, smooth_w=smooth_w,
            feature_kind=None,
            rate_profile=None,
        )

    @classmethod
    def build_from_features(
        cls,
        t: np.ndarray,
        obs: np.ndarray,
        pos: np.ndarray,
        smooth_w: int = 5,
        *,
        feature_kind: str,
        rate_profile: dict | None,
    ) -> "Reference":
        """Построить эталон из физических наблюдений (N,4), затем box-smooth + z-score."""
        obs_sm = _smooth_box(np.atleast_2d(obs), smooth_w)
        mean = obs_sm.mean(axis=0)
        std = obs_sm.std(axis=0) + 1e-6
        sticks_norm = (obs_sm - mean) / std

        deltas = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(deltas)])

        return cls(
            s=s, pos=pos, sticks_norm=sticks_norm,
            mean=mean, std=std, smooth_w=smooth_w,
            feature_kind=feature_kind,
            rate_profile=rate_profile,
        )


# ---------------------------------------------------------------------------
# Particle Filter
# ---------------------------------------------------------------------------

class ParticleFilter:
    """1D Particle Filter по дуговому параметру трассы.

    Состояние каждой частицы: (s, v) — позиция на трассе (м) и скорость (м/с).
    """

    def __init__(
        self,
        ref: Reference,
        n_particles: int = 1000,
        v_init_mps: float = 10.0,
        v_init_std: float = 3.0,
        v_min: float = 0.5,
        v_max: float = 30.0,
        obs_sigma: float = 2.0,
        process_noise_s: float = 1.5,
        process_noise_v: float = 3.0,
        ess_threshold: float = 0.5,
        roughening_s: float = 0.2,
        random_inject_frac: float = 0.02,
        seed: int = 42,
        channel_weights: list[float] | np.ndarray | None = None,
    ):
        self.ref = ref
        self.N = n_particles
        self.v_init = v_init_mps
        self.v_init_std = v_init_std
        self.v_min = v_min
        self.v_max = v_max
        n_ch = ref.sticks_norm.shape[1]
        if channel_weights is not None:
            self._ch_w = np.asarray(channel_weights, dtype=np.float64)
            if self._ch_w.shape != (n_ch,):
                raise ValueError(
                    f"channel_weights must have length {n_ch}, got {self._ch_w.shape}"
                )
            # Scale obs_sigma2_eff by the number of active (non-zero) channels so
            # that the likelihood bandwidth stays consistent regardless of how many
            # channels are used.
            n_active = float(np.count_nonzero(self._ch_w))
            self.obs_sigma2_eff = obs_sigma ** 2 * max(n_active, 1.0)
        else:
            self._ch_w = None
            self.obs_sigma2_eff = obs_sigma ** 2 * n_ch
        self.q_s = process_noise_s
        self.q_v = process_noise_v
        self.ess_thr = ess_threshold
        self.roughening_s = roughening_s
        self.random_inject_frac = random_inject_frac
        self.rng = np.random.default_rng(seed)

        self._s: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._w: np.ndarray | None = None

    def reset(self) -> None:
        """Сбросить фильтр в начальное состояние (старт/финиш трассы)."""
        half = self.N // 2
        s0 = np.abs(self.rng.normal(0, 3.0, half))
        s1 = self.ref.L - np.abs(self.rng.normal(0, 3.0, self.N - half))
        self._s = np.mod(np.concatenate([s0, s1]), self.ref.L)
        self._v = np.clip(
            self.rng.normal(self.v_init, self.v_init_std, self.N),
            self.v_min, self.v_max,
        )
        self._w = np.ones(self.N) / self.N

    def update(self, stick_norm: np.ndarray, dt: float | None) -> tuple[float, float]:
        """Один шаг фильтра.

        Parameters
        ----------
        stick_norm : (4,) нормализованный вектор стиков
        dt         : время с предыдущего шага, сек (None для первого шага)

        Returns
        -------
        s_est     : оценка позиции на трассе, м
        sigma     : неопределённость, м
        """
        if self._s is None:
            self.reset()

        # --- предсказание ---
        if dt is not None and dt > 0:
            self._v += self.rng.normal(0, self.q_v * np.sqrt(dt), self.N)
            self._v = np.clip(self._v, self.v_min, self.v_max)
            self._s += self._v * dt + self.rng.normal(0, self.q_s * np.sqrt(dt), self.N)
            self._s = np.mod(self._s, self.ref.L)

        # --- обновление весов по наблюдению ---
        idx = np.searchsorted(self.ref.s, self._s).clip(0, len(self.ref.s) - 1)
        feats = self.ref.sticks_norm[idx]
        diff2 = (feats - stick_norm[None, :]) ** 2
        if self._ch_w is not None:
            diff2 = diff2 * self._ch_w[None, :]
        d2 = np.sum(diff2, axis=1)
        log_w = np.log(self._w + 1e-30) - 0.5 * d2 / self.obs_sigma2_eff
        log_w -= log_w.max()
        self._w = np.exp(log_w)
        self._w /= self._w.sum() + 1e-12

        # --- ресемплинг при вырождении ---
        ess = 1.0 / (np.sum(self._w ** 2) + 1e-12)
        if ess < self.ess_thr * self.N:
            positions = (self.rng.uniform(0, 1) + np.arange(self.N)) / self.N
            cumw = np.cumsum(self._w)
            cumw[-1] = 1.0
            idx_r = np.clip(np.searchsorted(cumw, positions), 0, self.N - 1)
            self._s = self._s[idx_r] + self.rng.normal(0, self.roughening_s, self.N)
            self._s = np.mod(self._s, self.ref.L)
            self._v = self._v[idx_r]
            if self.random_inject_frac > 0:
                n_inj = max(1, int(self.N * self.random_inject_frac))
                inj = self.rng.choice(self.N, n_inj, replace=False)
                self._s[inj] = self.rng.uniform(0, self.ref.L, n_inj)
            self._w = np.ones(self.N) / self.N

        # --- оценка позиции (circular mean) ---
        theta = 2 * np.pi * self._s / self.ref.L
        cx = float(np.sum(self._w * np.cos(theta)))
        cy = float(np.sum(self._w * np.sin(theta)))
        ang = np.arctan2(cy, cx)
        if ang < 0:
            ang += 2 * np.pi
        s_est = float(ang * self.ref.L / (2 * np.pi))

        R = np.sqrt(cx ** 2 + cy ** 2)
        sigma = float(np.sqrt(max(1e-6, 1.0 - R)) * self.ref.L / (2 * np.pi))
        return s_est, sigma


# ---------------------------------------------------------------------------
# Публичный класс-обёртка
# ---------------------------------------------------------------------------

class OnlineLocalizer:
    """Онлайн-локализатор дрона на трассе по стикам пульта.

    Пример использования
    --------------------
        loc = OnlineLocalizer.from_file("reference.npz")
        loc.reset()

        prev_ts = None
        for ts, throttle, yaw, pitch, roll in telemetry_stream:
            dt = (ts - prev_ts) if prev_ts is not None else None
            result = loc.update([throttle, yaw, pitch, roll], dt=dt)
            prev_ts = ts

            print(f"Позиция: {result.position_xyz}  Прогресс: {result.progress:.1%}")
    """

    def __init__(self, ref: Reference, **pf_kwargs):
        """
        Parameters
        ----------
        ref       : эталонный круг (Reference)
        pf_kwargs : параметры Particle Filter (см. ParticleFilter.__init__)
        """
        self.ref = ref
        self._pf = ParticleFilter(ref, **pf_kwargs)
        self._prev_sticks_buffer: list[np.ndarray] = []
        self._initialized = False

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path, **pf_kwargs) -> "OnlineLocalizer":
        """Создать локализатор из сохранённого .npz-файла эталона.

        Convenience: pass ``use_throttle=False`` to zero-out the throttle
        channel (index 0) in the distance metric.  Useful when testing
        cross-drone generalization where throttle levels differ between drones.
        """
        use_throttle: bool = pf_kwargs.pop("use_throttle", True)
        ref = Reference.load(path)
        if not use_throttle:
            n_ch = ref.sticks_norm.shape[1]
            weights = np.ones(n_ch, dtype=np.float64)
            weights[0] = 0.0  # disable throttle channel
            pf_kwargs.setdefault("channel_weights", weights)
        return cls(ref, **pf_kwargs)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Сбросить состояние фильтра (например, при старте нового круга)."""
        self._pf.reset()
        self._prev_sticks_buffer = []
        self._initialized = True

    # ------------------------------------------------------------------
    def update(
        self,
        sticks: list[float] | np.ndarray,
        dt: float | None,
        rate_profile: dict | None = None,
    ) -> LocalizerResult:
        """Обновить оценку позиции по новому вектору стиков.

        Parameters
        ----------
        sticks       : [throttle, yaw, pitch, roll] в диапазоне -1..1
        dt           : время с предыдущего вызова update(), сек.
                       Передать None для первого вызова.
        rate_profile : rate profile **текущей сессии** (dict с model="betaflight").
                       Если передан — используется для sticks→setpoint вместо
                       rate profile из референса.  Это позволяет корректно
                       сравнивать сессии с разными настройками рейтов: один и
                       тот же манёвр даёт одинаковый setpoint (deg/s) независимо
                       от rate profile, поэтому нормализатор из референса
                       остаётся применимым.
                       Если None — используется rate profile из референса
                       (поведение по умолчанию, обратная совместимость).

        Returns
        -------
        LocalizerResult с полями position_xyz, s, progress, uncertainty_m, track_length
        """
        if not self._initialized:
            self.reset()

        sticks = np.asarray(sticks, dtype=float)

        if self.ref.feature_kind == FEATURE_BETAFLIGHT_CLASSIC_V1:
            # Use current-session rate profile when provided so that the
            # same physical manoeuvre produces the same setpoint (deg/s)
            # regardless of rate settings, making the reference reusable
            # across different rate configurations.
            active_prof = rate_profile if rate_profile is not None else self.ref.rate_profile
            if not active_prof:
                raise RuntimeError(
                    "Reference uses Betaflight feature mode but no rate_profile is available "
                    "(neither from the reference .npz nor passed to update())",
                )
            obs_row = physical_observation_row(sticks, active_prof)
            self._prev_sticks_buffer.append(obs_row)
        else:
            self._prev_sticks_buffer.append(sticks)

        w = self.ref.smooth_w
        if len(self._prev_sticks_buffer) > w:
            self._prev_sticks_buffer.pop(0)

        buf = np.array(self._prev_sticks_buffer)
        smoothed = buf.mean(axis=0)
        stick_norm = (smoothed - self.ref.mean) / self.ref.std

        s_est, sigma = self._pf.update(stick_norm, dt)
        xyz = self.ref.pos_at_s(s_est)

        return LocalizerResult(
            position_xyz=xyz,
            s=s_est,
            progress=s_est / self.ref.L,
            uncertainty_m=sigma,
            track_length=self.ref.L,
        )


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _smooth_box(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    padded = np.concatenate([
        np.repeat(x[:1], pad, axis=0),
        x,
        np.repeat(x[-1:], pad, axis=0),
    ], axis=0)
    kernel = np.ones(w) / w
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kernel, mode="valid")[:len(x)]
    return out
