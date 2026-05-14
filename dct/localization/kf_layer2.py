"""KF второго контура: сглаживание выхода Layer 1 (Particle Filter)
с использованием скоростного профиля эталона.

Архитектура:
  Состояние:    x = [s, v]   (дистанция вдоль дуги, м; скорость, м/с)
  Prediction:   s' = s + v*dt
                v' = v              (случайное блуждание, шум sigma_v)
  Update L1:    z  = result.s       (дистанция от Layer 1 PF)
                R  = result.uncertainty_m^2
  Псевдо-изм.:  z_v = v_ref(s')    (мягкий аттрактор к эталонной скорости)
                R_v = sigma_v_pseudo^2

Зависимости: numpy, scipy.interpolate
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from dct.localization.online_localizer import LocalizerResult, Reference

# Предполагаемый шаг эталона (Liftoff UDP ≈ 100 Гц).
# Используется для вычисления скоростного профиля из ref.s,
# если временно́й массив недоступен.
_DT_REF = 1.0 / 100.0


# ─── Kalman Filter ────────────────────────────────────────────────────────────

class _SpeedProfileKF:
    """Линейный фильтр Калмана с состоянием [s, v] и мягким аттрактором
    скорости к эталонному профилю v_ref(s).
    """

    def __init__(
        self,
        sigma_v: float,
        sigma_v_pseudo: float,
        v_ref_fn,
        track_length: float,
    ) -> None:
        self.sigma_v        = sigma_v
        self.sigma_v_pseudo = sigma_v_pseudo
        self.v_ref_fn       = v_ref_fn
        self.L              = track_length
        self.initialized    = False

        self.x = np.zeros(2)       # [s, v]
        self.P = np.eye(2) * 1e6   # большая начальная неопределённость

    def initialize(self, s0: float, q0: float) -> None:
        self.x = np.array([s0, float(self.v_ref_fn(s0))])
        self.P = np.diag([q0 ** 2, self.sigma_v ** 2])
        self.initialized = True

    @staticmethod
    def _wrap(delta: float, L: float) -> float:
        """Невязка с учётом цикличности трассы."""
        d = delta % L
        if d > L / 2:
            d -= L
        return d

    def predict(self, dt: float) -> None:
        s, v = self.x
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.diag([0.0, (self.sigma_v * dt) ** 2])
        self.x    = F @ self.x
        self.x[0] = self.x[0] % self.L
        self.P    = F @ self.P @ F.T + Q

    def update_layer1(self, z_s: float, q: float) -> None:
        H = np.array([[1.0, 0.0]])
        R = np.array([[q ** 2]])
        innovation = self._wrap(z_s - self.x[0], self.L)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x    = self.x + K.flatten() * innovation
        self.x[0] = self.x[0] % self.L
        self.P    = (np.eye(2) - K @ H) @ self.P

    def update_velocity_prior(self) -> None:
        H_v = np.array([[0.0, 1.0]])
        R_v = np.array([[self.sigma_v_pseudo ** 2]])
        z_v         = float(self.v_ref_fn(self.x[0]))
        innovation_v = z_v - self.x[1]
        S_v = H_v @ self.P @ H_v.T + R_v
        K_v = self.P @ H_v.T @ np.linalg.inv(S_v)
        self.x    = self.x + K_v.flatten() * innovation_v
        self.x[0] = self.x[0] % self.L
        self.P    = (np.eye(2) - K_v @ H_v) @ self.P

    @property
    def s(self) -> float:
        return float(self.x[0])

    @property
    def v(self) -> float:
        return float(self.x[1])

    @property
    def s_std(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], 0.0)))


# ─── Публичный класс ──────────────────────────────────────────────────────────

class KFLayer2:
    """KF второго контура для онлайн-использования поверх Layer 1 (RC PF).

    Принимает :class:`LocalizerResult` от RC-локализатора (Layer 1),
    возвращает новый :class:`LocalizerResult` с KF-сглаженной позицией.

    Пример использования
    --------------------
        kf = KFLayer2(ref)
        kf.reset()

        prev_ts = None
        for result in rc_localizer_stream:
            dt = ...
            result_kf = kf.update(result, dt)
            print(result_kf.position_xyz, result_kf.uncertainty_m)

    Параметры
    ---------
    ref             : эталонный круг (Reference) того же RC-локализатора
    sigma_v         : шум процесса на скорость (м/с); оптимум ≈ 2.0 для
                      того же пилота, 6.0 для другого пилота
    sigma_v_pseudo  : жёсткость аттрактора v → v_ref (обычно = sigma_v)
    init_q_thresh   : порог uncertainty_m Layer 1 для инициализации KF (м)
    """

    def __init__(
        self,
        ref: Reference,
        sigma_v: float = 2.0,
        sigma_v_pseudo: float = 2.0,
        init_q_thresh: float = 10.0,
    ) -> None:
        self.ref             = ref
        self.init_q_thresh   = init_q_thresh
        self._v_ref_fn       = self._build_v_ref(ref)
        self._kf             = _SpeedProfileKF(
            sigma_v, sigma_v_pseudo, self._v_ref_fn, ref.L,
        )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_v_ref(ref: Reference):
        """Строит интерполятор v_ref(s) из ref.s, предполагая шаг ~100 Гц."""
        ds    = np.diff(ref.s)
        v_arr = ds / _DT_REF
        s_mid = (ref.s[:-1] + ref.s[1:]) / 2
        return interp1d(
            s_mid, v_arr,
            kind="linear",
            bounds_error=False,
            fill_value=(v_arr[0], v_arr[-1]),
        )

    # ── public API ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Сбросить состояние фильтра."""
        self._kf = _SpeedProfileKF(
            self._kf.sigma_v,
            self._kf.sigma_v_pseudo,
            self._v_ref_fn,
            self.ref.L,
        )

    def update(
        self,
        result: LocalizerResult,
        dt: float | None,
    ) -> LocalizerResult:
        """Один шаг KF поверх результата Layer 1.

        Parameters
        ----------
        result : LocalizerResult от RC-локализатора (Layer 1)
        dt     : время с предыдущего вызова, сек; None для первого шага

        Returns
        -------
        LocalizerResult с KF-сглаженной позицией и uncertainty_m = s_std KF
        """
        kf = self._kf
        z_s = result.s
        q   = result.uncertainty_m

        if not kf.initialized:
            if q < self.init_q_thresh:
                kf.initialize(s0=z_s, q0=q)
            # До инициализации — возвращаем результат Layer 1 без изменений
            return result

        # Предсказание
        if dt is not None and 0 < dt <= 2.0:
            kf.predict(dt)

        # Обновление от Layer 1 (всегда, т.к. проверки XYZ-проекции нет)
        kf.update_layer1(z_s=z_s, q=q)

        # Псевдо-измерение скорости
        kf.update_velocity_prior()

        xyz = self.ref.pos_at_s(kf.s)
        return LocalizerResult(
            position_xyz=xyz,
            s=kf.s,
            progress=kf.s / self.ref.L,
            uncertainty_m=kf.s_std,
            track_length=self.ref.L,
        )

    @property
    def initialized(self) -> bool:
        """True после первого схождения Layer 1."""
        return self._kf.initialized
