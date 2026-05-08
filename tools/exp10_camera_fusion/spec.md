# Эксперимент 10: Слияние с курсовой камерой (симуляция)

## Статус: ГОТОВ К ЗАПУСКУ

---

## 1. Исследовательский вопрос

> Насколько улучшится точность локализатора, если с заданной периодичностью
> вводить абсолютные обновления позиции (эмулируя систему визуальной локализации
> по курсовой камере)? Какие требования к точности и частоте обновлений нужны,
> чтобы улучшение было значимым?

Эксперимент — **переходный**: его результат — формальное ТЗ для следующего
этапа работы (разработка системы визуальной локализации).

---

## 2. Предшествующие знания

- Exp 3: Текущий best: cross-drone p90=7.8 м (sigma=2.0, pnv=8.0, no_thr).
- Текущая система: только стики (RC+Rate) → 1D оценка arc-параметра `s`.
- Камерная система: периодически сообщает `s_obs ≈ s_real + noise(sigma_cam)`.
- В байесовском фильтре частиц добавление наблюдения — стандартная операция.

---

## 3. Модификация OnlineLocalizer

В `OnlineLocalizer` нужно добавить метод:

```python
def inject_position_observation(self, s_obs: float, sigma_obs: float) -> None:
    """
    Inject an absolute position observation from an external sensor.
    s_obs: observed arc position (m)
    sigma_obs: uncertainty of the observation (m)
    """
    # Update particle weights with Gaussian likelihood
    for particle in self.particles:
        d = self._wrap_diff(particle.s, s_obs)  # periodic wrap
        particle.w *= math.exp(-0.5 * (d / sigma_obs) ** 2)
    self._normalize_weights()
    self._resample_if_needed()
```

*Реализацию добавить в `dct/localization/online_localizer.py` перед запуском.*

---

## 4. Факторы и уровни

| Фактор | Уровни | Описание |
|---|---|---|
| **sigma_cam** (точность камеры, м) | 1, 3, 5, 10, 20 | СКО ошибки камерного наблюдения |
| **T_update** (период обновления, с) | 0.2, 0.5, 1.0, 2.0, 5.0 | Как часто камера сообщает позицию |
| **Условие** | same_drone, cross_drone | Как в Exp 1–4 |

**Фиксировано**: RC+Rate, obs_sigma=2.0, pnv=8.0, preset=no_thr, трек track-002.

**Итого**: 5 × 5 × 2 × 3 (лапа) = 150 измерений.

---

## 5. Симуляция наблюдений камеры

В скрипте, на каждом шаге времени `t`:

```python
if t - t_last_camera_update >= T_update:
    # Симуляция: наблюдение = истина + гауссов шум
    s_obs = s_real(t) + np.random.normal(0, sigma_cam)
    loc.inject_position_observation(s_obs, sigma_cam)
    t_last_camera_update = t
```

---

## 6. Гипотезы

| # | Гипотеза | Ожидаемый результат |
|---|---|---|
| H1 | Камера с sigma_cam < 5 м и T_update = 1 с улучшает p90 > 30% | Подтверждено/нет |
| H2 | Существует "насыщение": T_update < 0.5 с не даёт прироста vs 1 с | Платó при высокой частоте |
| H3 | Для cross-drone камерная коррекция важнее, чем для same-drone | Relative improvement(cross) > Relative improvement(same) |
| H4 | При sigma_cam > 10 м камера бесполезна (не улучшает vs baseline) | p90 при sigma_cam=20 ≈ baseline |

---

## 7. Выходные файлы (после запуска)

| Файл | Содержимое |
|---|---|
| `results.csv` | Per-lap метрики для всех (sigma_cam, T_update, condition) |
| `summary.csv` | Агрегация по (sigma_cam, T_update, condition) |
| `camera_requirements.md` | **ТЗ для камерной системы** — главный выход |
| `report.md` | Полный отчёт |
| `plots/heatmap_same_drone.png` | Тепловая карта sigma_cam × T_update для same_drone |
| `plots/heatmap_cross_drone.png` | Тепловая карта sigma_cam × T_update для cross_drone |
| `plots/improvement_vs_baseline.png` | Улучшение % vs baseline без камеры |

---

## 8. Формат ТЗ для камерной системы (camera_requirements.md)

После анализа результатов заполнить:

```markdown
# ТЗ для системы визуальной локализации (камерный модуль)

## Минимальные требования (p90_improvement > 20%)
- Точность позиционирования: sigma_cam <= X м
- Частота обновления: T_update <= Y с

## Рекомендуемые требования (p90_improvement > 50%)
- Точность: sigma_cam <= X м
- Частота: T_update <= Y с

## Ожидаемый прирост при выполнении требований
- same_drone p90: X м → Y м (Z%)
- cross_drone p90: X м → Y м (Z%)
```
