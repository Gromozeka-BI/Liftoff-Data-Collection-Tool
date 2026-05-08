# Эксперимент 6: Стратегия построения референса

## Статус: ГОТОВ К ЗАПУСКУ

---

## 1. Исследовательский вопрос

> Как выбор лапа, количество лапов и степень сглаживания при построении
> референса влияют на точность локализатора?

Эксперимент 4 показал, что **качество референса важнее типа дрона**. Данный
эксперимент раскрывает, что именно определяет "качество" референса и как
его максимизировать.

---

## 2. Предшествующие знания

- Exp 4: GromFF_1 (1 хороший лап, smooth_w=5) даёт p90=5.8 м (same-drone) и 9.7 м (cross-drone).
- LiftOff200_Grom_1 (1 лап, smooth_w=1) — хуже: p90=16.2 м (same-drone). Разница — качество лапа и/или сглаживание.
- Текущий метаданные референсов: Grom_4 (smooth_w=5), Grom_5 (smooth_w=5), RedRC_2 (smooth_w=1), RedRC_3 (smooth_w=9).

---

## 3. Факторы и уровни

| Фактор | Уровни | Описание |
|---|---|---|
| **Выбор лапа** | fastest, median_time, manual_best | Какой лап брать как референс |
| **smooth_w** | 1, 3, 5, 9, 15 | Параметр сглаживания при построении |
| **Число лапов** | 1, 3, 5 | Усреднение по N лапам перед построением |
| Условие | same_drone, cross_drone | Как в Exp 1–4 |

**Фиксировано**: режим RC+Rate, obs_sigma=2.0, pnv=8.0, preset=no_thr, трек track-002.

---

## 4. Гипотезы

| # | Гипотеза | Ожидаемый результат |
|---|---|---|
| H1 | Большее сглаживание (smooth_w↑) улучшает cross-drone точность | p90 монотонно убывает с ростом smooth_w |
| H2 | Усреднение по 3+ лапам лучше 1 лапа | p90(3 лапа) < p90(1 лап) |
| H3 | Выбор "fastest" лапа хуже "median" (outlier effect) | p90(fastest) > p90(median) |
| H4 | Оптимальный smooth_w не зависит от типа дрона (same vs cross) | sigma_opt_same ≈ sigma_opt_cross |

---

## 5. Что нужно реализовать

### 5.1 Построение референсов с разными параметрами

Для каждой комбинации (выбор лапа, smooth_w, число лапов) нужно построить
референс программно — через Python API DCT, без GUI.

Пример (псевдокод):
```python
from dct.localization.reference_builder import build_reference
ref = build_reference(
    session_dir=session_dir,
    lap_selection="median",  # "fastest" | "median" | index
    n_laps=1,               # усреднение по N лапам
    smooth_w=5,
)
```

### 5.2 Проверить наличие Python API

Перед написанием скрипта: найти функцию построения референса в DCT codebase.

---

## 6. Метрики

- Первичная: `p90_err_m` (same_drone и cross_drone)
- Вторичная: `generalization_gap = p90_cross - p90_same`

---

## 7. Выходные файлы (после запуска)

| Файл | Содержимое |
|---|---|
| `results.csv` | Per-lap метрики |
| `summary.csv` | По (lap_selection, smooth_w, n_laps, condition) |
| `report.md` | Отчёт |
| `plots/smooth_effect.png` | p90 vs smooth_w |
| `plots/nlaps_effect.png` | p90 vs n_laps |
| `plots/lap_selection.png` | Сравнение стратегий выбора лапа |
