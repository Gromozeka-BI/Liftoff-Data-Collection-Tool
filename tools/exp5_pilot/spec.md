# Эксперимент 5: Кросс-пилотная обобщаемость

## Статус: ОЖИДАЕТ ДАННЫХ

Необходимо записать полётные данные второго пилота.
См. `data_collection_spec.md` — подробное ТЗ для пилота.

---

## 1. Цель и мотивация

Эксперименты 1–4 показали, что тип дрона — доминирующий фактор точности
(eta2=0.41), однако при RC+Rate нормализации кросс-дроновая p90 достигает
7.8–9.7 м (Экс. 3–4). Главным фактором оказалось **качество референса**.

Следующий нераскрытый вопрос:
> *Работает ли система, если референс построен одним пилотом, а тестируемые
> полёты выполнены другим? Насколько стиль пилотирования влияет на точность
> локализации?*

Это последний шаг в иерархии обобщаемости:
**Дрон → Rate profile → Пилот**

---

## 2. Необходимые данные

### 2.1 Что уже есть (Пилот 1 — Gromozeka, track-002)

| ID | Сессия | Дрон | Rate | Статус |
|---|---|---|---|---|
| G-F1 | session-001 | MadTrainer | Gromozeka_rate | ✅ готово |
| G-F2 | session-002 | MadTrainer | RedSheep_rate | ✅ готово |
| G-F3 | session-003 | LiftOff_200 | Gromozeka_rate | ✅ готово |
| G-F4 | session-004 | LiftOff_200 | RedSheep_rate | ✅ готово |

Референсы: `GromFF_1.npz`, `LiftOff200_Grom_1.npz`, `LiftOff200_Red_1.npz`

### 2.2 Что нужно записать (Пилот 2 — <Callsign>)

| ID | Дрон | Rate | Минимум лапов | Приоритет |
|---|---|---|---|---|
| P2-S1 | MadTrainer | Gromozeka_rate | 12 | ⭐ Обязательно |
| P2-S2 | MadTrainer | Gromozeka_rate | 10 | ⭐ Обязательно |
| P2-S3 | MadTrainer | Own rate | 10 | ⭐ Обязательно |
| P2-S4 | MadTrainer | Own rate | 10 | ⭐ Обязательно |
| P2-S5 | LiftOff_200 | Gromozeka_rate | 8 | ✨ Желательно |
| P2-S6 | LiftOff_200 | Own rate | 8 | ✨ Желательно |

После записи — создать в DCT GUI референсы:
- `<Callsign>_1.npz` (из P2-S1)
- `<Callsign>_2.npz` (из P2-S2, для проверки воспроизводимости)

---

## 3. Экспериментальный план

**Трек**: track-002  
**Режим**: RC+Rate (оптимальный из Эксп. 1)  
**Гиперпараметры**: obs_sigma=2.0, pnv=8.0 (оптимальные из Эксп. 3)

### 3.1 Матрица условий

| Референс | Тест | pilot_match | drone_match | rate_match | Условие |
|---|---|---|---|---|---|
| GromFF_1 | G-F1 | ✅ same | ✅ same | ✅ same | same_pilot_same_drone_same_rate |
| GromFF_1 | G-F2 | ✅ same | ✅ same | ❌ cross | same_pilot_same_drone_cross_rate |
| GromFF_1 | P2-S1 | ❌ cross | ✅ same | ✅ same | **cross_pilot_same_drone_same_rate** |
| GromFF_1 | P2-S3 | ❌ cross | ✅ same | ❌ cross | **cross_pilot_same_drone_cross_rate** |
| Callsign_1 | P2-S1 | ✅ same | ✅ same | ✅ same | **same_pilot_ceiling** |
| Callsign_1 | P2-S3 | ✅ same | ✅ same | ❌ cross | **same_pilot_cross_rate** |
| Callsign_1 | G-F1 | ❌ cross | ✅ same | ❌ cross | **cross_pilot_reversed** |
| GromFF_1 | P2-S5 | ❌ cross | ❌ cross | ✅ same | cross_pilot_cross_drone |
| Callsign_1 | G-F3 | ❌ cross | ❌ cross | ✅ same | cross_pilot_cross_drone_rev |

### 3.2 Ключевые вопросы для анализа

1. **p90(GromFF_1 → P2) vs p90(Callsign_1 → P2)**
   Насколько чужой референс хуже своего (кросс-пилотный разрыв)?

2. **p90(кросс-пилот) vs p90(кросс-дрон, Эксп. 4)**
   Что хуже — другой пилот или другой дрон?

3. **Симметрия**: GromFF_1→P2 ≈ Callsign_1→G ?
   Является ли качество референса симметричным между пилотами?

4. **Влияние own rate пилота на cross_pilot**: насколько rate-mismatch
   дополнительно ухудшает кросс-пилотную точность?

### 3.3 Гипотезы

| # | Гипотеза | Ожидаемый результат |
|---|---|---|
| H1 | Кросс-пилотная p90 < кросс-дроновой (из Эксп. 4) | p90(cross_pilot) < 9.7 м |
| H2 | RC+Rate нормализует пилотные различия лучше, чем дроновые | Подтверждено или нет |
| H3 | same_pilot_ceiling < same_drone_ceiling из Эксп. 4 | p90 < 5.8 м |
| H4 | Кросс-пилотный разрыв симметричен | |G→P2 - P2→G| < 30% |

---

## 4. Пресеты весов и гиперпараметры

| Параметр | Значение | Источник |
|---|---|---|
| obs_sigma | 2.0 | Оптимум Эксп. 3 |
| process_noise_v | 8.0 | Оптимум Эксп. 3 |
| process_noise_s | 1.5 | Фиксировано |
| baseline | [1.0, 1.0, 1.0, 1.0] | Эксп. 1 default |
| angular_scaled | [0.0, 0.7, 0.5, 1.0] | Ранг-1 Эксп. 2 |
| no_thr | [0.0, 1.0, 1.0, 1.0] | Best cross-drone Эксп. 3 |

---

## 5. Файлы, которые появятся после запуска

| Файл | Содержимое |
|---|---|
| `results/results.csv` | Per-lap метрики |
| `results/summary.csv` | Per (ref, flight, preset, condition) |
| `results/condition_summary.csv` | По типу условия |
| `plots/condition_comparison.png` | p90 по 4 типам условий |
| `plots/pilot_comparison.png` | Same-pilot vs cross-pilot vs cross-drone |
| `plots/full_matrix.png` | Матрица ref × test |
| `report.md` | Финальный отчёт |

---

## 6. Как запустить после получения данных

1. Добавить пути к сессиям Пилота 2 в `experiment_pilot.py`
2. Добавить референсы в `tracks/track-002/references/`
3. Запустить: `python tools/experiment_pilot.py`

*Скрипт будет написан по образцу `experiment_reference.py`.*
