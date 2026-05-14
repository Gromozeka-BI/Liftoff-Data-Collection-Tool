# Эксперимент 11: Реальная Replay-интеграция FPV-камеры

## Статус: ГОТОВИТСЯ К ЗАПУСКУ

---

## 1. Исследовательский вопрос

> Насколько реальная FPV-цепочка камеры (`YOLO pose -> gate association -> PnP
> -> CameraObservation`) улучшает Replay-локализацию DCT по сравнению с тем же
> RC/KF-контуром без камеры? Сколько camera inject'ов и с каким периодом нужно,
> чтобы улучшение было статистически и практически значимым?

Exp 10 отвечал на вопрос **"что даст идеализированная камера с заданной
точностью и частотой"**. Exp 11 проверяет следующий уровень: **что даёт уже
импортированный реальный камерный модуль на записанных Liftoff-видео**.

Второй практический вопрос Exp 11: может ли YOLO/PnP быть не только источником
улучшения локализации, но и **визуальным подтверждением, что аппарат продолжает
лететь по текущей трассе**. Если камера долго не даёт валидных observations и
успешных inject'ов, это может быть признаком потери визуальной связи с трассой:
мы больше не подтверждаем, что дрон видит ожидаемые ворота.

---

## 2. Предшествующие знания

- Exp 10: при `sigma_cam <= 3 м` и частоте `>= 2 Гц` ожидаемый p90 может быть
  `<= 4 м`; при `sigma_cam > 10 м` камеру лучше не подмешивать.
- Exp 10 HP-frontier: слишком частые/слишком точные инжекты могут ломать PF
  бимодальностью, поэтому нужен gate/downweight.
- В Exp 11 уже реализовано:
  - `camera_observations.jsonl` из `video.mp4`;
  - YOLO overlay в Replay Video Preview;
  - отдельный `CamKF` Replay-контур;
  - `RC -> OnlineLocalizer(cam) -> camera inject -> KFLayer2(cam)`;
  - красная вспышка CamKF-стрелки при успешном inject;
  - conservative gate: `innovation_xz <= 15 м`,
    `sigma_eff = max(sigma_cam * 1.5, 4.0 м)`.

---

## 3. Сравниваемые контуры

### 3.1 Базовый контур `KF`

```text
RC-стики -> OnlineLocalizer(rc) -> KFLayer2 -> KF
```

Это текущая красная линия/стрелка в Replay.

### 3.2 Экспериментальный контур `CamKF`

```text
RC-стики -> OnlineLocalizer(cam) -> camera inject -> KFLayer2(cam) -> CamKF
```

Контуры используют один и тот же источник RC, один reference и один
`rate_profile`. Отличие `CamKF` от `KF` должно быть только в пути camera inject.

---

## 4. Входные данные

Папка с тестовыми сессиями:

```text
D:\DroneTrackerDB\Liftoff\Part_5_Эксперементальный
```

Для каждой сессии должны быть:

```text
video.mp4
video_timestamps.parquet
telemetry.parquet
rc_channels.parquet
camera_observations.jsonl
```

`camera_observations.jsonl` генерируется офлайн:

```powershell
python -u tools/exp11_camera_module_import/generate_camera_observations.py `
  "<session_dir>" `
  --every-n-frames 1
```

---

## 5. Дизайн эксперимента

### 5.1 Единица измерения

Основная единица анализа:

```text
session x timestamp
```

Дополнительно агрегировать по:

```text
session
lap
drone
camera inject event
```

### 5.2 Условия

| Условие | Описание |
|---|---|
| `KF` | базовый RC-only контур без camera inject |
| `CamKF` | тот же базовый RC-контур + принятые camera inject |

### 5.3 Фиксированные параметры первого запуска

| Параметр | Значение |
|---|---|
| Трасса | `track-002` |
| Эталон | `GromFF_1.npz` |
| Источник стиков | RC batch |
| Camera observations | `status == "ok"` |
| `max_sigma_cam` | `10 м` |
| `innovation_xz_gate` | `15 м` |
| `sigma_eff` | `max(sigma_cam * 1.5, 4.0 м)` |
| Шаг обработки YOLO-кадров | `1` |
| Таймаут visual watchdog | `60 с` без успешного inject |

### 5.4 Дополнительный sweep после базового анализа

Если первый запуск покажет чувствительность к gate/downweight, выполнить
мини-сетку:

| Фактор | Уровни |
|---|---|
| `innovation_xz_gate` | `10 м`, `15 м`, `20 м`, `off` |
| `sigma_scale` | `1.0`, `1.5`, `2.0` |
| `min_sigma_eff` | `3 м`, `4 м`, `5 м` |

---

## 6. Метрики

### 6.1 Основные метрики локализации

Если в сессии есть ground truth из Liftoff telemetry:

| Метрика | Описание |
|---|---|
| `p90_err_m` | p90 XZ/XYZ ошибки относительно telemetry |
| `median_err_m` | медианная ошибка |
| `mean_err_m` | средняя ошибка |
| `jump_rate_15m` | доля кадров с ошибкой `> 15 м` |
| `max_err_m` | максимальный выброс |

Главное сравнение:

```text
delta_p90 = p90(CamKF) - p90(KF)
improvement_pct = (p90(KF) - p90(CamKF)) / p90(KF)
```

### 6.2 Метрики camera inject

| Метрика | Описание |
|---|---|
| `n_candidates` | число принятых `CameraObservation` до innovation gate |
| `n_injected` | число реально injected observations |
| `n_skipped_innovation` | число наблюдений, отклонённых по innovation gate |
| `inject_rate_hz` | `n_injected / session_duration` |
| `median_inject_period_s` | медианный период между inject |
| `mean_inject_period_s` | средний период между inject |
| `max_inject_gap_s` | максимальная пауза между inject |
| `burstiness` | отношение `mean_period / median_period` |
| `sigma_cam_median` | медианный raw `sigma_cam` |
| `sigma_eff_median` | медианная effective sigma |
| `innovation_xz_median` | медианный innovation до inject |
| `watchdog_timeout_count` | число пауз `> 60 с` без успешного inject |
| `watchdog_timeout_duration_s` | суммарное время в состоянии `visual_lost` |

### 6.3 Метрики downstream-эффекта inject

На каждом `CamKF inject` логировать и анализировать:

```text
pre_xz
post_xz
dxz
pre_sigma
post_sigma
obs_xyz
gate_id
frame_idx
sigma_cam
sigma_eff
innovation_xz
```

Производные метрики:

| Метрика | Описание |
|---|---|
| `inject_dxz_median` | медианный сдвиг PF от camera update |
| `inject_dxz_p90` | p90 сдвига |
| `sigma_reduction_median` | медианное изменение uncertainty |
| `good_inject_rate` | доля inject'ов, после которых ошибка стала меньше |
| `bad_inject_rate` | доля inject'ов, после которых ошибка стала больше |

---

## 7. Гипотезы

| # | Гипотеза | Ожидаемый результат |
|---|---|---|
| H1 | `CamKF` снижает `p90_err_m` относительно `KF` на сессиях с достаточным числом inject'ов | `p90(CamKF) < p90(KF)` минимум на 10-20% |
| H2 | Значимый выигрыш появляется только при `inject_rate >= 0.5-1 Гц` или `max_inject_gap <= 5 с` | сессии с редкими inject'ами не улучшаются |
| H3 | Пачечные inject'ы менее полезны, чем равномерные | высокий `burstiness` коррелирует с меньшим улучшением |
| H4 | Большой `innovation_xz` чаще ухудшает результат | bad inject rate растёт при `innovation_xz > 10-15 м` |
| H5 | Консервативный `sigma_eff` уменьшает скачки, но снижает мгновенный эффект camera update | `inject_dxz_p90` падает, но `p90_err_m` может улучшиться меньше |
| H6 | Отсутствие успешного inject дольше 60 с является полезным watchdog-сигналом потери визуального подтверждения трассы | интервалы `> 60 с` совпадают с участками без видимых/валидных ворот или с потерей трассы |

---

## 8. План выполнения

### Шаг 1. Подготовить observations для всех сессий

Сгенерировать полный `camera_observations.jsonl` (`--every-n-frames 1`) для
каждой сессии в `Part_5_Эксперементальный`.

Проверить summary:

```text
frames_processed
frames_with_detections
accepted
rejected
observations_written
```

### Шаг 2. Replay-прогон и сбор логов

Для каждой сессии:

1. Запустить Replay.
2. Включить `CamKF`.
3. Дать сессии проиграться полностью.
4. Сохранить лог `CamKF inject` / `CamKF inject skipped`.

### Шаг 3. Построить offline-анализ

Скрипт анализа должен собрать:

```text
ground truth из telemetry
траектория KF
траектория CamKF
inject logs
camera_observations.jsonl
```

и записать:

```text
results.csv
inject_events.csv
summary.csv
```

### Шаг 4. Сравнить `KF` и `CamKF`

Основной отчёт:

```text
p90/median/jump_rate по сессиям
взвешенный общий summary
inject rate vs improvement
распределение периодов inject
анализ bad/good inject
timeline visual watchdog
```

### Шаг 4.1. Проверить visual watchdog

По `inject_events.csv` построить интервалы между успешными inject'ами.

Если:

```text
time_since_last_successful_inject > 60 s
```

то пометить состояние:

```text
visual_track_confirmed = False
```

Это не означает автоматически ошибку `KF`: RC/KF может продолжать выдавать
позицию. Но это означает, что камерный модуль больше не подтверждает, что дрон
видит трассу и ожидаемые ворота.

### Шаг 5. Сделать выводы

Ответить:

1. Улучшает ли реальная камерная цепочка локализацию уже сейчас?
2. Сколько inject'ов нужно для значимого эффекта?
3. Какой период inject'ов сейчас фактически получается?
4. Были ли интервалы `> 60 с` без успешного inject, и можно ли использовать это
   как watchdog потери визуального подтверждения трассы?
5. Какие причины мешают улучшению: YOLO recall, keypoint confidence, gate
   association, PnP/sigma, innovation gate?
6. Что нужно доработать перед Record/online-интеграцией?

---

## 9. Выходные файлы

| Файл | Содержимое |
|---|---|
| `spec.md` | план эксперимента |
| `integration_log.md` | журнал реализации Replay-интеграции |
| `session_summary.csv` | summary генерации observations по сессиям |
| `inject_events.csv` | все camera inject / skipped events |
| `results.csv` | per-frame/per-lap метрики `KF` и `CamKF` |
| `summary.csv` | агрегированные метрики по сессиям |
| `report.md` | финальный отчёт |
| `plots/inject_periods.png` | распределение периодов inject |
| `plots/kf_vs_camkf_error.png` | сравнение ошибок |
| `plots/inject_innovation_vs_delta.png` | эффект inject от innovation |
| `plots/visual_watchdog_timeline.png` | интервалы visual confirmed / visual lost |

---

## 10. Критерии успеха

Минимально полезный результат:

```text
CamKF p90 лучше KF хотя бы на 10% на >= 50% сессий
и не увеличивает jump_rate_15m.
```

Хороший результат:

```text
CamKF p90 лучше KF на 20-30%
при inject_rate >= 1 Гц и max_inject_gap <= 5 с.
```

Минимально полезный результат для visual watchdog:

```text
система корректно сообщает visual_lost,
если успешный inject отсутствует > 60 с.
```

Отрицательный, но полезный результат:

```text
CamKF не улучшает p90,
но анализ показывает, что причина в редкости/неравномерности inject'ов
или в ошибочной gate association.
```

Такой результат всё равно полезен: он задаёт требования к следующему этапу
камерного модуля.

---

## 11. Ожидаемые предварительные выводы

На первом полном тесте `session-002` с компромиссным gate:

```text
n_injected = 18
n_skipped = 14
mean_period = 6.8 s
median_period = 0.4 s
max_gap = 27.6 s
```

Это означает, что observations идут пачками и пока не достигают требований
Exp 10 (`>= 1-2 Гц` равномерно). Поэтому главный риск Exp 11 — не качество
Bayes-update как такового, а недостаточная плотность и регулярность пригодных
camera observations.
