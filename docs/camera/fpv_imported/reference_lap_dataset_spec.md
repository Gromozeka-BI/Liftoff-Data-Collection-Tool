# ТЗ: оценочный круг для проверки OnlineLocalizer, KFLayer2 и визуальной локализации

## 1. Цель

Подготовить один синхронизированный круг полета, по которому можно оценить
качество:

- `OnlineLocalizer` до второго слоя;
- `KFLayer2` после второго слоя;
- визуальной локализации по камере;
- fusion-схемы `camera XYZ -> inject_position_observation -> KFLayer2`.

Этот документ предназначен как ТЗ для соседнего проекта, который готовит
reference/evaluation данные.

## 2. Важное условие

Если один и тот же круг используется и для построения `Reference`, и для оценки
качества, результат считается только sanity check.

Для честной оценки желательно иметь два круга:

```text
reference lap -> используется для построения DCT Reference
eval lap      -> используется для разметки кадров и оценки ошибок
```

Минимально допустимо начать с одного круга, но в отчете нужно явно пометить его
как `sanity validation`, а не как полноценную проверку обобщения.

## 3. Что нужно собрать

Для каждого выбранного кадра или timestamp нужны:

- изображение кадра;
- timestamp изображения;
- timestamp телеметрии;
- истинная позиция дрона `gt_xyz = [x, y, z]`;
- истинный yaw `gt_yaw`, если доступен;
- 2D-разметка видимых ворот;
- правильные `gate_id` для размеченных ворот;
- результат `OnlineLocalizer` до второго слоя;
- результат `KFLayer2` после второго слоя;
- при тесте камеры: результат `FPVCamDetectV2/PnP`;
- при тесте fusion: результат после `inject_position_observation()` и финальный
  результат после `KFLayer2`.

Все координаты должны быть в метрах. `gt_xyz`, `online_xyz`, `kf_xyz`,
`vision_xyz`, `fused_xyz_before_kf` и `final_xyz_after_kf` должны быть приведены
к одной системе координат.

## 4. Рекомендуемый формат записи

Рекомендуемый формат - `jsonl`: одна строка на один кадр. Можно использовать и
обычный `json`, если удобнее хранить массив записей.

Минимальная структура одной записи:

```json
{
  "frame_id": 123,
  "timestamp": 12.345,
  "image_path": "frames/frame_000123.jpg",
  "gt": {
    "xyz": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0
  },
  "annotations": {
    "gates": [
      {
        "gate_id": 0,
        "corners_2d": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
      }
    ]
  },
  "online_localizer": {
    "xyz": [0.0, 0.0, 0.0],
    "s": 0.0,
    "progress": 0.0,
    "uncertainty_m": 0.0
  },
  "kf_layer2": {
    "xyz": [0.0, 0.0, 0.0],
    "s": 0.0,
    "progress": 0.0,
    "uncertainty_m": 0.0
  },
  "vision": {
    "xyz": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "sigma_cam": 0.0,
    "confidence": 0.0,
    "reprojection_rmse_px": 0.0,
    "source_mode": "tracking"
  },
  "fusion": {
    "xyz_before_kf": [0.0, 0.0, 0.0],
    "uncertainty_before_kf_m": 0.0,
    "final_xyz_after_kf": [0.0, 0.0, 0.0],
    "final_uncertainty_after_kf_m": 0.0
  }
}
```

Если часть данных еще не считается, поле можно временно ставить в `null`, но
ключ должен оставаться в схеме. Это упростит последующий анализ.

## 5. Обязательные поля

Для минимальной оценки без визуальной fusion нужны:

```text
frame_id
timestamp
image_path
gt.xyz
online_localizer.xyz
online_localizer.s
online_localizer.uncertainty_m
kf_layer2.xyz
kf_layer2.s
kf_layer2.uncertainty_m
```

Для оценки визуального алгоритма дополнительно нужны:

```text
annotations.gates[].gate_id
annotations.gates[].corners_2d
vision.xyz
vision.yaw_deg
vision.sigma_cam
vision.confidence
vision.reprojection_rmse_px
vision.source_mode
```

Для оценки полной fusion-схемы дополнительно нужны:

```text
fusion.xyz_before_kf
fusion.uncertainty_before_kf_m
fusion.final_xyz_after_kf
fusion.final_uncertainty_after_kf_m
```

## 6. Требования к разметке ворот

Для каждого видимого и размеченного гейта нужно указать:

- `gate_id` из карты трассы;
- четыре 2D-угла в пикселях;
- порядок углов `TL -> TR -> BR -> BL`, если он известен;
- если порядок отличается или был восстановлен автоматически, это нужно
  отметить в metadata.

Разметка должна соответствовать тому же набору ворот, который используется в
`config/track.json` или в эквивалентной карте соседнего проекта.

## 7. Формат CVAT-разметки

Разметка кадров выполняется через CVAT по тому же принципу, что и предыдущая
сессия:

```text
calibration/multi_gate_cvat_session_001/
```

Ожидаемый состав CVAT-сессии:

```text
annotations.xml
manifest.csv
gate_mapping.csv
```

`annotations.xml` - экспорт CVAT с объектами `gate` типа `skeleton`. У каждого
скелета должны быть четыре точки:

```text
label="1"
label="2"
label="3"
label="4"
```

Эти точки интерпретируются как углы ворот. При чтении текущими скриптами они
собираются в массив:

```text
[point_1, point_2, point_3, point_4]
```

`manifest.csv` связывает исходный индекс кадра с именем изображения:

```csv
frame_index,image
158,frame_000158.jpg
232,frame_000232.jpg
```

`gate_mapping.csv` связывает CVAT-скелет с реальным `gate_id` из карты трассы:

```csv
frame_index,cvat_gate_index,gate_id
158,1,2
158,6,3
232,11,4
```

Важно: в существующем пайплайне `cvat_gate_index` может быть глобальным индексом
скелета в XML-файле, а не локальным номером ворот внутри одного кадра. Это
совместимо со скриптом `scripts/check_multi_gate_cvat_pnp.py`, который пробует
и глобальный, и локальный 1-based индекс.

Для нового оценочного круга нужно сохранить эту совместимость:

- экспортировать CVAT-разметку в `annotations.xml`;
- сохранить список кадров в `manifest.csv`;
- подготовить `gate_mapping.csv` с колонками `frame_index,cvat_gate_index,gate_id`;
- использовать те же skeleton labels `1..4`;
- не менять смысл `gate_id` относительно карты трассы.

Если соседний проект сможет сразу записывать `gate_id` как attribute объекта в
CVAT, это можно добавить как дублирующую информацию, но `gate_mapping.csv` все
равно нужно оставить для совместимости с текущими скриптами.

Рекомендуемый вариант для новой разметки: добавить `gate_id` как attribute
родительского skeleton-объекта `gate` прямо в CVAT. Логически объект должен
выглядеть так:

```text
skeleton label="gate"
  attribute gate_id = 12
  point label="1"
  point label="2"
  point label="3"
  point label="4"
```

`gate_id` должен быть именно attribute у всего skeleton `gate`, а не отдельной
точкой и не label одной из точек `1..4`. Тип attribute можно сделать `number`
или `text`; если используется `text`, парсер должен приводить значение к `int`.

При таком варианте разметки `gate_mapping.csv` можно формировать автоматически из
CVAT XML, но сам CSV все равно нужно сохранять в артефактах датасета, пока
текущие скрипты ожидают этот файл.

## 8. Требования к синхронизации

Все источники должны быть привязаны к одному времени:

```text
image timestamp
telemetry timestamp
ground truth timestamp
OnlineLocalizer result timestamp
KFLayer2 result timestamp
vision result timestamp
fusion result timestamp
```

Если есть известная задержка изображения, телеметрии или ground truth, ее нужно
явно указать в отдельном metadata-файле или в полях записи.

Рекомендуемые дополнительные поля:

```text
latency.image_ms
latency.telemetry_ms
latency.gt_ms
latency.vision_ms
```

## 9. Метрики качества

Основные ошибки:

```text
error_online_xyz = ||online_xyz - gt_xyz||
error_kf_xyz     = ||kf_xyz - gt_xyz||
error_vision_xyz = ||vision_xyz - gt_xyz||
error_fused_xyz  = ||final_xyz_after_kf - gt_xyz||
```

Отдельно нужно считать ошибку в горизонтальной плоскости:

```text
error_xz = sqrt((x_pred - x_gt)^2 + (z_pred - z_gt)^2)
```

Также желательно считать:

```text
error_y   = abs(y_pred - y_gt)
error_yaw = abs(yaw_pred - gt_yaw)
```

По каждой метрике желательно вывести:

- mean;
- median;
- p90;
- p95;
- max;
- долю кадров с ошибкой меньше заданного порога.

## 10. Raw PnP sanity check и production-путь

При проверке CVAT-разметки важно различать два режима.

### 10.1. Raw PnP sanity check

Если `gate_id` уже известен из CVAT-разметки, можно напрямую вызвать:

```text
PnPSolver.solve([(gate_id, keypoints)])
```

Такой прогон проверяет только базовую техническую корректность:

- `annotations.xml` парсится;
- у каждого skeleton есть точки `1..4`;
- `gate_id` читается из attribute `gate_id` или `id`;
- `PnPSolver` технически может вернуть позу.

Raw PnP не является финальной production-логикой. Для одиночных плоских ворот
он может выбрать неправильную IPPE/зеркальную ветку с очень маленьким
`reprojection_rmse_px`.

Признак такой неоднозначности:

```text
reprojection_rmse_px < 1 px
position_error >> 10 m
```

Это не обязательно ошибка CVAT-разметки. Такой кадр может быть размечен
правильно, но сама single-gate PnP-задача остается неоднозначной.

### 10.2. Production-путь через `gate_localization`

В рабочей системе визуальное наблюдение должно проходить через
`gate_localization`, а не через слепой raw PnP по минимальному RMSE:

```text
2D gate detections
  -> gate_localization(coarse_pose from OnlineLocalizer/KFLayer2)
  -> candidate gate IDs + candidate PnP poses
  -> choose hypothesis consistent with coarse_pose
  -> CameraObservation(xyz, sigma_cam)
```

`gate_localization` нужен не только для выбора `gate_id`, но и для выбора
правильной стороны ворот / правильной PnP-ветки.

Для одиночного gate:

```text
single planar gate + weak prior = ambiguous
single planar gate + good DCT prior = usable observation
```

Если PnP-поза далеко от prior-а `OnlineLocalizer/KFLayer2`, наблюдение нужно
отклонить или вернуть с большим `sigma_cam`, а не инжектить как точное камерное
измерение.

Текущее базовое правило reject/gating:

```text
dist_to_prior_xz = ||vision_xyz_xz - coarse_xyz_xz||

multi-gate:
    reject if dist_to_prior_xz > max(3 * q_m, 5 м)

single-gate:
    reject if dist_to_prior_xz > max(2 * q_m, 3 м)
```

Если наблюдение отклонено, оно не должно попадать в
`OnlineLocalizer.inject_position_observation(...)`. Если результат все же нужно
передать дальше для анализа, его `sigma_cam/q_out_m` должен быть выставлен в
плохое значение, например `20 м`.

Дополнительно визуальное наблюдение может быть отклонено как бесполезное для
fusion, даже если оно не является грубо ошибочным:

```text
reject if q_out_m > 3 м
reject if dist_to_prior_xz < max(q_m, 1 м)
```

Смысл: если сама visual-оценка имеет слишком большую неопределенность
`q_out_m`, ее не нужно инжектить. Если visual-поза почти не отличается от
prior-а относительно текущей неопределенности `q_m`, такое наблюдение тоже не
уточняет состояние заметно и может быть пропущено. Если visual-поза заметно
отличается, она проходит только при условии, что не нарушает верхний
distance-gating.

Текущий результат проверки на
`reference_lap_dataset_5fps_frames0_862`:

```text
annotated frames with gates: 56
accepted for KFLayer2 refinement: 13
rejected: 43

rejected reasons:
    too far from coarse prior: 24
    not useful versus coarse prior: 13
    q_out too high for injection: 6

accepted ID status:
    OK: 5
    same gate set, different order: 5
    BAD: 3

accepted impact against KFLayer2, XZ error:
    improved: 9 frames
    worsened: 4 frames
    mean gain over all accepted: +0.52 м
    median gain over all accepted: +0.96 м
```

Лучшие улучшения среди accepted:

```text
frame 138: 4.70 м -> 0.18 м, gain +4.52 м
frame 276: 2.17 м -> 0.07 м, gain +2.10 м
frame 144: 2.67 м -> 0.67 м, gain +2.00 м
frame 270: 2.07 м -> 0.27 м, gain +1.80 м
frame 192: 1.94 м -> 0.31 м, gain +1.63 м
```

Основные ухудшения среди accepted:

```text
frame 180: 1.25 м -> 4.73 м, loss -3.48 м
frame 54: 0.00 м -> 2.16 м, loss -2.16 м
frame 360: 0.51 м -> 2.42 м, loss -1.91 м
frame 150: 1.56 м -> 2.79 м, loss -1.22 м
```

Вывод по текущему состоянию: фильтр уже сильно уменьшает число опасных
визуальных наблюдений, но еще не является финальным. Для production-fusion
нужно дополнительно разбирать false-positive accepted кадры, особенно
`180`, `54`, `360`, `150`, и добавлять ID-consistency/ambiguity фильтр.

Кадры, где raw PnP дает маленький RMSE, но большую ошибку позиции, нужно
использовать как тест-кейсы для `gate_localization`. Их не следует автоматически
считать ошибками CVAT-разметки.

## 11. Что нужно сравнить

После подготовки круга нужно сравнить:

1. `OnlineLocalizer` против `GT`.
2. `KFLayer2` против `GT`.
3. Визуальный `PnP` против `GT`.
4. Fusion после камеры против `GT`.
5. Fusion + `KFLayer2` против `GT`.

Камера считается полезной, если после `inject_position_observation()` ошибка
и/или неопределенность уменьшается без сильных скачков позиции.

`KFLayer2` считается полезным, если он уменьшает шум и выбросы относительно
`OnlineLocalizer`, не создавая неприемлемую задержку.

## 12. Выходные артефакты

Нужны:

```text
reference_lap_eval.jsonl
frames/
metadata.json
cvat/
```

Где:

- `reference_lap_eval.jsonl` - индекс кадров и все численные данные;
- `frames/` - изображения, на которые ссылается `image_path`;
- `metadata.json` - описание трассы, координатной системы, задержек, источников
  данных и версии алгоритмов.
- `cvat/` - CVAT-экспорт разметки: `annotations.xml`, `manifest.csv`,
  `gate_mapping.csv`.

Минимальное содержимое `metadata.json`:

```json
{
  "coordinate_system": {
    "units": "meters",
    "axes": "X right, Y up, Z forward",
    "origin": "track/map origin description"
  },
  "track": {
    "name": "track-name",
    "map_source": "config/track.json or DCT reference file"
  },
  "reference": {
    "is_same_lap_as_eval": false,
    "reference_file": "reference.npz"
  },
  "sync": {
    "image_latency_ms": 0.0,
    "telemetry_latency_ms": 0.0,
    "gt_latency_ms": 0.0
  },
  "versions": {
    "online_localizer": "unknown",
    "kf_layer2": "unknown",
    "fpvcamdetect": "unknown"
  }
}
```

## 13. Критерии приемки датасета

Датасет считается пригодным для первичной оценки, если:

- есть хотя бы один полный круг;
- все кадры имеют timestamp;
- для каждого кадра есть `gt.xyz`;
- для каждого кадра есть результат `OnlineLocalizer` и `KFLayer2`;
- координатная система явно описана;
- видимые ворота размечены хотя бы на выбранном подмножестве кадров;
- CVAT-разметка экспортирована в `annotations.xml`;
- есть `manifest.csv` и `gate_mapping.csv` в формате предыдущих CVAT-сессий;
- пути `image_path` указывают на существующие изображения;
- явно указано, является ли круг отдельным eval lap или тем же кругом, что и
  reference lap.
