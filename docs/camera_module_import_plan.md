# План импорта экспериментального модуля камеры

Документ фиксирует безопасный план переноса FPV camera localization в DCT.
Цель первого этапа - не заменить основной локализатор, а добавить
экспериментальный источник абсолютных XYZ-наблюдений, который можно включать
offline и строго контролировать перед инжектом в Particle Filter.

## 1. Источники

### FPVCamDetectV2

Основной источник логики. Переносить по слоям:

- `src/yolo_gate_adapter/` - парсинг YOLO pose labels, фильтрация keypoint confidence, dedup, top-N.
- `src/gate_model/` - геометрия ворот и преобразование `track.json` в world corners.
- `src/pnp_solver/` или `src/pnp_solver_2/` - PnP по 2D keypoints и 3D углам ворот.
- `src/gate_localization/` - association `detection -> gate_id`, top-K гипотезы, `CoarseRefineLocalizer`.
- `config/camera_calibration.json` - стартовая калибровка камеры.
- `config/track.json` - стартовая карта ворот для проверки импорта.
- `requirements.txt` - минимальные зависимости (`numpy`, `opencv-python`).

### FPVCamDetect

Использовать только как источник обученных весов YOLO и метаданных модели.
Остальную старую экспериментальную логику на первом этапе не переносить.

Нужно отдельно зафиксировать:

- путь к файлу весов;
- формат весов (`.pt`, `.onnx`, `.hef` или другой);
- input size модели;
- порядок keypoints;
- confidence format выходов модели.

## 2. Целевая структура в DCT

```text
dct/
  camera_localization/
    __init__.py
    observation.py
    yolo_gate_adapter/
    gate_model/
    pnp_solver/
    gate_localization/
    runtime/
      offline_pipeline.py
      observation_writer.py
      fusion_policy.py

tools/
  exp11_camera_module_import/
    README.md
    run_yolo_labels_to_observations.py
    run_camera_fusion_offline.py
    report.md

models/
  yolo_gate_pose/
    README.md
    weights.*          # может не храниться в git
    model_meta.json
```

`dct/localization/online_localizer.py` не должен зависеть от YOLO/PnP.
Связь с основным локализатором выполняется только через готовое наблюдение:

```python
OnlineLocalizer.inject_position_observation(xyz_obs, sigma_cam)
```

## 3. Контракт наблюдения

Промежуточный формат для камеры:

```python
CameraObservation(
    timestamp: float,
    xyz_obs: tuple[float, float, float],
    sigma_cam: float,
    confidence: float,
    gate_id: int | str | None,
    reprojection_error_px: float | None,
    status: str,
)
```

`status="ok"` означает, что наблюдение прошло проверки камерного пайплайна.
Это еще не означает, что его можно inject'ить в DCT: финальное решение принимает
`fusion_policy`.

## 4. Этапы импорта

### Этап 0. Подготовка структуры

Создать папки:

- `dct/camera_localization/`
- `dct/camera_localization/runtime/`
- `tools/exp11_camera_module_import/`
- `models/yolo_gate_pose/`

Результат: в репозитории есть место для импортируемого кода, весов и
экспериментальных скриптов.

### Этап 1. Инвентаризация внешних файлов

Составить список переносимых файлов из `FPVCamDetectV2` и путь к весам из
`FPVCamDetect`.

Критерии готовности:

- понятен источник каждого переносимого файла;
- известно, какие файлы являются runtime-кодом, а какие только debug/validation;
- зафиксирован путь к YOLO weights;
- подтвержден порядок keypoints: `TL -> TR -> BR -> BL`.

### Этап 2. Перенос геометрии и PnP

Перенести:

- `gate_model`;
- `pnp_solver`;
- стартовые camera/track configs.

Критерии готовности:

- PnP запускается внутри DCT на тестовом наборе keypoints;
- возвращает `position_world`;
- зависимости от старой структуры проекта удалены;
- импорты работают из namespace `dct.camera_localization`.

### Этап 3. Перенос `yolo_gate_adapter`

Перенести adapter, который превращает YOLO labels в `GateDetection[]`.

Стартовые параметры:

```text
min_keypoint_confidence = 0.7
max_detections = 6
deduplicate_iou_threshold = 0.75
deduplicate_center_distance_px = 12
```

Критерии готовности:

- label-файл YOLO pose читается без внешних зависимостей от FPVCamDetectV2;
- слабые keypoints отбрасываются;
- дубликаты удаляются;
- результат совместим с `CoarseRefineLocalizer`.

### Этап 4. Observation writer

Собрать offline pipeline:

```text
YOLO labels
  -> yolo_gate_adapter
  -> CoarseRefineLocalizer
  -> CameraObservation[]
  -> observations.csv/json/parquet
```

Критерии готовности:

- логируются accepted/rejected наблюдения;
- есть причины reject;
- сохраняются `xyz_obs`, `sigma_cam`, `confidence`, `gate_id`, `status`;
- можно повторить прогон на одном наборе кадров.

### Этап 5. Fusion policy

Перед вызовом `inject_position_observation` добавить защитный слой.

Минимальные правила:

- reject `sigma_cam > 10 м`;
- rate limit камеры: начать с `1-2 Гц`, затем тестировать до `5 Гц`;
- reject резких скачков относительно текущего PF/KF;
- осторожный режим для single-gate PnP;
- не inject'ить ambiguous `gate_id`;
- логировать причину каждого skip/reject.

Критерии готовности:

- плохие наблюдения не попадают напрямую в PF;
- можно отключить камеру без изменения основного локализатора;
- все решения policy воспроизводимо логируются.

### Этап 6. Offline fusion experiment

Подать `CameraObservation` в текущий DCT pipeline:

```text
sticks update
  -> optional camera inject
  -> KFLayer2 / metrics
```

Сравнить:

- baseline без камеры;
- camera accepted only;
- разные `sigma_cam`;
- разные `T_update`;
- p50/p90/max error;
- число улучшений и ухудшений.

Критерий успеха: камера стабильно улучшает p90 на выбранном сценарии и не
создает тяжелые выбросы.

### Этап 7. Подключение реальных YOLO weights

После offline labels pipeline подключить inference:

```text
frame
  -> YOLO pose model
  -> detections/keypoints
  -> yolo_gate_adapter
  -> localization
```

Критерии готовности:

- формат выхода модели совпадает с adapter;
- измерен FPS;
- проверена перестановка keypoints;
- распределение confidence похоже на ожидаемое;
- качество не хуже offline labels path.

## 5. Что не переносить на первом этапе

- старые SLAM-модули из `FPVCamDetect`;
- старый `ekf_layer2`, потому что в DCT уже есть `KFLayer2`;
- исторические `outputs/`;
- все debug-скрипты без прямой пользы для импорта;
- большие датасеты и видео;
- веса модели в git, если файл большой или меняется часто.

## 6. Риски

### Нечестный `sigma_cam`

Если `sigma_cam` занижен, PF начинает доверять плохому observation и может
ухудшиться. На старте лучше завышать `sigma_cam` и отбрасывать сомнительные
кадры.

### Single-gate planar PnP

Один плоский gate может давать зеркальные или близкие по reprojection error
позы. Для таких кадров нужны temporal consistency, jump rejection и отдельная
политика доверия.

### Gate association

Неверный `gate_id` создает правдоподобное, но неправильное `xyz_obs`. При
неоднозначности лучше не inject'ить наблюдение.

### Latency

`CoarseRefineLocalizer` может быть дорогим для 2-3 ворот. Для real-time режима
нужно ограничивать число detections, top-K и частоту обработки.

## 7. Definition of Done для экспериментального включения

Модуль готов к экспериментальному включению, когда:

1. DCT может прочитать `observations.*`, созданные камерным pipeline.
2. Каждое observation содержит `xyz_obs`, `sigma_cam`, `status` и диагностические поля.
3. `fusion_policy` явно принимает или отклоняет observation.
4. Камера включается отдельным флагом и по умолчанию выключена.
5. Offline-эксперимент показывает p90 не хуже baseline и желательно стабильное улучшение.
6. Все accepted/rejected camera updates сохраняются в лог для анализа.
