# ТЗ: FPV Camera Gate Detection and Localization

Документ описывает отдельный Python-проект для визуальной локализации дрона по
FPV-камере. Проект разрабатывается независимо от DCT, а затем интегрируется с
основным локализатором через простой контракт: `timestamp + xyz_obs +
sigma_cam + confidence + gate_id`.

Целевая версия первого этапа работает на данных Liftoff. Реальное железо
целевого исполнения: Hailo-8L. Разработка, обучение и первичная валидация:
Windows CPU/GPU.

---

## 1. Цель проекта

Система должна по кадру FPV-видео:

1. Найти видимые ворота трассы.
2. Определить 4 ключевые точки внутреннего проёма ворот.
3. Идентифицировать, какие это ворота из `track.json`.
4. По 2D keypoints + 3D геометрии ворот + калибровке камеры решить PnP.
5. Получить позицию камеры/дрона в координатах трассы.
6. Выдать наблюдение для внешнего фильтра:

```python
CameraObservation(
    timestamp: float,
    xyz_obs: tuple[float, float, float],
    sigma_cam: float,
    confidence: float,
    gate_id: str | int | None,
    source: str = "fpv_gate_pnp",
)
```

Целевые характеристики по результатам Exp 10:

- минимум: `sigma_cam <= 5 м`, частота `>= 1-2 Гц`;
- хороший уровень: `sigma_cam <= 1 м`, частота `>= 5 Гц`;
- идеальный уровень для максимальной точности: `sigma_cam ~= 0.1-0.5 м`,
  частота `5-10 Гц`;
- не стремиться к частоте выше `10 Гц` без отдельной настройки PF: при слишком
  частых и слишком точных наблюдениях возможна бимодальность фильтра.

---

## 2. Исходные допущения

### 2.1 Данные и среда

- Первый датасет: видео из Liftoff.
- На первом этапе ground truth позиции дрона отсутствует: есть только видео.
- В дальнейшем желательно добавить экспорт телеметрии Liftoff для количественной
  проверки `xyz_obs`.
- Проект пишется на Python.
- Тесты запускаются на Windows CPU/GPU.
- Целевой runtime для модели: Hailo-8L.

### 2.2 Ворота

- Все ворота одинаковые.
- Внутренний проём: `1.56 м x 1.56 м`.
- Размечается именно внутренний проём, не внешняя рамка.
- В `track.json` уже есть позиция и ориентация ворот.

### 2.3 Камера

- Камера жёстко закреплена на дроне.
- Transform `camera -> drone_body` пока неизвестен.
- На первом этапе PnP даёт позицию камеры. Для позиции центра дрона нужен
  отдельный этап калибровки внешнего смещения камеры.

### 2.4 Сцены

- В кадре часто будет несколько ворот.
- Ворота могут быть частично перекрыты, смазаны motion blur или находиться на
  краю кадра.
- В Liftoff возможны визуальные условия, отличающиеся от реального FPV: это
  допустимо для первого прототипа, но позже нужен доменный перенос.

---

## 3. Рекомендуемая структура отдельного проекта

```text
dct-camera-localizer/
  README.md
  pyproject.toml
  configs/
    dataset.yaml
    train_yolo_pose.yaml
    camera_calibration.yaml
    runtime.yaml
  data/
    raw_videos/
    frames/
    annotations/
    yolo/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
      data.yaml
  tracks/
    track-002/
      track.json
      gates_world.json
  models/
    yolo/
    onnx/
    hailo/
  src/
    camera_localizer/
      dataset/
        extract_frames.py
        split_dataset.py
        validate_annotations.py
      calibration/
        calibrate_intrinsics.py
        estimate_camera_to_body.py
      detection/
        train_yolo_pose.py
        export_onnx.py
        infer_yolo.py
      geometry/
        gates.py
        pnp.py
        projection.py
        association.py
      runtime/
        pipeline.py
        api.py
      evaluation/
        eval_detector.py
        eval_pnp.py
        eval_runtime.py
  notebooks/
    dataset_inspection.ipynb
    pnp_debug.ipynb
  tests/
```

---

## 4. Что нужно собрать

### 4.1 Минимальный датасет для первого прототипа

Минимум:

- `500-1000` размеченных кадров;
- 1-2 трассы;
- разные дистанции до ворот;
- разные углы захода;
- несколько ворот в кадре;
- кадры с motion blur;
- кадры, где ворота близко к краю изображения;
- кадры без ворот для проверки false positives.

Рекомендуемый объём:

- `3000-5000` размеченных кадров;
- 3+ трассы или вариации трасс;
- отдельные видео для train/val/test, не случайная нарезка одного видео.

Важно: train/val/test нужно делить **по видео-сессиям**, а не случайно по
кадрам. Иначе соседние кадры попадут и в train, и в val, метрики будут
завышены.

### 4.2 Какие видео записывать в Liftoff

Для каждой трассы записать несколько типов пролётов:

1. Нормальные круги:
   - стабильный полёт;
   - типичная гоночная скорость;
   - 5-10 кругов.

2. Медленные пролёты для качественной разметки:
   - ворота крупнее в кадре;
   - меньше motion blur;
   - полезно для первых 500-1000 кадров.

3. Агрессивные пролёты:
   - быстрые повороты;
   - крены;
   - вход в ворота под углом;
   - сильный blur.

4. Негативные сцены:
   - кадры между воротами;
   - земля/небо/декорации без ворот;
   - элементы, похожие на ворота.

5. Многоворотные сцены:
   - 2+ ворот в кадре;
   - дальние ворота за ближними;
   - ворота частично перекрывают друг друга.

### 4.3 Как извлекать кадры

Не нужно размечать каждый кадр подряд. Лучше извлекать разнообразные кадры:

```text
video_001.mp4
  -> 1 кадр каждые 0.2-0.5 секунды для обычных участков
  -> чаще в местах, где ворота близко/видны под углом
  -> вручную добавить сложные кадры
```

Практическое правило:

- первый датасет: 1000 кадров из 10-20 минут видео;
- финальный датасет: 3000-5000 кадров из 30-60 минут видео.

### 4.4 Что обязательно хранить рядом с видео

Для каждой записи:

```yaml
video_id: liftoff_track002_session001
track_id: track-002
resolution: [width, height]
fps: 60
camera_fov_deg: ...
liftoff_graphics_settings: ...
drone_model: ...
notes: "normal laps / slow laps / aggressive laps"
```

Если позже получится экспортировать телеметрию Liftoff, добавить:

```yaml
telemetry_file: telemetry.parquet
time_sync: video_timestamp_to_telemetry_timestamp
```

---

## 5. Калибровка камеры

### 5.1 Intrinsics

Для PnP нужны:

- `fx`, `fy`;
- `cx`, `cy`;
- коэффициенты дисторсии;
- размер изображения.

Для Liftoff возможны два режима:

1. Если известен FOV камеры:
   - оценить intrinsics аналитически:

```python
fx = width / (2 * tan(horizontal_fov / 2))
fy = height / (2 * tan(vertical_fov / 2))
cx = width / 2
cy = height / 2
distortion = zeros
```

2. Если FOV неизвестен:
   - подобрать FOV по синтетической сцене с известными воротами;
   - минимизировать reprojection error между известной 3D-геометрией ворот и
     размеченными keypoints.

Для реальной камеры позже:

- использовать ChArUco / checkerboard;
- собрать 30-50 изображений калибровочной доски;
- сохранить `camera_matrix` и `dist_coeffs`.

### 5.2 Extrinsics: camera -> drone_body

Transform пока неизвестен. На первом этапе допускается:

```text
camera position ~= drone center
camera orientation ~= drone forward axis
```

Это достаточно для прототипа, но для точности < 1 м нужно откалибровать:

- смещение камеры относительно центра дрона;
- угол наклона камеры;
- roll/pitch/yaw mounting offset.

Варианты калибровки:

1. Ручная:
   - измерить положение камеры на корпусе;
   - задать `camera_to_body` вручную.

2. Оптимизационная:
   - записать видео с телеметрией;
   - для кадров с уверенным PnP получить `camera_world`;
   - подобрать `T_body_camera`, минимизируя ошибку между телеметрическим
     `body_world` и PnP-derived `body_world`.

3. Для Liftoff:
   - если камера виртуальная и находится в центре дрона, принять transform
     identity;
   - если Liftoff задаёт FOV/tilt камеры, использовать эти параметры.

---

## 6. Разметка кадров

### 6.1 Формат задачи

Модель:

- YOLOv8n-pose;
- 1 класс: `gate`;
- 4 keypoints: углы внутреннего проёма.

Класс:

```text
0: gate
```

Keypoints в строгом порядке:

```text
0: inner_top_left
1: inner_top_right
2: inner_bottom_right
3: inner_bottom_left
```

Порядок задаётся **с точки зрения изображения**, а не с точки зрения ворот.
То есть `inner_top_left` — верхний левый угол проёма на кадре.

### 6.2 Правила разметки

1. Размечать внутренний проём `1.56 x 1.56 м`.
2. Не размечать внешнюю рамку.
3. Если в кадре несколько ворот, размечать все видимые ворота.
4. Если видны 4 внутренних угла — размечать.
5. Если один угол слегка перекрыт, но человек уверенно восстанавливает его
   положение — размечать как visible/occluded в зависимости от инструмента.
6. Если видно меньше 3 углов — лучше отправить объект в ignore или не размечать.
7. Если ворота сильно смазаны и человек не уверен в углах — ignore.
8. Если ворота очень далеко и занимают менее 10-15 px по ширине — ignore для
   keypoint-обучения, но можно оставить как negative/context.
9. Если ворота частично вне кадра, но 4 угла видны — размечать.
10. Если 4 угла не видны из-за выхода за границу кадра — не использовать для
    PnP-разметки первого этапа.

### 6.3 Инструмент разметки

Подходящие варианты:

- CVAT с keypoints/skeleton;
- Roboflow keypoint annotation;
- Label Studio keypoint template;
- любой инструмент, который экспортирует YOLO pose.

Рекомендуемый процесс:

1. Импортировать кадры в CVAT.
2. Создать label `gate`.
3. Создать skeleton/keypoints из 4 точек.
4. Размечать по правилам выше.
5. Экспортировать в YOLO pose.
6. Запустить локальный скрипт `validate_annotations.py`.

### 6.4 Проверка качества разметки

Скрипт проверки должен:

- проверить, что у каждого `gate` ровно 4 keypoints;
- проверить порядок точек;
- проверить, что bbox покрывает keypoints;
- проверить, что площадь проёма не отрицательная;
- найти слишком маленькие ворота;
- найти дубликаты;
- визуализировать случайные 100 кадров с keypoints.

Особенно важно проверять порядок точек. Ошибка порядка ломает PnP сильнее,
чем небольшая пиксельная ошибка.

---

## 7. Обучение YOLO pose

### 7.1 Базовая модель

Стартовая модель:

```text
YOLOv8n-pose
input: 320x320
classes: 1
keypoints: 4
```

Почему `YOLOv8n`:

- маленькая модель;
- подходит для Hailo-8L;
- проще достичь `>= 50 FPS`;
- достаточно для одного класса.

Позже проверить:

- `320x320`: быстрая, но хуже дальние ворота;
- `416x416`: компромисс;
- `640x640`: лучше keypoints, но может не пройти целевой FPS.

### 7.2 Аугментации

Обязательные:

- brightness/contrast;
- motion blur;
- Gaussian blur;
- compression artifacts;
- perspective transform;
- random crop;
- scale;
- noise;
- partial occlusion.

Осторожно:

- сильный rotation может ломать семантику `top_left/top_right`, если разметка
  задаётся в координатах изображения;
- horizontal flip допустим только если keypoint-порядок корректно переставляется.

### 7.3 Метрики обучения

Для детектора:

- bbox mAP;
- recall ворот;
- false positives на кадрах без ворот.

Для keypoints:

- keypoint mAP;
- mean pixel error;
- P90 pixel error;
- процент объектов, где все 4 keypoints найдены.

Для downstream PnP:

- reprojection error, px;
- доля кадров, где PnP успешно решён;
- стабильность distance/bearing/elevation на последовательных кадрах.

Цели первого прототипа:

- gate recall > 90%;
- 4-keypoint success rate > 80%;
- P90 keypoint error < 8 px на `320x320`.

Цели рабочей версии:

- gate recall > 95%;
- 4-keypoint success rate > 90%;
- P90 keypoint error < 3-5 px на `320x320` или `416x416`;
- inference на Hailo-8L >= 50 FPS.

---

## 8. Экспорт и runtime

Целевой pipeline:

```text
YOLOv8n-pose
  -> ONNX
  -> Hailo compilation
  -> HEF
  -> Hailo-8L runtime
```

На Windows:

```text
YOLO PyTorch / ONNX Runtime / OpenCV DNN
```

Нужно заранее проверить поддержку YOLO pose/keypoints в Hailo toolchain:

1. Поддерживается ли выбранная архитектура YOLOv8n-pose напрямую.
2. Где выполняется postprocess:
   - на Hailo;
   - на CPU после raw output.
3. Как получить keypoints из выходного тензора.
4. Есть ли quantization loss по keypoints после HEF.

Runtime-цели:

- detector FPS на Hailo-8L: `>= 50 FPS` на `320x320`;
- end-to-end observation FPS после PnP/association: `5-10 Гц`;
- latency observation: желательно `<= 200 мс`, максимум `<= 500 мс`.

Важно: detector может работать 50 FPS, но в DCT не нужно inject'ить все 50
наблюдений. Для PF целевой update rate камеры: `5-10 Гц`.

---

## 9. Геометрия ворот и track.json

### 9.1 Локальная система ворот

Внутренний проём квадратный `1.56 x 1.56 м`.

Задаём 3D-точки углов в локальной системе ворот:

```python
W = 1.56
H = 1.56

gate_corners_local = [
    [-W/2, +H/2, 0.0],  # inner_top_left
    [+W/2, +H/2, 0.0],  # inner_top_right
    [+W/2, -H/2, 0.0],  # inner_bottom_right
    [-W/2, -H/2, 0.0],  # inner_bottom_left
]
```

Ось `Z=0` — плоскость ворот. Направление нормали должно соответствовать
ориентации ворот в `track.json`.

### 9.2 World corners

Для каждого gate из `track.json`:

```python
gate_corners_world = T_world_gate @ gate_corners_local
```

Нужно создать промежуточный файл:

```text
tracks/track-002/gates_world.json
```

Пример структуры:

```json
{
  "track_id": "track-002",
  "gate_inner_size_m": [1.56, 1.56],
  "gates": [
    {
      "gate_id": 0,
      "center_world": [x, y, z],
      "rotation_world_gate": [[...], [...], [...]],
      "corners_world": {
        "inner_top_left": [x, y, z],
        "inner_top_right": [x, y, z],
        "inner_bottom_right": [x, y, z],
        "inner_bottom_left": [x, y, z]
      }
    }
  ]
}
```

---

## 10. PnP: как из кадра получить позицию

### 10.1 Один detection + известный gate_id

Вход:

- `image_points`: 4 keypoints в пикселях;
- `object_points`: 4 угла этих ворот в world frame;
- `camera_matrix`;
- `dist_coeffs`.

Алгоритм:

```python
success, rvec, tvec = cv2.solvePnP(
    objectPoints=gate_corners_world,
    imagePoints=keypoints_2d,
    cameraMatrix=K,
    distCoeffs=dist,
    flags=cv2.SOLVEPNP_IPPE_SQUARE,  # для квадратной плоской цели
)
```

Для плоского квадрата лучше тестировать:

- `SOLVEPNP_IPPE_SQUARE`;
- `SOLVEPNP_ITERATIVE`;
- `solvePnPGeneric`, потому что для плоской цели возможны две позы.

После PnP:

```text
R_world_camera, t_world_camera
  -> camera_center_world
  -> если известен T_body_camera:
       drone_center_world
     иначе:
       xyz_obs = camera_center_world
```

### 10.2 Проверки валидности PnP

Отбрасывать observation, если:

- `success == False`;
- reprojection error > threshold;
- дистанция до ворот вне физически разумного диапазона;
- ворота получаются за камерой;
- нормаль ворот смотрит в невозможную сторону;
- оценка прыгает слишком сильно относительно предыдущей;
- `sigma_cam > 10 м`.

Базовые пороги для старта:

```yaml
max_reprojection_error_px: 5-10
min_gate_pixel_size: 20
max_distance_m: 80
max_sigma_cam_m: 10
```

---

## 11. Идентификация ворот при нескольких воротах в кадре

Так как визуальных маркеров нет, идентификация должна использовать контекст.

### 11.1 Входы association-модуля

```python
associate_gates(
    detections,          # YOLO detections with 4 keypoints
    track_gates,         # gates from track.json
    camera_calibration,
    belief_state=None,   # optional: s_est, sigma_s from DCT
    previous_state=None, # optional: previous camera pose
)
```

### 11.2 Кандидаты ворот

Если есть `belief_state` из DCT:

```text
candidate_gates = gates_near_s(s_est, window=30-80 м)
```

Если `belief_state` нет:

- перебрать все ворота;
- оставить решения PnP с хорошим reprojection error;
- выбрать уникальное согласованное решение;
- если неоднозначность высокая — не выдавать observation.

### 11.3 Scoring detection-gate пары

Для каждой пары `(detection, candidate_gate)`:

1. Решить PnP.
2. Посчитать reprojection error.
3. Посчитать ожидаемый bearing/elevation/distance из текущего belief.
4. Проверить физическую валидность.
5. Сформировать score:

```python
score = (
    w_reproj * reprojection_error_px
    + w_bearing * bearing_error_rad
    + w_distance * distance_prior_error_m
    + w_conf * (1 - detection_confidence)
    + w_size * gate_size_penalty
)
```

### 11.4 Hungarian algorithm

Если в кадре несколько detections и несколько candidate gates:

```text
cost_matrix[detection_i, gate_j] = score(i, j)
assignment = Hungarian(cost_matrix)
```

После assignment:

- принять только пары со score < threshold;
- если несколько решений близки по score, пометить как ambiguous;
- если ambiguous, observation не inject'ить.

### 11.5 Выбор observation из нескольких ворот

Если несколько ворот дали валидную позу:

Вариант 1, простой:

- выбрать observation с минимальным `sigma_cam`.

Вариант 2, лучше:

- объединить несколько pose estimates weighted average по `1/sigma_cam^2`;
- sigma уменьшить пропорционально числу независимых ворот.

Для первого прототипа выбрать вариант 1.

---

## 12. Оценка sigma_cam

`sigma_cam` — обязательная часть API. Exp 10 показал: плохие observations
лучше отбросить, чем подмешивать.

### 12.1 Факторы sigma_cam

`sigma_cam` должен расти, если:

- большой reprojection error;
- ворота маленькие в кадре;
- низкий confidence детектора;
- keypoints имеют низкий confidence;
- ворота сильно под углом;
- большая дистанция;
- сильный blur;
- несколько gate_id дают похожий score;
- PnP имеет две похожие плоские гипотезы.

### 12.2 Первая эвристическая модель

```python
sigma_cam = base_sigma
sigma_cam += k_reproj * reprojection_error_px
sigma_cam += k_dist * distance_m
sigma_cam += k_conf * (1 - detection_confidence)
sigma_cam += k_size * (1 / gate_pixel_size)
sigma_cam += k_amb * ambiguity_penalty
```

Стартовые значения подбирать на валидации:

```yaml
base_sigma: 0.5
k_reproj: 0.2
k_dist: 0.03
k_conf: 2.0
k_size: tuned
k_amb: 3.0
```

### 12.3 Правила использования

```text
sigma_cam <= 1 м   -> отличное наблюдение
1-3 м              -> хорошее
3-5 м              -> полезное
5-10 м             -> осторожно, возможно downweight
>10 м              -> reject, не inject'ить
```

Для первого прототипа можно начать с дискретных уровней:

```python
if reproj < 2 px and gate_size > 80 px:
    sigma_cam = 1.0
elif reproj < 5 px and gate_size > 40 px:
    sigma_cam = 3.0
elif reproj < 10 px:
    sigma_cam = 5.0
else:
    reject
```

---

## 13. API независимого проекта

### 13.1 Python API

```python
from camera_localizer import CameraLocalizer

localizer = CameraLocalizer(
    model_path="models/yolo/gate_pose.onnx",
    camera_calibration="configs/camera_calibration.yaml",
    track_gates="tracks/track-002/gates_world.json",
)

obs = localizer.process_frame(
    frame=frame_bgr,
    timestamp=ts,
    belief_state={
        "s": 123.4,
        "sigma_s": 5.0,
    },
)

if obs is not None and obs.sigma_cam <= 10:
    # later in DCT:
    # dct_localizer.inject_position_observation(obs.xyz_obs, obs.sigma_cam)
    pass
```

### 13.2 Observation schema

```python
@dataclass
class CameraObservation:
    timestamp: float
    xyz_obs: np.ndarray              # shape (3,), world coordinates
    sigma_cam: float                 # meters
    confidence: float                # 0..1
    gate_id: int | str | None
    reprojection_error_px: float
    distance_to_gate_m: float
    bearing_rad: float
    elevation_rad: float
    detection_bbox: tuple[float, float, float, float]
    keypoints_2d: np.ndarray         # shape (4, 2)
    status: str                      # ok / rejected / ambiguous / no_gate
```

### 13.3 CLI API

```bash
python -m camera_localizer.infer_video \
  --video data/raw_videos/track002_session001.mp4 \
  --track tracks/track-002/gates_world.json \
  --calib configs/camera_calibration.yaml \
  --model models/yolo/gate_pose.onnx \
  --out outputs/track002_session001_observations.parquet \
  --debug-video outputs/debug_overlay.mp4
```

Выход:

- `observations.parquet`;
- `debug_overlay.mp4`;
- `metrics.json`;
- `rejected_frames.csv`.

---

## 14. Валидация

### 14.1 Этап A: detector/keypoints без ground truth позиции

Можно сделать уже на первом датасете.

Метрики:

- detection recall;
- false positive rate;
- keypoint pixel error на val/test;
- 4-keypoint success rate;
- FPS на CPU/GPU;
- визуальный debug overlay.

Критерий готовности:

- модель уверенно находит ворота;
- keypoints визуально стоят на внутренних углах;
- PnP не разваливается на типичных кадрах.

### 14.2 Этап B: PnP consistency без ground truth позиции

Даже без позиции дрона можно проверять:

- reprojection error;
- плавность estimated camera trajectory;
- отсутствие резких скачков gate_id;
- согласованность нескольких ворот в одном кадре;
- физически разумную дистанцию до ворот.

Критерий готовности:

- trajectory визуально гладкая;
- gate association не прыгает;
- плохие кадры уходят в reject.

### 14.3 Этап C: количественная проверка с Liftoff telemetry

Желательно добавить на втором этапе.

Нужно получить:

- видео;
- timestamp кадров;
- позицию/ориентацию дрона из Liftoff;
- синхронизацию видео и telemetry.

Метрики:

- `median_xyz_error_m`;
- `p90_xyz_error_m`;
- `sigma calibration`: соответствует ли заявленный `sigma_cam` фактической
  ошибке;
- ошибка distance/bearing/elevation;
- recall валидных observations.

Критерии:

- хороший уровень: p90 <= 2 м;
- целевой уровень: p90 <= 1 м;
- идеальный уровень: p90 <= 0.5 м на простых сценах.

### 14.4 Этап D: интеграция с DCT

После появления `observations.parquet`:

1. Подать observations в DCT PF.
2. Ограничить inject rate до `5-10 Гц`.
3. Reject observations с `sigma_cam > 10 м`.
4. Проверить p90 против baseline.
5. Проверить, что не появляется бимодальность при слишком частых updates.

---

## 15. Этапный план работ

### Milestone 0: подготовка проекта

Результат:

- создан Python-проект;
- настроены зависимости;
- добавлена структура папок;
- добавлен parser `track.json -> gates_world.json`;
- добавлен stub `CameraLocalizer`.

### Milestone 1: первый датасет

Результат:

- записаны Liftoff-видео;
- извлечены 1000 кадров;
- размечены ворота с 4 keypoints;
- train/val/test split по видео;
- написан annotation validator.

### Milestone 2: первая YOLO pose модель

Результат:

- обучена YOLOv8n-pose;
- получены detector/keypoint metrics;
- есть debug overlay;
- выбран input size: 320 или 416.

### Milestone 3: PnP prototype

Результат:

- реализован `solvePnP` по одному gate;
- реализован reprojection error;
- реализован reject плохих кадров;
- есть видео с отрисованными осями ворот и estimated pose.

### Milestone 4: gate association

Результат:

- реализован candidate selection по belief state;
- реализован scoring detection-gate;
- реализован Hungarian matching;
- обработка ambiguous cases.

### Milestone 5: sigma_cam estimation

Результат:

- первая эвристическая модель `sigma_cam`;
- reject `sigma_cam > 10 м`;
- отчёт по распределению sigma/reprojection/distance.

### Milestone 6: runtime/export

Результат:

- ONNX export;
- ONNX Runtime inference на Windows;
- замер FPS;
- подготовлен путь к Hailo HEF;
- проверен postprocess keypoints.

### Milestone 7: DCT integration package

Результат:

- CLI генерирует `observations.parquet`;
- DCT может читать observations и inject'ить их в PF;
- проведён end-to-end тест на Liftoff.

---

## 16. Риски и решения

### Риск 1: нет ground truth позиции в первом датасете

Что можно сделать:

- сначала обучить detector/keypoints;
- PnP валидировать по reprojection error и визуально;
- на втором этапе обязательно добавить Liftoff telemetry.

### Риск 2: несколько ворот без маркеров неоднозначны

Решение:

- использовать belief state DCT;
- использовать геометрический score;
- если неоднозначность высокая, лучше reject, чем неверный inject.

### Риск 3: Hailo не поддержит YOLO pose postprocess напрямую

Решение:

- модель на Hailo выдаёт raw tensors;
- keypoint postprocess выполняется на CPU;
- проверить latency.

### Риск 4: `sigma_cam` занижен

Решение:

- всегда калибровать sigma по фактическим ошибкам;
- если сомнение, лучше завысить sigma;
- `sigma_cam > 10 м` reject.

### Риск 5: слишком точная и частая камера ломает PF

Решение:

- inject rate limit: `5-10 Гц`;
- для `sigma_cam < 0.5 м` не inject'ить чаще 10 Гц;
- при необходимости отдельно настроить PF: `random_inject_frac`,
  `process_noise_v`, `n_particles`.

---

## 17. Definition of Done для первой версии

Первая рабочая версия считается готовой, если:

1. Есть датасет минимум 1000 размеченных кадров.
2. YOLOv8n-pose находит ворота и 4 keypoints.
3. PnP строит физически правдоподобную позу по одному gate.
4. Gate association работает при нескольких воротах в кадре.
5. Есть `CameraObservation` API.
6. Есть debug overlay видео.
7. Есть `observations.parquet`.
8. Есть измеренный FPS на Windows.
9. Есть план/скрипт экспорта ONNX -> HEF.
10. Плохие observations корректно reject'ятся.

---

## 18. Что нужно сделать первым

Самый короткий путь к первому результату:

1. Записать 10-20 минут Liftoff-видео на track-002.
2. Извлечь 1000 разнообразных кадров.
3. Разметить `gate + 4 keypoints`.
4. Обучить YOLOv8n-pose `320x320`.
5. Сделать overlay: bbox + keypoints.
6. Подключить PnP по одному заранее известному gate.
7. Затем добавить association нескольких ворот через belief state.

После этого станет понятно, хватает ли `320x320` или нужен `416x416`, и насколько
реально попасть в целевые `sigma_cam <= 1 м`.
