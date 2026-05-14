# Эксперимент 11: журнал интеграции камеры в Replay

Этот файл фиксирует шаги интеграции импортированного модуля камерной
локализации: сначала в Replay, затем в Record.

## 2026-05-14

### Шаг 1 - Область работ

Цель первого этапа реализации:

```text
записанная сессия video.mp4
  -> offline-наблюдения камеры
  -> camera_observations.jsonl
  -> затем фиолетовая стрелка Replay CamKF
```

Важное решение: пока не запускать YOLO внутри GUI-цикла Replay. Сначала
генерировать отдельный файл наблюдений, чтобы timestamps, PnP, sigma и политику
отбраковки можно было отлаживать независимо от отрисовки GUI.

### Шаг 2 - Источник timestamps

Использовать `video_timestamps.parquet` как основной источник timestamps кадров.
Он задаёт соответствие `frame_idx -> ts_wall` и описан в `docs/session_outputs.md`.

Резервная формула только при отсутствии этого файла:

```text
timestamp = first_session_ts + frame_idx / fps
```

Эта резервная формула менее надёжна, поэтому её нужно отражать в summary генератора.

### Шаг 3 - Схема наблюдения

Добавлены `dct.camera_localization.observation.CameraObservation` и JSONL-хелперы.
Первым Replay-артефактом будет:

```text
session/camera_observations.jsonl
```

Каждая строка - одно наблюдение, включая отклонённые наблюдения с `status` и
`reason`, чтобы всю камерную цепочку можно было проверить.

### Шаг 4 - Offline-генератор по видео

Добавлен:

```text
tools/exp11_camera_module_import/generate_camera_observations.py
```

Текущее поведение:

```text
session/video.mp4
  -> timestamps кадров из session/video_timestamps.parquet
  -> YOLO-веса из models/yolo_gate_pose/testgate/weights/best.pt
  -> импортированный адаптер ворот / CoarseRefineLocalizer
  -> session/camera_observations.jsonl
```

Генератор использует telemetry `pos_x/pos_y/pos_z` как первый грубый prior. Это
допустимо для первичной offline-проверки артефакта, но в Replay-интеграции этот
prior должен быть заменён на live-prior от `OnlineLocalizer`/`KFLayer2`.

### Шаг 5 - Инфраструктура Replay HUD/Preview

Добавлена GUI-инфраструктура для будущего Replay-контура с камерой:

- строка `CamKF` в HUD и слой маркера на карте;
- фиолетовые цвета стрелки и trail на карте;
- API для overlay ворот в `VideoPreviewWidget` для YOLO bbox/keypoints;
- checkbox `CamKF` в HUD включает и выключает overlay на видео.

Решение по поведению:

```text
CamKF включён  -> Video Preview может рисовать YOLO bbox/keypoints.
CamKF выключен -> Video Preview показывает исходное видео как раньше.
```

Чтение камерных observations и вычисление CamKF в Replay остаются следующим шагом.
На этом этапе подготовлены только визуальный слой и переключатель on/off.

### Шаг 6 - Overlay наблюдений в Replay

Добавлена загрузка на стороне Replay:

```text
session/camera_observations.jsonl
```

Файл читается при выборе Replay-сессии. Для каждого telemetry/video update
ближайшее наблюдение по `ts_wall` преобразуется в overlay для `VideoPreview`,
если оно не старше текущего допуска.

Генератор и схема теперь содержат:

```text
bbox_xyxy
keypoints
```

Благодаря этому Video Preview может рисовать YOLO-рамки ворот и углы. Строка
`CamKF` в HUD по-прежнему управляет видимостью overlay. Когда `CamKF` выключен,
replay-видео не меняется.

Также исправлен расчёт высоты HUD: постоянная строка `CamKF` больше не выходит
за пределы панели.

## 2026-05-15 - Отладка Replay CamKF overlay

Первый Replay-тест не показал YOLO overlay при включённом `CamKF`, потому что в
выбранной сессии ещё не было:

```text
camera_observations.jsonl
```

Для сессии был сгенерирован sampled-файл наблюдений. Логика Replay в GUI теперь
также перезагружает `camera_observations.jsonl`, когда включается `CamKF`, но
наблюдения ещё не загружены. Поиск overlay теперь использует ближайший timestamp
camera observation вокруг текущего времени Replay-видео, а не только предыдущий
telemetry timestamp.

## 2026-05-15 - Replay-контур локализации CamKF

Добавлен отдельный экспериментальный Replay-контур:

```text
LF sticks -> OnlineLocalizer(cam) -> camera observation inject -> KFLayer2(cam) -> CamKF
```

Контур независим от существующих LF/RC/KF localizers. Он использует загруженный
`camera_observations.jsonl` только для наблюдений с `status == "ok"` и
`sigma_cam <= 10 m`, а отклонённые наблюдения остаются полезными для отладки
video overlay. Финальная позиция CamKF рисуется существующим фиолетовым слоем
стрелки/trail и выводится в строке HUD `CamKF`.

Добавлена короткая красная вспышка стрелки CamKF при каждом принятом camera
observation, injected в экспериментальный localizer. Это делает успешные
подтверждения позиции от камеры видимыми прямо на карте без изменения цвета
фиолетового trail.

После первого визуального Replay-теста окно timestamp для camera injection было
расширено с `0.05 s` до `0.2 s`, а также добавлена защита от повторного inject
одного и того же observation frame. Так редкие принятые observations легче
увидеть как красные вспышки CamKF, при этом сохраняется правило: один inject на
один camera frame.

Overlay камеры в Video Preview изменён: теперь рисуются только красные рамки
ворот и углы. Текстовые подписи gate ID и confidence скрыты, чтобы overlay было
легче читать.

Добавлено INFO-логирование каждого Replay CamKF inject. Для каждого принятого
camera inject теперь записываются timestamp, исходный frame, связанные gate ids,
camera sigma, camera XYZ, позиция CamKF PF до/после injection, uncertainty
до/после и XZ delta, вызванная camera update.

Replay CamKF переключён с LF telemetry sticks на тот же источник RC batch/stick,
который использует существующий красный контур `KF`. Теперь `KF` и `CamKF`
сравнимы: намеренное отличие между ними - только экспериментальный путь camera
injection.

Дедупликация Replay camera-inject изменена с "последний injected frame" на set
всех injected `frame_idx` для текущего replay run. Это предотвращает ситуацию,
когда соседние принятые observations чередуются и один camera frame inject'ится
несколько раз.

Добавлен консервативный Replay CamKF camera gating, чтобы уменьшить скачки на
карте:

```text
inject только если XZ innovation <= 10 m
sigma_eff = max(sigma_cam * 2.0, 5.0 m)
```

Отклонённые camera updates логируются как `CamKF inject skipped` с innovation
distance. Принятые camera updates логируются и с raw `sigma_cam`, и с effective
`sigma_eff`, который используется для PF injection.

После визуального тестирования gate изменён на более мягкий компромисс:

```text
inject только если XZ innovation <= 15 m
sigma_eff = max(sigma_cam * 1.5, 4.0 m)
```

