# DCT: входы и выходы (контракт данных)

Дата: 2026-05-04

Этот документ фиксирует, какие данные DCT принимает на вход, что формирует на выход, и что означает каждое поле.

## 1) Входные данные

### 1.1 Liftoff UDP telemetry

- Транспорт: UDP
- По умолчанию: `127.0.0.1:9001`
- Частота: около 100 Гц
- Назначение: основная телеметрия полета из симулятора

Пакет (little-endian) содержит:
- `ts_sim` (float32) - время симулятора, сек
- `pos_x`, `pos_y`, `pos_z` (float32) - позиция, м
- `att_x`, `att_y`, `att_z`, `att_w` (float32) - ориентация (кватернион)
- `vel_x`, `vel_y`, `vel_z` (float32) - скорость, м/с
- `gyro_pitch`, `gyro_roll`, `gyro_yaw` (float32) - угловая скорость, град/с
- `in_throttle`, `in_yaw`, `in_pitch`, `in_roll` (float32) - входы стиков (обычно -1..1)
- опционально `bat_v`, `bat_pct` (float32)
- опционально motor block: количество моторов + RPM

Что добавляет DCT:
- `seq` - монотонный номер пакета в сессии
- `ts_wall` - время ПК (`time.time()`, Unix seconds)

### 1.2 RC по UART (ESP32/ELRS)

- Транспорт: serial/COM
- Скорость: 115200 baud
- Формат строки:
  - `<timestamp_us>,<ch1>,<ch2>,<ch3>,<ch4>,<ch5>,<ch6>,<ch7>,<ch8>`

Поля:
- `timestamp_us` - `micros()` на устройстве, мкс
- `ch1..ch8` - значения каналов (обычно ~988..2012)

Как формируется время:
- На первом пакете берется якорь `T_anchor = time.time()`
- Далее `ts_wall` вычисляется из `micros()` с учетом rollover `2^32`
- Это снижает джиттер/дрейф против `time.time()` на каждый пакет

### 1.3 HTTP события (дискретные источники)

- Транспорт: HTTP (FastAPI)
- По умолчанию: `0.0.0.0:8765`
- Эндпоинты:
  - `GET /api/v1/status`
  - `POST /api/v1/rh/lap`
  - `POST /api/v1/rh/gate`
  - `POST /api/v1/button/lap`
  - `POST /api/v1/button/gate`

Основные поля тела:
- `gate_id` (int, для gate и опционально для lap)
- `ts_wall` (float, сек; если не передан - ставится текущее время)
- `pilot` (опционально)

### 1.4 Видео вход

- `screen` - захват окна/экрана
- `device` - USB capture card / camera

Ключевой момент:
- Для каждого кадра записывается `ts_wall` в отдельный parquet
- Синхронизация делается по timestamps, а не по fps заголовка видео

## 2) Выходные данные (артефакты сессии)

Сессия создается в `sessions/<session_id>/`.

### 2.1 `meta.json`

Метаданные сессии:
- `version` - версия формата
- `session_id` - id сессии
- `pilot`, `drone`, `track`, `purpose`
- `created_at`, `finished_at` (ISO time)
- `duration_s` - длительность, сек
- `total_packets` - количество Liftoff-пакетов
- `total_laps` - количество кругов
- `validated` - флаг валидации

### 2.2 `telemetry.parquet`

Поля:
- `seq` (int64) - порядковый номер
- `ts_wall` (float64) - время ПК, Unix sec
- `ts_sim` (float32) - время симулятора, sec
- `pos_x/y/z` (float32) - позиция, м
- `att_x/y/z/w` (float32) - кватернион ориентации
- `vel_x/y/z` (float32) - скорость, м/с
- `gyro_pitch/roll/yaw` (float32) - град/с
- `in_throttle/yaw/pitch/roll` (float32) - стики
- `bat_v` (float32) - вольтаж, В
- `bat_pct` (float32) - заряд, %
- `motor_0..motor_3` (float32) - RPM (или NaN если не пришли)

### 2.3 `rc_channels.parquet`

Поля:
- `seq` (int64) - номер RC-пакета
- `ts_wall` (float64) - время ПК
- `ts_device_us` (int64) - время устройства в мкс
- `ch1..ch8` (int32) - значения каналов

### 2.4 `events.parquet`

Поля:
- `seq` (int64) - номер события
- `ts_wall` (float64) - время события
- `event_type` (string) - тип (`rh_lap`, `rh_gate`, `button_lap`, `button_gate`, `session_start`, `session_stop`)
- `gate_id` (int32) - id ворот или `-1`
- `lap_num` (int32) - номер круга или `-1`
- `source` (string) - источник события

### 2.5 `events_edited.parquet` (опционально)

- Создается при редактировании событий в Replay
- Схема полей та же, что у `events.parquet`
- Нужен, чтобы сохранить исходные события неизменными

### 2.6 `timeline.parquet`

Поля:
- `seq` (int64) - номер тика
- `ts_wall` (float64) - время тика

Назначение:
- Основная шкала времени для Replay/seek
- Позволяет replay работать даже в RC-only режиме

### 2.7 `video.mp4`

- Контейнер mp4, кодек H.264
- Видео поток для Record/Replay

### 2.8 `video_timestamps.parquet`

Поля:
- `frame_idx` (int64) - индекс кадра (0-based)
- `ts_wall` (float64) - точное время кадра на ПК

Назначение:
- Точная синхронизация видео с телеметрией и событиями

### 2.9 `track.json`

- Копия конфигурации трассы для воспроизводимости
- Обычно содержит `gates`, позиции, S/F, радиусы проверки

### 2.10 `invert.json`

- Сохраненное состояние инверсий осей графиков на момент записи
- Нужен для правильного восстановления вида в Replay

### 2.11 `system.jsonl`

- Технические системные записи (JSON per line)
- Примеры: конфигурация источников, статус RC online/offline

## 3) Режимы и что получается на выходе

### Liftoff only

Вход:
- UDP Liftoff
- (опционально) видео
- (опционально) HTTP события

Выход:
- `telemetry.parquet`
- `events.parquet`
- `timeline.parquet`
- `meta.json`
- + видео файлы, если запись видео включена

### RC only

Вход:
- UART RC
- (опционально) видео
- (опционально) HTTP события

Выход:
- `rc_channels.parquet`
- `events.parquet`
- `timeline.parquet`
- `meta.json`
- + видео файлы, если запись видео включена

### Liftoff + RC

Вход:
- UDP + UART + HTTP + (опционально) видео

Выход:
- `telemetry.parquet`, `rc_channels.parquet`, `events.parquet`, `timeline.parquet`, `meta.json`
- + видео файлы, если запись видео включена

## 4) Дополнительно: CLI Replay output

Для `dct replay`:
- Вход: `telemetry.parquet`
- Выход: UDP поток в wire-формате Liftoff на указанный host/port
- Скорость отправки управляется параметром `--speed`

