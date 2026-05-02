# Data Collection Toolkit (DCT)

> Система сбора телеметрических данных для FPV-дрона.  
> Записывает данные Liftoff-симулятора, аппаратуры управления (RC/ELRS), видео с экрана или карты захвата, синхронизирует их на общей временной шкале и предоставляет GUI для мониторинга в реальном времени и воспроизведения.

---

## Содержание

1. [Требования](#1-требования)
2. [Установка](#2-установка)
3. [Аппаратная конфигурация](#3-аппаратная-конфигурация)
4. [Профили и трассы](#4-профили-и-трассы)
5. [Запуск](#5-запуск)
6. [Графический интерфейс (GUI)](#6-графический-интерфейс-gui)
   - [Режим Record](#61-режим-record)
   - [Режим Replay](#62-режим-replay)
   - [Графики стиков](#63-графики-стиков)
   - [Настройки интерфейса](#64-настройки-интерфейса)
7. [CLI-команды](#7-cli-команды)
8. [Конфигурация](#8-конфигурация)
9. [Структура сессии](#9-структура-сессии)
10. [REST API](#10-rest-api)
11. [Валидация](#11-валидация)
12. [Лог-файлы](#12-лог-файлы)
13. [Зависимости](#13-зависимости)
14. [Структура проекта](#14-структура-проекта)

---

## 1. Требования

### Операционная система
- **Windows 10 / 11** (основная платформа; захват экрана через `mss`/GDI)
- Python **3.10+**

### Программное обеспечение
| Компонент | Версия | Назначение |
|-----------|--------|-----------|
| Python | ≥ 3.10 | Интерпретатор |
| Liftoff | любая | FPV-симулятор (источник телеметрии по UDP) |

### Аппаратура (опционально)
| Компонент | Назначение |
|-----------|-----------|
| RC-передатчик (ELRS) | Пульт управления дроном |
| ESP32 + ELRS-приёмник | Считывает каналы аппаратуры и шлёт CSV по UART 115 200 бод |
| Карта захвата (HDZero / HDMI) | Запись FPV-видео с реального дрона вместо экрана |

---

## 2. Установка

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd DCT

# 2. Создать виртуальное окружение (рекомендуется)
python -m venv .venv
.venv\Scripts\activate      # Windows

# 3. Установить проект и все зависимости
pip install -e .
```

После установки становится доступна команда `dct` в терминале.

---

## 3. Аппаратная конфигурация

### ESP32 (RC-приёмник через UART)

ESP32 должен слать строки в формате CSV через USB-UART со скоростью **115 200 бод**:

```
<timestamp_us>,<ch1>,<ch2>,<ch3>,<ch4>,<ch5>,<ch6>,<ch7>,<ch8>\n
```

- `timestamp_us` — значение `micros()` (мкс с момента загрузки ESP32)
- `ch1`–`ch8` — значения каналов в диапазоне **988–2012** (типично 1000–2000)
- Частота: **100 Гц** (одна строка каждые 10 мс)

**Маппинг каналов по умолчанию:**

| Канал ESP32 | Стик | Функция |
|-------------|------|---------|
| ch1 | Правый горизонталь | Roll |
| ch2 | Правый вертикаль | Pitch |
| ch3 | Левый вертикаль | Throttle |
| ch4 | Левый горизонталь | Yaw |
| ch5 | Тумблер | Arm |
| ch6 | Тумблер | Turtle mode |
| ch7 | Тумблер | Option |
| ch8 | Тумблер | Rate profile |

### Liftoff UDP телеметрия

В настройках Liftoff включите **UDP Telemetry** на порт **9001** (адрес 127.0.0.1).

### Карта захвата (HDZero / HDMI)

Подключите устройство захвата до запуска DCT. В GUI нажмите кнопку **↻** в секции «Video source», чтобы обнаружить устройство. Устройства DirectShow/UVC определяются автоматически через OpenCV.

---

## 4. Профили и трассы

DCT использует JSON-профили для идентификации пилота, дрона, камеры и скоростных настроек.

### Структура директорий профилей

```
profiles/
├── pilots.json              # список пилотов
├── drones/
│   ├── cinemarc.json
│   └── toothpick.json
├── rates/
│   ├── betaflight-default.json
│   └── kiss-race.json
└── cameras/
    ├── runcam-thumb.json
    └── gopro-hero11.json

tracks/
├── track-001.json
└── oval.json
```

### Формат `pilots.json`

```json
[
  { "id": "pilot-A", "nickname": "AlphaRacer", "name": "Алексей" },
  { "id": "pilot-B", "nickname": "BetaFly",    "name": "Борис" }
]
```

### Формат профиля дрона (`drones/cinemarc.json`)

```json
{
  "id": "cinemarc",
  "name": "CinemaRC 5\"",
  "weight_g": 580,
  "motors": "2306 2400kv",
  "props": "5x4.5x3",
  "fc": "SpeedyBee F7"
}
```

### Формат трассы (`tracks/track-001.json`)

```json
{
  "id": "track-001",
  "name": "Oval Track",
  "gates": [
    { "id": 0, "label": "S/F", "position": [0, 0, 0], "is_start_finish": true, "check_radius": 3.0 },
    { "id": 1, "label": "Gate 1", "position": [10, 0, 5], "check_radius": 2.0 },
    { "id": 2, "label": "Gate 2", "position": [20, 0, 0], "check_radius": 2.0 }
  ]
}
```

---

## 5. Запуск

```bash
# Запустить графический монитор (основной режим работы)
dct monitor
```

Откроется главное окно приложения с двумя режимами: **Record** и **Replay**.

---

## 6. Графический интерфейс (GUI)

### 6.1 Режим Record

Нижняя панель «Record» содержит следующие секции:

#### Session config
Выпадающие списки для выбора:
- **Pilot** — пилот из `profiles/pilots.json`
- **Drone** — профиль дрона из `profiles/drones/`
- **Rate** — профиль ставок из `profiles/rates/`
- **Camera** — профиль камеры из `profiles/cameras/`
- **Track** — трасса из `tracks/`

#### Data sources
| Режим | Описание |
|-------|----------|
| **Liftoff only** | Телеметрия только из Liftoff по UDP |
| **RC only** | Только данные аппаратуры (ESP32 по UART) |
| **Liftoff + RC** | Оба источника одновременно |

При выборе режима с RC появляются:
- Выпадающий список COM-портов с кнопкой **↻** обновления
- Индикатор соединения (🔴 offline / 🟢 online)

#### Video source
- Выпадающий список источника видео: экран Liftoff или устройство захвата
- Кнопка **↻** — сканировать DirectShow-устройства
- Строка с путём папки сессий + кнопка **📁** выбора папки

#### Controls — горячие клавиши

| Кнопка | Клавиша | Действие |
|--------|---------|----------|
| START  | `6`     | Начать запись сессии |
| STOP   | `7`     | Остановить запись |
| LAP    | `8`     | Отметить круг |
| GATE   | `9`     | Отметить ближайшие ворота |
| S/F    | `0`     | Отметить ворота Start/Finish |

> Горячие клавиши работают **глобально** (даже когда Liftoff в фокусе).

#### Индикаторы статуса
- Частота телеметрии (Гц)
- Количество пакетов / кругов / пропущенных пакетов
- Продолжительность сессии
- Количество записанных видеофреймов

### 6.2 Режим Replay

1. Переключитесь в режим **▶ REPLAY** кнопкой в верхней панели
2. Выберите папку сессий кнопкой **📁**
3. Выберите сессию в выпадающем списке
4. Управляйте воспроизведением:

| Элемент | Действие |
|---------|----------|
| **▶ Play / ⏸ Pause** | Пробел или кнопка |
| Скруббер | Перемотка по временной шкале |
| **0.5× / 1× / 2×** | Скорость воспроизведения |

При загрузке сессии автоматически восстанавливаются настройки реверса осей, действовавшие во время записи.

### 6.3 Графики стиков

Область графиков отображает последние **10 секунд** данных в режиме реального времени.

#### Структура

```
[⇔ Merge Liftoff + RC]   ← переключатель наложения

["Liftoff"  колонка] ["RC"  колонка]  [Реверс]
 T — Throttle         Thr — ch3        T ☐  Thr ☐
 Y — Yaw              Yaw — ch4        Y ☐  Yaw ☐
 P — Pitch            Pit — ch2        P ☐  Pit ☐
 R — Roll             Rol — ch1        R ☐  Rol ☐

[    Комбинированный график (все каналы)    ]

[CH5-Arm] [CH6-Turtle] [CH7-Option] [CH8-Rate]
```

#### Режим Merge (⇔)
- Колонка RC скрывается
- RC-каналы нормализуются в диапазон −1…1 и отображаются пунктиром поверх LF-графиков
- Маппинг: ch3→T, ch4→Y, ch2→P, ch1→R

#### Индикаторы переключателей
| Цвет | Значение канала |
|------|----------------|
| 🔴 Красный | < 1300 (LOW) |
| 🟡 Жёлтый | 1300–1700 (MID) |
| 🟢 Зелёный | > 1700 (HIGH) |

#### Реверс осей
Чекбоксы в панели **«Реверс»** инвертируют отображение оси без изменения записываемых данных.
- Настройка сохраняется между запусками
- При записи состояние чекбоксов сохраняется в сессию (`invert.json`)
- При воспроизведении настройки восстанавливаются автоматически

### 6.4 Настройки интерфейса

Все настройки GUI автоматически сохраняются в `ui_settings.json` и восстанавливаются при следующем запуске:

| Настройка | Ключ в JSON |
|-----------|------------|
| Пилот | `pilot` |
| Дрон | `drone` |
| Ставки | `rate` |
| Камера | `camera` |
| Трасса | `track` |
| Режим источников данных | `ds_mode` |
| COM-порт | `com_port` |
| Индекс видеоустройства | `video_source_index` |
| Папка записи сессий | `sessions_dir` |
| Папка сессий Replay | `replay_dir` |
| Настройки реверса осей | `invert.lf.*`, `invert.rc.*` |

---

## 7. CLI-команды

### `dct monitor`
```
Описание:  Запустить графическое окно DCT
Использование: dct monitor
```

### `dct record`
```
Описание:  Записать новую сессию (консольный режим без GUI)
Использование: dct record [OPTIONS]

Опции:
  -p, --pilot TEXT      Идентификатор пилота (обязательно)
  -d, --drone TEXT      Название профиля дрона (обязательно)
  -t, --track TEXT      ID трассы или путь к track.json (обязательно)
  --purpose TEXT        Метка цели сессии [по умолчанию: training]
  --no-video            Отключить запись экрана
  --no-rh-sim           Отключить симулятор RotorHazard
```

Пример:
```bash
dct record --pilot AlphaRacer --drone cinemarc --track track-001 --purpose "baseline"
dct record -p A -d toothpick -t oval --no-video
```

### `dct list`
```
Описание:  Показать таблицу всех записанных сессий
Использование: dct list [OPTIONS]

Опции:
  --sessions-dir PATH   Переопределить папку сессий
```

Пример:
```bash
dct list
dct list --sessions-dir D:\my-sessions
```

### `dct inspect`
```
Описание:  Подробная информация о конкретной сессии
Использование: dct inspect SESSION_PATH

Аргументы:
  SESSION_PATH   Путь к директории сессии (обязательно)
```

Пример:
```bash
dct inspect sessions/2026-05-01_pilot-A_drone-cinemarc_track-track-001_session-001
```

Выводит: метаданные, кол-во строк, длительность, частоту дискретизации, статистику событий, размер видео.

### `dct validate`
```
Описание:  Повторно запустить валидацию для уже записанной сессии
Использование: dct validate SESSION_PATH

Аргументы:
  SESSION_PATH   Путь к директории сессии (обязательно)
```

Пример:
```bash
dct validate sessions/2026-05-01_pilot-A_drone-cinemarc_track-track-001_session-001
```

### `dct align`
```
Описание:  Показать статистику синхронизации видео и телеметрии
Использование: dct align SESSION_PATH

Аргументы:
  SESSION_PATH   Путь к директории сессии (обязательно)
```

Пример:
```bash
dct align sessions/2026-05-01_...
```

Выводит: кол-во фреймов, среднее/медианное/P95/максимальное отклонение Δt между видеофреймом и ближайшим пакетом телеметрии.

| Оценка | Критерий |
|--------|----------|
| Отлично | P95 < 10 мс |
| Хорошо | P95 < 33 мс (< 1 видеофрейм) |
| Плохо | P95 ≥ 33 мс |

---

## 8. Конфигурация

### Файл `.env` (переменные окружения)

Создайте файл `.env` в корне проекта для переопределения параметров по умолчанию:

```ini
# UDP-порт для телеметрии Liftoff
DCT_UDP_PORT=9001

# Адрес UDP (оставьте 127.0.0.1 для локального Liftoff)
DCT_UDP_HOST=127.0.0.1

# REST API (RotorHazard-совместимый mock)
DCT_API_HOST=0.0.0.0
DCT_API_PORT=8765

# Папка по умолчанию для сохранения сессий
DCT_SESSIONS_DIR=sessions

# Запись видео
DCT_SCREEN_FPS=30
DCT_SCREEN_WIDTH=1280
DCT_SCREEN_HEIGHT=720
DCT_SCREEN_WINDOW_TITLE=Liftoff

# Радиус детекции ворот (метры, для mock RH-sim)
DCT_RH_GATE_RADIUS=2.0

# Запись Parquet (сброс буфера)
DCT_PARQUET_FLUSH_ROWS=500
DCT_PARQUET_FLUSH_INTERVAL=2.0
```

### Приоритет настроек

```
Переменные окружения  >  .env файл  >  Значения по умолчанию в config.py
```

### Файл `ui_settings.json`

Создаётся автоматически в рабочей директории при первом изменении любой настройки GUI. Редактировать вручную не требуется.

```json
{
  "pilot": "AlphaRacer",
  "drone": "CinemaRC 5\"",
  "rate": "Betaflight Default",
  "camera": "RunCam Thumb",
  "track": "Oval Track",
  "ds_mode": "both",
  "com_port": "COM3",
  "sessions_dir": "D:\\sessions",
  "replay_dir": "D:\\sessions",
  "video_source_index": null,
  "invert": {
    "lf": { "in_throttle": false, "in_yaw": false, "in_pitch": true, "in_roll": false },
    "rc": { "ch3": false, "ch4": false, "ch2": true, "ch1": false }
  }
}
```

---

## 9. Структура сессии

```
sessions/
└── 2026-05-01_pilot-AlphaRacer_drone-cinemarc_track-track-001_session-001/
    ├── meta.json                  # метаданные сессии
    ├── telemetry.parquet          # LF-телеметрия ~100 Гц (snappy-сжатие)
    ├── rc_channels.parquet        # RC-каналы 100 Гц (только при ds_mode=rc/both)
    ├── timeline.parquet           # единая временна́я шкала ~30 Гц
    ├── events.parquet             # дискретные события (круги, ворота)
    ├── system.jsonl               # системный лог в формате JSONL
    ├── track.json                 # копия трассы
    ├── invert.json                # состояние реверса осей при записи
    ├── video.mp4                  # видео (1280×720 / 30fps, H.264)
    └── video_timestamps.parquet   # метки времени видеофреймов
```

### Схема `telemetry.parquet`

| Поле | Тип | Описание |
|------|-----|---------|
| `seq` | int64 | Порядковый номер пакета |
| `ts_wall` | float64 | Время PC (Unix timestamp, с) |
| `ts_sim` | float32 | Внутреннее время симулятора (с) |
| `pos_x/y/z` | float32 | Позиция (м) |
| `att_x/y/z/w` | float32 | Кватернион ориентации |
| `vel_x/y/z` | float32 | Скорость (м/с) |
| `gyro_pitch/roll/yaw` | float32 | Угловые скорости (°/с) |
| `in_throttle/yaw/pitch/roll` | float32 | Входы стиков (−1…1) |
| `bat_v` | float32 | Напряжение АКБ (В) |
| `bat_pct` | float32 | Заряд АКБ (%) |
| `motor_0/1/2/3` | float32 | Обороты моторов (RPM) |

### Схема `rc_channels.parquet`

| Поле | Тип | Описание |
|------|-----|---------|
| `seq` | int64 | Порядковый номер пакета |
| `ts_wall` | float64 | Время PC (Unix timestamp, с) |
| `ts_device_us` | int64 | Время ESP32 `micros()` (мкс) |
| `ch1`–`ch8` | int32 | Значения каналов (988–2012) |

### Схема `events.parquet`

| Поле | Тип | Описание |
|------|-----|---------|
| `seq` | int64 | Порядковый номер события |
| `ts_wall` | float64 | Время события |
| `event_type` | string | Тип: `session_start`, `session_stop`, `lap`, `gate_pass` |
| `gate_id` | int32 | ID ворот (для `gate_pass`) |
| `lap_num` | int32 | Номер круга (для `lap`) |
| `source` | string | Источник: `dct`, `api`, `rh_sim` |

### Формат `system.jsonl`

Одна запись JSON на строку:
```jsonl
{"ts": 1746000000.123, "event": "sources_configured", "data_source": "both", "video_source": "screen", "rc_port": "COM3"}
{"ts": 1746000001.456, "event": "rc_status", "online": true}
{"ts": 1746000061.789, "event": "session_stop"}
```

### Формат `meta.json`

```json
{
  "version": "0.1",
  "session_id": "2026-05-01_pilot-AlphaRacer_..._session-001",
  "pilot": "AlphaRacer",
  "drone": "cinemarc",
  "track": "track-001",
  "purpose": "training",
  "created_at": "2026-05-01T12:00:00+00:00",
  "finished_at": "2026-05-01T12:05:30+00:00",
  "duration_s": 330.0,
  "total_packets": 33000,
  "total_laps": 5,
  "validated": true
}
```

---

## 10. REST API

DCT поднимает локальный HTTP-сервер (RotorHazard-совместимый mock) на порту **8765**.

| Метод | Путь | Тело | Описание |
|-------|------|------|---------|
| `GET` | `/api/v1/status` | — | Статус и статистика сессии |
| `POST` | `/api/v1/rh/lap` | `{"ts_wall": <float>}` | Событие круга от RotorHazard |
| `POST` | `/api/v1/rh/gate` | `{"gate_id": <int>, "ts_wall": <float>}` | Проход ворот от RH |
| `POST` | `/api/v1/button/lap` | `{"ts_wall": <float>}` | Ручная отметка круга |
| `POST` | `/api/v1/button/gate` | `{"gate_id": <int>, "ts_wall": <float>}` | Ручная отметка ворот |

Пример запроса:
```bash
curl -X POST http://localhost:8765/api/v1/button/lap \
     -H "Content-Type: application/json" \
     -d '{"ts_wall": 1746000030.5}'
```

---

## 11. Валидация

Валидация запускается автоматически после каждой сессии. Также можно запустить вручную: `dct validate <session>`.

### Критерии для Liftoff-телеметрии

| Проверка | Порог | Описание |
|---------|-------|---------|
| Минимум пакетов | ≥ 50 | Сессия не пустая |
| Частота дискретизации | ≥ 95 Гц | Стабильный поток UDP |
| Пропущенные seq | ≤ 1% | Потери в UDP-очереди |
| Дрейф времени | ≤ 100 мс | Монотонность `ts_wall` |

### Критерии для RC-каналов

| Проверка | Описание |
|---------|---------|
| Движение стиков | Хотя бы один из ch1–ch4 изменился на ≥ 200 ед. в любом 3-секундном окне |

### Автоудаление пустых сессий

Сессия удаляется автоматически если выполнены **оба** условия:
- Не зафиксировано ни одного прохода ворот
- RC-валидация не пройдена (аппаратура не двигалась)

---

## 12. Лог-файлы

Каждый запуск DCT создаёт **новый** лог-файл в директории `logs/`:

```
logs/
├── dct_2026-05-01_10-30-00.log
├── dct_2026-05-01_14-15-22.log
└── dct_2026-05-02_09-05-11.log
```

Формат строки:
```
2026-05-01 10:30:01.234  DEBUG     dct.rc_receiver  RC: connected to COM3 @115200 baud
2026-05-01 10:30:02.100  INFO      dct.data_source  Session started: sessions/...
2026-05-01 10:30:02.345  WARNING   dct.validator    Validation issue: hz=94.8 < 95
```

- Консоль: уровень `INFO` и выше
- Файл: уровень `DEBUG` и выше

---

## 13. Зависимости

Все зависимости устанавливаются автоматически через `pip install -e .`

| Библиотека | Версия | Назначение |
|-----------|--------|-----------|
| `click` | ≥ 8.1 | CLI-интерфейс |
| `fastapi` | ≥ 0.111 | REST API (RotorHazard mock) |
| `uvicorn[standard]` | ≥ 0.29 | ASGI-сервер для FastAPI |
| `pyarrow` | ≥ 16.0 | Чтение/запись Parquet |
| `pydantic` | ≥ 2.7 | Валидация данных и схем |
| `pydantic-settings` | ≥ 2.3 | Загрузка конфигурации из `.env` |
| `mss` | ≥ 9.0 | Захват экрана (GDI/DXGI) |
| `opencv-python-headless` | ≥ 4.9 | Кодирование видео, работа с устройствами захвата |
| `numpy` | ≥ 1.26 | Числовые вычисления, обработка данных |
| `rich` | ≥ 13.7 | Красивый вывод в терминал |
| `pygetwindow` | ≥ 0.0.9 | Поиск окна Liftoff (только Windows) |
| `PyQt6` | ≥ 6.6 | GUI-фреймворк |
| `pyqtgraph` | ≥ 0.13 | Высокопроизводительные графики в реальном времени |
| `keyboard` | ≥ 0.13 | Глобальные горячие клавиши (только Windows) |
| `pyserial` | ≥ 3.5 | Чтение данных с ESP32 по UART |

---

## 14. Структура проекта

```
DCT/
├── pyproject.toml              # метаданные проекта и зависимости
├── .env                        # локальная конфигурация (не в git)
├── ui_settings.json            # настройки GUI (создаётся автоматически)
│
├── profiles/                   # профили пилотов, дронов, камер, ставок
│   ├── pilots.json
│   ├── drones/
│   ├── rates/
│   └── cameras/
│
├── tracks/                     # JSON-описания трасс
│
├── sessions/                   # записанные сессии (по умолчанию)
│
├── logs/                       # лог-файлы (создаются автоматически)
│
└── dct/                        # исходный код пакета
    ├── cli.py                  # точка входа CLI
    ├── config.py               # pydantic-настройки
    ├── log.py                  # конфигурация логирования
    ├── session.py              # создание/финализация директорий сессий
    ├── validator.py            # валидация записанных данных
    ├── rc_receiver.py          # чтение ESP32 по UART (100 Гц)
    ├── screen_recorder.py      # запись экрана / карты захвата
    ├── rh_simulator.py         # mock RotorHazard
    ├── system_logger.py        # JSONL системный лог
    ├── video_preview_source.py # превью-захват без записи
    ├── align.py                # статистика синхронизации видео
    │
    ├── commands/               # CLI-команды
    │   ├── monitor.py          # dct monitor
    │   ├── record.py           # dct record
    │   ├── list_cmd.py         # dct list
    │   ├── inspect.py          # dct inspect
    │   ├── validate.py         # dct validate
    │   ├── align_cmd.py        # dct align
    │   └── replay.py           # dct replay (legacy CLI)
    │
    ├── receivers/
    │   ├── liftoff_udp.py      # UDP-приёмник телеметрии Liftoff
    │   └── button_api.py       # HTTP-клиент REST API событий
    │
    ├── storage/
    │   ├── schema.py           # PyArrow-схемы Parquet
    │   └── writer.py           # потоковая запись Parquet
    │
    └── gui/
        ├── app.py              # точка входа GUI
        ├── theme.py            # цветовая тема
        ├── ui_settings.py      # load/save ui_settings.json
        ├── main_window.py      # главное окно QMainWindow
        ├── data_source.py      # LiveDataSource / ReplayDataSource
        ├── global_hotkeys.py   # глобальные горячие клавиши
        ├── video_reader.py     # чтение видео при Replay
        │
        └── widgets/
            ├── record_bar.py   # нижняя панель режима Record
            ├── replay_bar.py   # нижняя панель режима Replay
            ├── stick_graphs.py # графики стиков (LF + RC)
            ├── track_map.py    # карта трассы с траекторией
            ├── video_preview.py# виджет предпросмотра видео
            └── status_panel.py # панель статуса
```

---

## Быстрый старт

```bash
# 1. Установить
pip install -e .

# 2. Создать профили
mkdir -p profiles/drones profiles/rates profiles/cameras

# profiles/pilots.json
echo '[{"id":"pilot-A","nickname":"Pilot A"}]' > profiles/pilots.json

# profiles/drones/myquad.json
echo '{"id":"myquad","name":"My Quad 5\""}' > profiles/drones/myquad.json

# profiles/rates/default.json
echo '{"id":"default","name":"Default"}' > profiles/rates/default.json

# profiles/cameras/naked.json
echo '{"id":"naked","name":"Naked GoPro"}' > profiles/cameras/naked.json

# 3. Запустить Liftoff с UDP-телеметрией на порт 9001

# 4. Запустить DCT
dct monitor
```
