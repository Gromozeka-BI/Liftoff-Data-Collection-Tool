# Data Collection Toolkit (DCT)

**FPV Drone Localization System — Этап 0**

## Установка

```bash
pip install -e .
```

## Команды

```bash
# Записать сессию
dct record --pilot A --drone cinemarc --track track-001 --purpose "baseline"

# Список сессий
dct list

# Просмотр сессии
dct inspect sessions/2026-04-29_pilot-A_drone-cinemarc_track-track001_session-001

# Валидация
dct validate sessions/2026-04-29_...

# Replay
dct replay sessions/2026-04-29_... --speed 2.0
```

### Флаги `dct record`

| Флаг | Описание |
|------|----------|
| `--no-video` | Отключить запись экрана |
| `--no-rh-sim` | Отключить мок RotorHazard |

## Переменные окружения (`.env`)

```
DCT_UDP_PORT=9001
DCT_API_PORT=8765
DCT_SESSIONS_DIR=sessions
DCT_SCREEN_FPS=60
DCT_SCREEN_WINDOW_TITLE=Liftoff
DCT_RH_GATE_RADIUS=2.0
```

## Структура сессии

```
sessions/
└── YYYY-MM-DD_pilot-X_drone-Y_track-Z_session-NNN/
    ├── meta.json          # метаданные
    ├── telemetry.parquet  # ~100 Гц телеметрия (snappy)
    ├── events.parquet     # дискретные события
    ├── track.json         # копия трассы
    ├── video.mp4          # запись экрана 720p/60fps
    └── notes.md           # заметки оператора
```

## REST API (порт 8765)

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/v1/status` | Статус сессии |
| POST | `/api/v1/rh/lap` | Событие RotorHazard: круг |
| POST | `/api/v1/rh/gate` | Событие RotorHazard: ворота |
| POST | `/api/v1/button/lap` | Кнопка: круг |
| POST | `/api/v1/button/gate` | Кнопка: конкретные ворота |

## Критерии валидации

- Частота телеметрии ≥ 95 Гц
- Пропущенных seq ≤ 1%
- Дрейф времени ≤ 100 мс
- Минимум 1 круг
