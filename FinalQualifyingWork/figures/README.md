# Рисунки для ВКР

## Основной формат: C4 + draw.io

| Файл | Назначение |
|------|------------|
| `c4_dct_architecture.drawio` | Исходник архитектуры (3 вкладки: Context, Container, Component) |
| `rc_chain.drawio` | Цепочка RC: EdgeTX → ELRS → приёмник → ESP32 → COM → DCT (п. **2.3**) |
| `video_chain.drawio` | Видеотракт: камера → HDZero VTX → EventVRX → захват → DCT (п. **2.4**) |
| `pf_cycle.drawio` | Цикл PF: predict → weight update → ESS/resampling → circular mean (п. **3.4**) |
| `pf_cycle.md` | Пояснение к блок-схеме PF: все блоки и математика простым языком (п. **3.4**) |
| `camera_observation.drawio` | Pipeline CameraObservation: YOLO → gate_id → PnP → xyz_obs (п. **3.5**) |
| `camera_observation.md` | Пояснение: как по изображению получают положение (п. **3.5**) |
| `integration_monitoring.drawio` | Интеграция DCT → MAVLink → Connect → «Небосвод» (п. **4.1**) |
| `geo_anchors.drawio` | Геопривязка origin / x / z (п. **4.2**) |
| `geo_transform.drawio` | Алгоритм ENU → WGS84 (п. **4.3**) |
| `C4.md` | Правила C4, экспорт в Word, соответствие главам ВКР |

**Просмотр и правки:** [draw.io](https://app.diagrams.net/) или VS Code (Draw.io Integration).

**Для п. 2.1:** вкладка **«2. C4 Container»** → Export PNG/PDF → вставить в Word.

**Для п. 2.3:** `rc_chain.drawio` → Export PNG (300 DPI, ширина 14–16 см). Подпись: *«Рисунок 2.X — Цепочка получения RC-каналов на наземной станции»*.

**Для п. 2.4:** `video_chain.drawio` → Export PNG. Подпись: *«Рисунок 2.X — Цепочка захвата видеопотока (HDZero, устройство захвата, DCT)»*.

**Для п. 3.4:** `pf_cycle.drawio` → Export PNG. Подпись: *«Рисунок 3.X — Блок-схема одного шага Particle Filter (predict, обновление весов, ресемплинг, circular mean)»*. Текст к рисунку и формулы — [`pf_cycle.md`](pf_cycle.md).

**Для п. 3.5:** `camera_observation.drawio` → Export PNG. Подпись: *«Рисунок 3.X — Формирование камерного наблюдения (YOLO, привязка ворот, PnP)»*. Пояснение — [`camera_observation.md`](camera_observation.md).

**Для гл. 4:** `integration_monitoring.drawio` (§ 4.1), `geo_anchors.drawio` (§ 4.2), `geo_transform.drawio` (§ 4.3). Экспорт PNG, ширина 14–16 см.

**Подпись (рекомендуемая):**

> Рисунок 2.1 — Диаграмма контейнеров C4 комплекса отслеживания БВС

В draw.io включить библиотеку фигур: **More Shapes → C4**.

---

## Устаревший / вспомогательный

| Файл | Примечание |
|------|------------|
| `fig_2_1_architecture.svg` | Блок-схема цепочек RC/видео; заменена C4-диаграммой, можно не использовать |
