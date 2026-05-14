# Модуль: pnp_solver_2

Экспериментальный PnP-solver, сделанный как отдельная ветка идей из проекта
`gate-detection-and-localization-in-drone-racing-using-convolutional-neural-networks`.

## Что перенесено из референса

- `solvePnPGeneric(..., SOLVEPNP_AP3P)` для получения нескольких кандидатов по 4 углам.
- Физическая фильтрация кандидатов по стороне плоскости ворот.
- Выбор seed-ворот по максимальной площади четырёхугольника на изображении.
- Оценка seed-гипотезы по суммарной репроекции всех видимых ворот.
- Финальное уточнение одной общей позы через `SOLVEPNP_ITERATIVE` + `solvePnPRefineLM`.

## Что адаптировано под наш проект

- Порядок углов остаётся нашим: `TL -> TR -> BR -> BL`.
- Используется `src/gate_model/gate_model.py` и `config/track.json`.
- Система координат остаётся нашей: `x,z` горизонтальные, `y` вверх, yaw вокруг `Y`.
- Входной формат остаётся совместимым с текущим solver-ом:

```python
gate_detections = [
    (gate_id, keypoints_2d),  # keypoints_2d shape (4, 2), TL -> TR -> BR -> BL
]
```

## Статус

Это прототип для сравнения с `dct.camera_localization.pnp_solver`, а не замена
текущего solver-а. Главная цель - проверить, даст ли AP3P-disambiguation +
seed/global refine более стабильную позу на наших multi-gate кадрах.

В DCT модуль лежит в `dct.camera_localization.experimental`, поэтому он не
используется default runtime-пайплайном.
