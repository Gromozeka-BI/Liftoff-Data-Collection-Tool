# Эксперимент 11: импорт камерного модуля

Эта папка содержит материалы по поэтапному импорту FPV-модуля камерной
локализации в DCT.

Основной план:

- `docs/camera_module_import_plan.md`

Начальный объём работ:

1. Импортировать геометрию и PnP-код из `FPVCamDetectV2`.
2. Импортировать адаптер YOLO-разметки из `FPVCamDetectV2`.
3. Генерировать offline-файлы `CameraObservation`.
4. Проверить защищённое слияние через `OnlineLocalizer.inject_position_observation(...)`.

## Дымовой тест

Запуск импортированного адаптера и цепочки локализации на размеченном
калибровочном кадре:

```bash
python tools/exp11_camera_module_import/smoke_camera_import.py
```

Пример с несколькими воротами:

```bash
python tools/exp11_camera_module_import/smoke_camera_import.py --section multi_frames --frame-idx 0 --q-m 10 --coarse-offset-x 0
```

Скрипт преобразует импортированные калибровочные keypoints во временную
YOLO-pose-разметку, загружает её через `yolo_gate_adapter` и запускает
`CoarseRefineLocalizer`.

## Дополнительно импортировано

Также импортированы:

- `dct/camera_localization/calibration_tools/` для воспроизведения
  `camera_calibration.json`;
- `docs/camera/fpv_imported/` с исходными проектными документами из
  `FPVCamDetectV2`;
- `dct/camera_localization/experimental/pnp_solver_2/` как сравнительный
  прототип AP3P/LM.

`pnp_solver_2` не входит в основной путь выполнения, потому что текущая проверка
показывает большие выбросы на сценах с несколькими воротами.

