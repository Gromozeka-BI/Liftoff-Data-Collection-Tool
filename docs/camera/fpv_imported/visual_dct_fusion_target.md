# Целевая интеграция FPVCamDetectV2 и DCT OnlineLocalizer

## 1. Цель

Нужно объединить два источника локализации:

1. `FPVCamDetectV2` определяет абсолютную позу камеры/дрона по изображению и
   воротам.
2. `DCT` ведет онлайн-оценку положения дрона по стикам/телеметрии вдоль
   эталонной траектории.

Итоговая система должна использовать визуальное `XYZ + sigma_cam` как внешнее
наблюдение для `DCT OnlineLocalizer`, а оценку `DCT` использовать как грубый
prior для выбора `gate_id` в `FPVCamDetectV2`.

## 2. Роли компонентов

### 2.1. FPVCamDetectV2

`FPVCamDetectV2` отвечает за визуальное наблюдение:

```text
image
  -> gate corner detections
  -> gate_id localization
  -> PnPSolver
  -> CameraObservation(xyz, sigma_cam, confidence, metadata)
```

Ключевые модули:

- `gate_localization` - присваивает детекциям `gate_id`;
- `pnp_solver` - решает позу камеры/дрона по известным `gate_id` и 2D-углам;
- `gate_model` - хранит 3D-карту ворот.

`FPVCamDetectV2` не должен сам сглаживать траекторию поверх RC/телеметрии. Его
основная задача - выдавать редкие, но качественные абсолютные наблюдения.

### 2.2. DCT

`DCT OnlineLocalizer` отвечает за непрерывную онлайн-оценку:

```text
sticks / telemetry
  -> OnlineLocalizer.update(sticks, dt)
  -> optional inject_position_observation(xyz_obs, sigma_cam)
  -> optional KFLayer2.update(result, dt)
  -> final XYZ + uncertainty
```

`OnlineLocalizer` работает не в произвольном 3D-пространстве, а в параметре `s`
вдоль эталонной трассы. После оценки `s` он восстанавливает `XYZ` через
`Reference.pos`.

## 3. Целевая схема обмена

```text
                         +----------------------+
                         | DCT OnlineLocalizer  |
                         | sticks -> s -> XYZ   |
                         +----------+-----------+
                                    |
                                    | coarse pose / uncertainty
                                    v
+----------+      +-----------------+------------------+
|  image   | ---> | FPVCamDetectV2 gate localization   |
+----------+      | coarse prior + refine / tracking   |
                  +-----------------+------------------+
                                    |
                                    | gate_id + 2D corners
                                    v
                         +----------+-----------+
                         |      PnPSolver       |
                         |   visual XYZ + q     |
                         +----------+-----------+
                                    |
                                    | CameraObservation
                                    v
                         +----------+-----------+
                         | inject_position_     |
                         | observation(...)     |
                         +----------------------+
```

Связь двусторонняя:

- `DCT -> FPVCamDetectV2`: текущая оценка позиции помогает сузить список
  возможных ворот.
- `FPVCamDetectV2 -> DCT`: успешный PnP уточняет частицы `OnlineLocalizer` через
  `inject_position_observation`.

## 4. Контракт визуального наблюдения

Рекомендуемый формат результата `FPVCamDetectV2`:

```python
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class CameraObservation:
    xyz: np.ndarray           # shape (3,), meters, DCT Reference.pos coordinates
    sigma_cam: float          # meters, standard deviation of visual position error
    confidence: float         # 0..1
    timestamp: float          # seconds
    gate_ids: list[int]
    reprojection_rmse_px: float
    source_mode: str          # tracking | coarse_refine | global
    debug_info: dict[str, Any] | None = None
```

Минимальные поля для DCT:

- `xyz` - позиция дрона/камеры в координатах `DCT Reference.pos`;
- `sigma_cam` - реалистичная стандартная ошибка в метрах;
- `timestamp` - время изображения.

Остальные поля нужны для диагностики, фильтрации плохих наблюдений и настройки
порогов.

## 5. Координатные системы

Главное требование: `CameraObservation.xyz` должен быть в той же системе
координат, что и `DCT Reference.pos`.

Если `FPVCamDetectV2` использует координаты из `config/track.json`, а DCT
использует координаты из своего reference-файла, нужно явно проверить:

- совпадают ли оси `X/Y/Z`;
- совпадает ли начало координат;
- совпадает ли масштаб в метрах;
- одинаково ли задана высота;
- не требуется ли поворот/сдвиг между картами.

Если системы не совпадают, перед отправкой в DCT нужен transform:

```text
xyz_dct = T_fpvcam_to_dct(xyz_fpvcam)
```

Без этого `OnlineLocalizer` будет считать корректное визуальное наблюдение
ошибочным и может ухудшить локализацию.

## 6. Оценка `sigma_cam`

`sigma_cam` нельзя занижать. Ошибочное наблюдение с маленьким `sigma_cam` опаснее,
чем отсутствие наблюдения.

Рекомендуемая политика:

- хорошие multi-gate решения с устойчивым consensus: `sigma_cam` около `1..3 м`;
- single-gate решения или слабый consensus: `sigma_cam` около `3..8 м`;
- неоднозначный `gate_id`, большой reprojection RMSE, слабый margin: не отдавать
  наблюдение или ставить `sigma_cam > 10 м`;
- наблюдения с `sigma_cam > 10 м` не подмешивать в `OnlineLocalizer`.

`reprojection_rmse_px` сам по себе недостаточен для маленького `sigma_cam`:
планарный PnP может иметь хороший RMSE при зеркальной или неверной позе. Нужно
учитывать:

- число видимых ворот;
- single-gate или multi-gate режим;
- consensus между независимыми PnP-кандидатами;
- margin между лучшей и второй гипотезой `gate_id`;
- согласованность с coarse prior от DCT;
- скачок относительно предыдущей визуальной позы.

## 7. Использование DCT prior в FPVCamDetectV2

Оценку DCT нужно подавать в `gate_localization` как грубую позу. Для самой
камерной локализации предпочтительно использовать последнюю финальную оценку DCT,
то есть результат после `KFLayer2`, если он уже инициализирован и достаточно
уверен:

```text
coarse_pose = {
    xyz: last_final_result.position_xyz,
    uncertainty_m: last_final_result.uncertainty_m,
    timestamp: frame_ts,
}
```

Это prior из прошлого уже известного состояния. На кадре `t` камера не должна
получать оценку, которая еще будет уточнена этой же картинкой. Поэтому базовый
порядок такой:

```text
last_final_result from frame t-1
  -> coarse prior for FPVCamDetectV2 on frame t
  -> camera XYZ for frame t
  -> OnlineLocalizer.inject_position_observation(...)
  -> KFLayer2.update(...)
  -> final_result for next frames
```

Если `KFLayer2` еще не инициализирован, его `uncertainty_m` слишком большой или
он дает заметную задержку, fallback-ом может быть обычный результат
`OnlineLocalizer` до второго слоя. Если и он слишком неопределенный, prior лучше
не передавать и перейти к `Global search`.

Дальше `FPVCamDetectV2` выбирает режим:

1. Если есть здоровый визуальный track, использовать `Tracking-first`.
2. Если track слабый, но есть DCT prior, использовать `Coarse prior + refine`.
3. Если prior нет или он слишком неопределенный, использовать `Global search`.

Практически это особенно важно для холодного старта и восстановления после
потери трека: чистый `Global search` может быть неоднозначен, потому что разные
ворота иногда дают похожий reprojection RMSE.

## 8. Рекомендуемый runtime-цикл

```python
loc = OnlineLocalizer.from_file("reference.npz")
loc.reset()

kf = KFLayer2(loc.ref)
kf.reset()

last_final_result = None

for frame in stream:
    result = loc.update(
        sticks=frame.sticks,
        dt=frame.dt,
        rate_profile=frame.rate_profile,
    )

    coarse_pose = None
    if last_final_result is not None:
        coarse_pose = {
            "xyz": last_final_result.position_xyz,
            "uncertainty_m": last_final_result.uncertainty_m,
            "timestamp": frame.ts,
        }

    obs = fpvcam_localizer.try_get_observation(
        image=frame.image,
        coarse_pose=coarse_pose,
    )

    if obs is not None and obs.sigma_cam <= 10.0:
        result = loc.inject_position_observation(
            xyz_obs=obs.xyz,
            sigma_cam=obs.sigma_cam,
        )

    result = kf.update(result, frame.dt)

    output_xyz = result.position_xyz
    output_uncertainty_m = result.uncertainty_m
    last_final_result = result
```

`KFLayer2` остается опциональным. Если визуальная локализация точная и частая, он
может добавить лишнюю инерционность. Если визуальные наблюдения редкие и шумные,
он полезен для сглаживания.

## 9. Критерии готовности

Интеграцию можно считать рабочей, когда выполнены условия:

- есть явный и проверенный transform между координатами `FPVCamDetectV2` и
  `DCT Reference.pos`;
- `FPVCamDetectV2` умеет возвращать `CameraObservation` с реалистичным
  `sigma_cam`;
- плохие и неоднозначные PnP-решения не отправляются в `OnlineLocalizer`;
- `DCT` prior используется для `Coarse prior + refine`;
- на последовательности кадров видно, что визуальные наблюдения уменьшают
  неопределенность `OnlineLocalizer`, а не создают скачки;
- логируются `source_mode`, `gate_ids`, `reprojection_rmse_px`, `sigma_cam` и
  факт инжекта/отказа.

## 10. Следующие шаги реализации

1. Сравнить координатные системы `config/track.json` и DCT `Reference.pos`.
2. Описать или реализовать `T_fpvcam_to_dct`.
3. Добавить тип результата `CameraObservation` на стороне `FPVCamDetectV2`.
4. Связать `Coarse prior + refine` с prior-ом из DCT.
5. Настроить первичную эвристику `sigma_cam` по RMSE, consensus и mode.
6. Сделать интеграционный прогон на короткой последовательности кадров.
