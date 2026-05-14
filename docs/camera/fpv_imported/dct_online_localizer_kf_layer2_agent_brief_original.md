# Brief for Agent: OnlineLocalizer, Camera XYZ Fusion, KFLayer2

Документ предназначен для другого агента/проекта, который должен интегрировать
внешнюю визуальную локализацию дрона с текущей системой DCT.

## 1. Общая задача

В DCT уже есть онлайн-локализатор дрона по RC-стикам/телеметрии:

```python
from dct.localization.online_localizer import OnlineLocalizer
```

Он принимает на вход стики управления и оценивает положение дрона на трассе.
Отдельная внешняя система по изображению с камеры может определять абсолютные
координаты дрона:

```text
XYZ = [x, y, z]
q   = точность/неопределенность в метрах
```

Эти данные нужно передавать в `OnlineLocalizer`, чтобы уточнять текущую позицию.
Для этого в `OnlineLocalizer` уже реализован метод:

```python
result = loc.inject_position_observation(
    xyz_obs=[x, y, z],
    sigma_cam=q,
)
```

Основная архитектура:

```text
RC sticks / telemetry
        |
        v
OnlineLocalizer.update(sticks, dt)
        |
        |  when camera observation is available
        v
OnlineLocalizer.inject_position_observation(XYZ, sigma_cam)
        |
        v
optional KFLayer2 smoothing
        |
        v
final XYZ + uncertainty
```

## 2. Важное ограничение по системе координат

`xyz_obs` от камеры должен быть в той же системе координат, что и эталон трассы
`Reference.pos`.

Если внешний проект получает позицию:

- в координатах камеры;
- в координатах корпуса дрона;
- в координатах другой карты;
- в координатах gate-localizer;

то перед вызовом `inject_position_observation()` нужно преобразовать ее в систему
координат трассы DCT.

Иначе фильтр будет считать корректное наблюдение ошибочным и может ухудшить
локализацию.

## 3. `online_localizer.py`

Файл:

```text
dct/localization/online_localizer.py
```

Назначение:

`OnlineLocalizer` оценивает положение дрона на эталонной траектории в реальном
времени. Основной источник информации - RC-стики или физические setpoint-ы,
полученные из стиков через Betaflight rate profile.

Внутри локализатор не ищет позицию в произвольном 3D-пространстве. Он работает
в 1D-параметризации трассы:

```text
s = расстояние вдоль эталонного круга, м
```

После оценки `s` координаты восстанавливаются по эталону:

```python
xyz = ref.pos_at_s(s)
```

### 3.1. `LocalizerResult`

Результат одного шага:

```python
@dataclass
class LocalizerResult:
    position_xyz: np.ndarray
    s: float
    progress: float
    uncertainty_m: float
    track_length: float
```

Поля:

- `position_xyz` - оценка позиции `[x, y, z]` в метрах;
- `s` - позиция вдоль трассы в метрах;
- `progress` - доля круга `0.0 .. 1.0`;
- `uncertainty_m` - неопределенность оценки в метрах;
- `track_length` - длина эталонного круга.

### 3.2. `Reference`

`Reference` - эталонный круг. Он хранит:

- `s` - массив расстояний вдоль траектории;
- `pos` - массив координат `[x, y, z]`;
- `sticks_norm` - нормализованные признаки управления;
- `mean/std` - параметры нормализации;
- `smooth_w` - окно сглаживания;
- `L` - длину круга;
- `feature_kind` и `rate_profile` - режим признаков, если используется
  Betaflight feature mode.

Эталон загружается из `.npz`:

```python
loc = OnlineLocalizer.from_file("reference.npz")
```

### 3.3. Particle Filter

Внутри `OnlineLocalizer` используется `ParticleFilter`.

Состояние каждой частицы:

```text
s - позиция вдоль трассы, м
v - скорость вдоль трассы, м/с
```

На каждом `update(sticks, dt)`:

1. Если фильтр не инициализирован, создаются частицы около старта/финиша круга.
2. Выполняется prediction:

   ```text
   s = s + v * dt + noise
   v = v + noise
   ```

3. Текущие стики преобразуются в тот же формат, что и признаки эталона.
4. Для каждой частицы берется ближайший эталонный вектор признаков.
5. Считается отличие текущих признаков от эталонных.
6. Чем меньше отличие, тем больше вес частицы.
7. При вырождении весов выполняется resampling.
8. Итоговый `s` считается как circular mean, потому что трасса циклическая.
9. `s` переводится в `XYZ` через `Reference.pos_at_s(s)`.

### 3.4. Обычный вызов `update`

Пример:

```python
loc = OnlineLocalizer.from_file("reference.npz")
loc.reset()

prev_ts = None

for frame in telemetry_stream:
    ts = frame["ts"]
    dt = (ts - prev_ts) if prev_ts is not None else None
    prev_ts = ts

    sticks = [
        throttle,
        yaw,
        pitch,
        roll,
    ]

    result = loc.update(
        sticks=sticks,
        dt=dt,
        rate_profile=current_rate_profile,
    )

    print(result.position_xyz, result.uncertainty_m)
```

Порядок каналов:

```text
[throttle, yaw, pitch, roll]
```

Стики ожидаются нормализованными в диапазоне примерно `-1..1`.

Если эталон построен в Betaflight feature mode, то `update()` преобразует стики
в физические setpoint-ы через rate profile. Это нужно, чтобы один и тот же
маневр сравнивался одинаково при разных настройках рейтов.

## 4. Интеграция внешнего `XYZ + q`

Главный метод для внешней визуальной системы:

```python
result = loc.inject_position_observation(
    xyz_obs=[x, y, z],
    sigma_cam=q,
)
```

`xyz_obs`:

- форма `(3,)`;
- единицы - метры;
- система координат - та же, что у `Reference.pos`.

`sigma_cam`:

- положительное число;
- метры;
- интерпретируется как стандартное отклонение изотропного 3D-шума камеры.

Важно: если внешний модуль сообщает `+-q`, нужно понять, что именно означает
`q`.

Если `q` уже является стандартным отклонением, передавать:

```python
sigma_cam = q
```

Если `+-q` означает примерно 95% доверительный интервал, лучше использовать:

```python
sigma_cam = q / 2
```

Если смысл `q` неизвестен, безопаснее не занижать точность. Заниженный
`sigma_cam` опасен: фильтр будет слишком сильно доверять ошибочному визуальному
наблюдению.

### 4.1. Что происходит внутри `inject_position_observation`

Для каждой частицы:

1. Берется ее текущий `s_i`.
2. Через эталон находится 3D-позиция частицы:

   ```python
   xyz_part = ref.pos[idx]
   ```

3. Считается расстояние до внешнего наблюдения:

   ```text
   d2 = ||xyz_part - xyz_obs||^2
   ```

4. Вес частицы обновляется по Gaussian likelihood:

   ```text
   weight *= exp(-0.5 * d2 / sigma_cam^2)
   ```

5. Выполняется нормализация весов.
6. При необходимости выполняется resampling.
7. Возвращается новый `LocalizerResult`.

Это честный 3D Bayes-update: камера не переводится заранее в `s`, а сравнивается
с каждой частицей в 3D.

### 4.2. Рекомендуемый цикл интеграции

```python
from dct.localization.online_localizer import OnlineLocalizer

loc = OnlineLocalizer.from_file("reference.npz")
loc.reset()

prev_ts = None

for frame in runtime_stream:
    ts = frame.ts
    dt = (ts - prev_ts) if prev_ts is not None else None
    prev_ts = ts

    # 1. Основное обновление по RC/телеметрии.
    result = loc.update(
        sticks=frame.sticks,  # [throttle, yaw, pitch, roll]
        dt=dt,
        rate_profile=frame.rate_profile,
    )

    # 2. Внешняя визуальная локализация, если есть новое наблюдение.
    obs = camera_localizer.try_get_observation(frame.image)

    if obs is not None:
        # obs.xyz must already be in DCT track coordinates.
        if obs.sigma_cam <= 10.0:
            result = loc.inject_position_observation(
                xyz_obs=obs.xyz,
                sigma_cam=obs.sigma_cam,
            )

    # 3. Использовать result как текущую оценку.
    current_xyz = result.position_xyz
    current_q = result.uncertainty_m
```

Вызовы `update()` и `inject_position_observation()` можно делать асинхронно.
Например:

- `update()` приходит на каждом кадре RC, условно 100 Гц;
- `inject_position_observation()` вызывается только когда камера дала уверенное
  наблюдение, например 2-5 Гц.

## 5. Практические требования к визуальной системе

Из эксперимента `Exp 10` по камерной фьюжн:

- `sigma_cam <= 5 м` и частота хотя бы `1 Гц` уже могут быть полезны;
- целевой уровень: `sigma_cam <= 3 м` и частота `>= 2 Гц`;
- хороший уровень: `sigma_cam <= 1 м` и частота `>= 5 Гц`;
- наблюдения с `sigma_cam > 10 м` лучше не подмешивать;
- плохая камера с большой ошибкой может ухудшить результат относительно чистого
  RC-локализатора.

Особенно важно не занижать `sigma_cam`. Если визуальный модуль не уверен,
лучше:

```python
do_not_inject = True
```

чем передать плохое наблюдение с маленькой заявленной ошибкой.

## 6. `kf_layer2.py`

Файл:

```text
dct/localization/kf_layer2.py
```

Назначение:

`KFLayer2` - второй слой фильтрации поверх результата `OnlineLocalizer`.
Он не принимает сырые стики и не принимает напрямую картинку/камеру.
Он принимает уже готовый `LocalizerResult`.

```python
from dct.localization.kf_layer2 import KFLayer2

kf = KFLayer2(ref)
kf.reset()

result_kf = kf.update(result, dt)
```

### 6.1. Состояние KF

Состояние:

```text
x = [s, v]

s - позиция вдоль трассы, м
v - скорость вдоль трассы, м/с
```

Prediction:

```text
s' = s + v * dt
v' = v
```

Update от первого слоя:

```text
z = result.s
R = result.uncertainty_m^2
```

То есть `KFLayer2` воспринимает результат Particle Filter как измерение позиции
вдоль трассы.

### 6.2. Псевдо-измерение скорости

`KFLayer2` строит профиль эталонной скорости `v_ref(s)` из `Reference.s`.

Затем после обычного update добавляет мягкий аттрактор:

```text
v -> v_ref(s)
```

Это сглаживает скачки и заставляет оценку двигаться с похожей скоростью, как
эталонный круг.

Параметры:

- `sigma_v` - шум процесса по скорости;
- `sigma_v_pseudo` - жесткость притяжения к `v_ref(s)`;
- `init_q_thresh` - фильтр инициализируется только когда первый слой достаточно
  уверен.

По умолчанию:

```python
kf = KFLayer2(
    ref,
    sigma_v=2.0,
    sigma_v_pseudo=2.0,
    init_q_thresh=10.0,
)
```

### 6.3. Когда использовать KFLayer2

`KFLayer2` полезен, если нужно сгладить результат RC Particle Filter.

Рекомендуемый порядок:

```text
OnlineLocalizer.update()
OnlineLocalizer.inject_position_observation()  # если есть камера
KFLayer2.update()
```

То есть камера должна уточнять именно `OnlineLocalizer`, а `KFLayer2` может
сглаживать уже объединенный результат.

Если внешняя камера очень точная и частая, `KFLayer2` может быть необязателен.
Он добавляет инерционность и привязку к эталонному скоростному профилю, что может
немного увеличивать задержку.

## 7. Рекомендуемый итоговый API для соседнего проекта

Внешний проект должен отдавать наблюдение примерно в таком виде:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class CameraObservation:
    xyz: np.ndarray       # shape (3,), meters, DCT track coordinates
    sigma_cam: float      # meters, standard deviation
    confidence: float     # optional, 0..1
    timestamp: float      # seconds
```

Минимальный контракт:

```text
xyz        - координаты дрона в DCT track coordinates
sigma_cam  - оценка стандартного отклонения ошибки в метрах
timestamp  - время наблюдения
```

Опционально:

- `confidence`;
- `gate_id`;
- `reprojection_error`;
- `debug_info`;
- `latency_ms`.

Если агент в соседнем проекте строит визуальный локализатор, его главная задача:

1. По изображению определить положение дрона в координатах трассы.
2. Оценить реалистичный `sigma_cam`.
3. Не отдавать наблюдение, если оно плохое или неоднозначное.
4. Передавать результат в DCT через `inject_position_observation()`.

## 8. Полный пример пайплайна

```python
from dct.localization.online_localizer import OnlineLocalizer
from dct.localization.kf_layer2 import KFLayer2

loc = OnlineLocalizer.from_file("reference.npz")
loc.reset()

kf = KFLayer2(loc.ref)
kf.reset()

prev_ts = None

for frame in stream:
    ts = frame.ts
    dt = (ts - prev_ts) if prev_ts is not None else None
    prev_ts = ts

    result = loc.update(
        sticks=frame.sticks,
        dt=dt,
        rate_profile=frame.rate_profile,
    )

    obs = camera_localizer.try_get_observation(frame.image)

    if obs is not None and obs.sigma_cam <= 10.0:
        result = loc.inject_position_observation(
            xyz_obs=obs.xyz,
            sigma_cam=obs.sigma_cam,
        )

    result = kf.update(result, dt)

    output_xyz = result.position_xyz
    output_uncertainty_m = result.uncertainty_m
```

## 9. Короткое резюме для агента

`OnlineLocalizer` - основной online-фильтр. Он работает как Particle Filter по
параметру `s` вдоль эталонной трассы и возвращает `XYZ`.

`inject_position_observation(xyz_obs, sigma_cam)` - правильная точка интеграции
внешней визуальной локализации. Передавать туда нужно `XYZ` в координатах трассы
и реалистичную ошибку в метрах.

`KFLayer2` - опциональный сглаживающий слой поверх результата `OnlineLocalizer`.
Он работает с `LocalizerResult`, а не с камерой напрямую.

Итоговая рекомендуемая цепочка:

```text
sticks -> OnlineLocalizer.update()
camera XYZ + q -> OnlineLocalizer.inject_position_observation()
optional -> KFLayer2.update()
final -> XYZ + uncertainty_m
```
