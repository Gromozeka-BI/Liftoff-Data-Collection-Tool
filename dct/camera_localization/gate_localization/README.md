# Модуль: gate_localization

## Назначение

`gate_localization` решает задачу присвоения `gate_id` детекциям ворот.
Детектор отдаёт только 4 угла ворот в пикселях, а этот модуль подбирает,
каким воротам из `config/track.json` они соответствуют, и затем вызывает
существующий `PnPSolver`.

Текущая реализация находится в `global_search.py`.

Для real-time сценария добавлен отдельный модуль `topk_hypotheses.py`. Он
возвращает не один assignment, а top-K ID-гипотез в рамках временного бюджета.

## Текущий Real-Time Pipeline

Главная идея: мы не выбираем `gate_id` по `RMSE`. Мы ищем такие ID ворот, при
которых все видимые ворота дают одну и ту же позицию камеры в XYZ. Потом
проверяем эту позицию относительно примерного положения дрона.

```text
[1. 2D-детекции ворот]
        |
        v
[2. Coarse position + q_m]
        |
        v
[3. Candidate gates вокруг coarse]
        |
        v
[4. Group-first shortlist для 4+ ворот]
        |
        v
[5. Single-gate IPPE candidates]
        |
        v
[6. Pairwise XYZ compatibility]
        |
        v
[7. Beam search по gate_id]
        |
        v
[8. XYZ-consensus scoring]
        |
        v
[9. Top-K гипотез ID]
        |
        v
[10. Fast PnPSolver.solve]
        |
        v
[11. Выбор гипотезы ближе к coarse]
        |
        v
[12. Итоговая позиция дрона]
```

### Блоки Алгоритма

1. `2D-детекции ворот`

   На вход приходят найденные ворота на изображении. Для каждых ворот есть 4
   точки `TL, TR, BR, BL`. На этом этапе мы знаем только координаты углов в
   пикселях, но ещё не знаем точный `gate_id`.

2. `Coarse position + q_m`

   Используем примерную позицию дрона `coarse_position_world = [x, y, z]` и
   неопределённость `q_m`. Простыми словами: дрон примерно здесь, но позиция
   может ошибаться на `q_m` метров. `q_m` влияет только на ширину поиска, а не
   на финальный score гипотезы.

3. `Candidate gates вокруг coarse`

   По карте ворот выбираем кандидатов рядом с coarse position. Ворота делятся
   на зоны `near`, `mid`, `far`, `extra`. Для каждой 2D-детекции получается
   список возможных `gate_id`.

4. `Group-first shortlist для 4+ ворот`

   Если на кадре найдено 4 или 5 ворот, сначала проверяем локальную группу
   ближайших ворот. Например, для 5 детекций рядом с coarse может получиться
   группа `[9, 10, 11, 12, 13]`. Если узкий поиск слабый, запускается fallback
   на широкий список кандидатов.

5. `Single-gate IPPE candidates`

   Для каждой пары `(detection, gate_id)` решается маленькая PnP-задача по
   одному гейту. Один гейт плоский, поэтому он может дать несколько возможных
   позиций камеры. Это ещё не финальный ответ, а набор предположений.

6. `Pairwise XYZ compatibility`

   Проверяем пары assignment-кандидатов. Если `det 0 -> gate 9` говорит, что
   камера находится в одной точке, а `det 1 -> gate 10` говорит, что камера
   находится совсем в другой точке, такую пару не соединяем в одну гипотезу.

7. `Beam search по gate_id`

   Строим ID-гипотезы постепенно: `det 0 -> gate A`, `det 1 -> gate B` и так
   далее. Beam search держит только несколько лучших веток и не перебирает все
   варианты полностью.

8. `XYZ-consensus scoring`

   Главный критерий: насколько все выбранные ворота согласны между собой по
   позиции камеры. Если все дают почти одну и ту же точку, гипотеза хорошая.
   Если позиции сильно расходятся, гипотеза плохая.

   ```text
   score = spread_m * spread_weight
   ```

   `RMSE` не участвует в выборе ID, потому что на симметричных воротах он часто
   одинаково хорош для правильных и неправильных вариантов.

9. `Top-K гипотез ID`

   На выходе beam search получаем несколько лучших вариантов ID. Например:
   `rank 1: [9, 10, 11, 12, 13]`, `rank 2: [...]`. Это список наиболее
   вероятных соответствий между 2D-детекциями и реальными воротами.

10. `Fast PnPSolver.solve`

    Для лучших гипотез считается итоговая позиция камеры. Для 4+ ворот
    используется быстрый режим `refine=False`: берём consensus pose без дорогого
    финального `SOLVEPNP_ITERATIVE`.

11. `Выбор гипотезы ближе к coarse`

    Среди top-K выбирается гипотеза, чья pose ближе к `coarse_position_world`:

    ```text
    final_score = consensus_score + distance_to_coarse * weight
    ```

    То есть победитель должен быть геометрически согласованным и не слишком
    далеко от примерной позиции дрона.

12. `Итоговая позиция дрона`

    На выходе получаем `position_world`, `yaw_deg`, `quaternion`, выбранные
    `gate_id`, `runtime_ms`, `timed_out` и top-K гипотезы. Простыми словами:
    система говорит, какие реальные ворота видны на кадре и где относительно
    карты находится дрон.

## Входной Формат

```python
from gate_localization import GateDetection, GlobalSearchLocalizer

detections = [
    GateDetection(keypoints),  # keypoints shape (4, 2), TL -> TR -> BR -> BL
]

result = GlobalSearchLocalizer().assign_and_solve(detections)
```

Можно также передавать `np.ndarray` напрямую вместо `GateDetection`.

Для ограничения поиска локальным участком трассы:

```python
result = GlobalSearchLocalizer().assign_and_solve(
    detections,
    candidate_gate_ids=[9, 10, 11, 12, 13],
)
```

## Выходной Формат

`assign_and_solve(...)` возвращает `GlobalSearchResult`:

- `success`: можно ли считать назначение уверенным;
- `assignments`: список `detection_index -> gate_id`;
- `pose`: результат `PnPSolver`;
- `score`: score лучшей гипотезы;
- `second_best_score`: score второй гипотезы;
- `n_hypotheses`: сколько ID-гипотез было проверено;
- `reason`: диагностическая строка.

Важно: при `success=False` поле `assignments` всё равно может содержать лучший
вариант, если он найден, но признан неоднозначным по confidence.

## Алгоритм Global Search

1. Нормализует входные детекции до массива `(4, 2)`.
2. Берёт все `gate_id` из `GateMap` или только `candidate_gate_ids`.
3. Перебирает упорядоченные уникальные комбинации ID.
4. Для каждой пары `(detection, gate_id)` строит single-gate IPPE-кандидаты:
   `IDENT/HFLIP x c0/c1`.
5. Для multi-gate гипотезы выбирает комбинацию кандидатов с минимальным
   разбросом XYZ-позиций камеры.
6. Считает score:

```text
score = spread_m * 100 + mean_candidate_rmse_px + physical_penalty
```

где `spread_m` — главный критерий, а RMSE только разбивает близкие варианты.

7. Лучший ID-набор дополнительно прогоняется через существующий `PnPSolver`.
8. Confidence считается по margin между первой и второй гипотезой.

## Почему Не Только RMSE

Одинаковые ворота можно хорошо перепроецировать из разных поз камеры.
На multi-gate кадрах RMSE-only часто выбирает неверный `gate_id`.

Более устойчивый признак: если несколько детекций действительно относятся к
одному кадру, то single-gate PnP-кандидаты должны давать близкие XYZ-позиции
камеры. Поэтому текущий `global_search` ранжирует multi-gate гипотезы по
`spread_m`.

## Ограничения

Полный перебор по всей карте быстро растёт:

```text
M=2: 13P2 = 156 гипотез
M=3: 13P3 = 1716 гипотез
M=5: 13P5 = 154440 гипотез
```

Поэтому для 4-5 детекций нужен shortlist через будущие режимы:

- `Coarse prior + refine`;
- `Tracking-first`;
- порядок прохождения трассы;
- локальная группа ворот.

Параметры защиты:

- `max_detections_for_full_search`;
- `max_hypotheses`;
- `candidate_gate_ids`.

## Проверка

Полная валидация с лимитами по умолчанию:

```bash
python src/gate_localization/global_search.py --validate
```

Только multi-gate кадры из Python:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from gate_localization.global_search import validate_on_calibration_json; validate_on_calibration_json(sections=('multi_frames',))"
```

## Текущие Выводы По Multi-Gate

Consensus scoring заметно лучше RMSE-only:

- `frame_000158.jpg`: правильная `[3, 2]` поднялась со 149 места на 3 место.
- `frame_000232.jpg`: правильная `[7, 4]` оказалась в top-10.
- `frame_000359.jpg` с shortlist `[9,10,11,12,13]`: правильная гипотеза top-1.
- `frame_000370.jpg`: правильная `[11,10,9]` top-2 по полной карте.
- `frame_000494.jpg` с shortlist `[9,10,11,13]`: правильная гипотеза top-1.
- `frame_000646.jpg`: правильная `[21,17,19]` top-2 по полной карте.

Главный вывод: `global_search` хорошо находит геометрически согласованные
ID-гипотезы, но в симметричных блоках часто остаются несколько почти равных
вариантов. Рабочая система должна передавать дальше top-K или выбирать среди
них через prior/track/coarse pose.

## Real-Time Top-K

`topk_hypotheses.py` предназначен для следующего слоя системы: не потерять
правильную ID-гипотезу, даже если она не top-1.

```python
from gate_localization import GateDetection, TopKHypothesisGenerator

generator = TopKHypothesisGenerator(
    top_k=10,
    time_budget_ms=200.0,
    search_mode="auto",
)

result = generator.generate(
    detections,
    candidate_gate_ids=[9, 10, 11, 12, 13],  # optional shortlist
)

for hyp in result.hypotheses:
    print(hyp.rank, hyp.gate_ids, hyp.spread_m, hyp.score)
```

Основные параметры:

- `time_budget_ms`: бюджет времени на кадр, по умолчанию 200 мс;
- `top_k`: сколько гипотез вернуть;
- `search_mode`: `auto`, `exhaustive` или `beam`;
- `per_detection_top_n`: размер shortlist ворот на детекцию в beam-режиме;
- `beam_width`: ширина beam search;
- `force_beam_for_detections`: начиная с какого числа детекций принудительно
  использовать beam вместо полного перебора;
- `max_ippe_candidates_per_pair`: сколько single-gate IPPE-кандидатов оставлять
  для пары `(detection, gate_id)`;
- `partial_spread_prune_m`: порог раннего отбрасывания веток beam search по
  XYZ-разбросу частичной гипотезы;
- `pairwise_compatibility_m`: максимальная XYZ-дистанция, при которой две
  assignment-пары считаются совместимыми перед расширением beam;
- `max_pose_solves`: сколько дорогих `PnPSolver.solve(...)` делать для top-K.

Алгоритм:

1. Один раз считает single-gate IPPE-кандидаты для пар `detection x gate_id`.
2. Быстро ранжирует ID-гипотезы по XYZ-consensus:

```text
score = spread_m * spread_weight
```

Внутри score не перебирается полный декартов product IPPE-кандидатов. Вместо
этого используется быстрый seed-based consensus: каждая single-gate позиция
пробуется как центр, для каждой детекции выбирается ближайший IPPE-кандидат,
затем spread уточняется вокруг медианы выбранных позиций.

`RMSE` хранится только как диагностическое поле. Он не используется для
ранжирования ID-гипотез и shortlist, потому что на симметричных воротах часто
даёт одинаково хорошие значения для правильных и неправильных ID.

3. Делает дорогой `PnPSolver.solve(...)` для top-K. Даже если fast ranking уже
   вышел за deadline, top-1 всё равно получает pose, а результат помечается как
   `timed_out=True`.
4. Возвращает `runtime_ms`, `timed_out`, число проверенных гипотез и число pose-solves.

Для 4+ детекций `auto`-режим переходит на beam search:

```text
per-detection shortlist
  -> pairwise XYZ compatibility
  -> beam expansion
  -> partial consensus pruning
  -> top-K
```

Pairwise compatibility заранее проверяет, есть ли у двух assignment-кандидатов
близкие single-gate camera positions. Если новая пара несовместима с уже
выбранными парами в ветке beam, эта ветка не расширяется. Shortlist строится в
порядке coarse-prior, без сортировки по `RMSE`, чтобы не потерять правильные
ворота в симметричных блоках.

Проверка:

```bash
python src/gate_localization/topk_hypotheses.py --validate
python src/gate_localization/topk_hypotheses.py --validate --candidate-from-gt
```

С локальным shortlist по GT-группе текущий модуль на multi-кадрах уже держит
правильную гипотезу в top-K. Без shortlist полный real-time поиск по всей карте
остаётся тяжёлым для 4-5 детекций и должен использоваться вместе с будущими
`Coarse prior` / `Tracking-first`.

## Coarse Prior + Refine

`coarse_refine.py` — V1-модуль, который использует примерную позицию камеры,
чтобы построить per-detection shortlist и выбрать гипотезу из top-K.

```python
from gate_localization import CoarseRefineLocalizer

result = CoarseRefineLocalizer().refine(
    detections,
    coarse_position_world=[39.0, 2.0, 26.0],
    q_m=10.0,
)

if result.success:
    print(result.selected.gate_ids)
    print(result.selected.pose.position_world, result.q_out_m)
```

Вход:

- `detections`: 4-точечные детекции ворот;
- `coarse_position_world`: примерная позиция камеры `[x, y, z]`;
- `q_m`: отклонение/радиус доверия к coarse position в метрах.

Выход:

- `selected.pose.position_world`: оценка позиции камеры/дрона;
- `q_out_m`: оценка неопределённости этой позиции в метрах;
- `selected.gate_ids`: выбранные ID ворот;
- `topk`: список проверенных top-K гипотез и диагностические поля.

V1 пока не использует yaw. Логика строится только по XZ-дистанции до ворот.

`q_m` не задаёт жёсткий максимальный радиус поиска и не участвует в финальном
выборе гипотезы. Он задаёт только ширину радиусных колец для формирования
candidate lists.

`q_out_m` — это не строгая статистическая sigma, а практический радиус доверия
для следующего кадра. Его можно передавать дальше как новый `q_m` вместе с
новой `coarse_position_world = selected.pose.position_world`.

Оценка `q_out_m` растёт, если:

- single-gate позиции плохо согласуются между собой (`spread_m` большой);
- top-1 гипотеза близка к следующим гипотезам, то есть есть неоднозначность;
- итоговая pose далеко от coarse position;
- обработка вышла за time budget (`timed_out=True`).

Оценка `q_out_m` уменьшается, когда видно больше ворот и они хорошо согласованы.

Радиусные кольца считаются с нижними пределами:

```text
near = max(q_m,      10 м)
mid  = max(2 * q_m,  25 м)
far  = max(4 * q_m,  40 м)
extra: всё дальше far
```

Так точная coarse позиция, например `q_m=0.5`, не схлопывает поиск до радиуса
0.5 м. Ворота в физически видимой зоне всё ещё остаются кандидатами.

Для каждой детекции формируется свой список кандидатов:

- крупная детекция: `near + mid`;
- средняя детекция: `near + mid + far`;
- маленькая детекция: `mid + far + near`.

Идея: близкие ворота чаще имеют большую площадь на изображении, но дальние
ворота не отрезаются полностью — они могут попасть в список как fallback.

Для 4+ детекций включается group-first режим:

```text
nearest coherent group -> Top-K -> fallback to wide candidate set if weak
```

Сначала берётся узкая локальная группа из ближайших к coarse position ворот.
Для 4 детекций используется `len(detections) + group_extra_candidates`; по
умолчанию `group_extra_candidates = 1`, чтобы не терять пограничные ворота.
Для 5+ детекций группа берётся ровно по числу детекций, чтобы не раздувать
перебор. Если узкий проход не дал pose, имеет слишком большой `spread_m` или
pose далеко от coarse position, запускается широкий проход по обычным radial
rings.

Full `PnPSolver.solve(...)` также ограничивается адаптивно: для 4+ детекций
считается только top-1 pose, а для 2-3 детекций остаётся до 6 pose-solves, чтобы
не терять случаи, где правильная гипотеза ниже в top-K.

Проверка:

```bash
python src/gate_localization/coarse_refine.py --validate
```

Текущий статус V1:

- хорошо работает на 2-3 воротах в рамках около 100-200 мс;
- на 4-5 воротах правильная гипотеза возвращается после удаления `RMSE`,
  fast-consensus scoring, pairwise compatibility, group-first shortlist и
  fast `PnPSolver.solve(refine=False)` для 4+ детекций;
- выбор среди top-K делается по близости `pose.position_world` к coarse position,
  без нормализации на `q_m`;
- следующий шаг — добавить yaw/frustum prior для симметричных 3-gate случаев.

## Известные Неоднозначности

На симметричных группах ворот возможна ситуация, когда выбранный `gate_id`
не совпадает с разметкой, но оценка позиции камеры остаётся достаточно близкой.

Текущие наблюдения:

- `frame_000370.jpg`
  - GT: `[11, 10, 9]`
  - выбранный вариант: `[11, 12, 13]`
  - ошибка позиции выбранного варианта: около `2.0 м`
  - причина: обе последовательности выглядят как зеркальные направления от
    `gate 11`, и без yaw/track-direction алгоритм не знает, в какую сторону
    выбирать блок.

- `frame_000646.jpg`
  - GT: `[21, 17, 19]`
  - выбранный вариант: `[21, 19, 17]`
  - ошибка позиции выбранного варианта: около `0.47 м`
  - причина: видны только 3 ворота из симметричного блока, поэтому две детекции
    могут поменяться местами между `17` и `19`.

Вывод: для оценки позиции такие варианты могут быть приемлемыми, но для строгого
assignment-а `detection -> gate_id` это ошибки. Следующие улучшения должны
добавить дополнительные признаки выбора среди top-K:

- coarse yaw;
- previous pose / tracking;
- направление движения по трассе;
- ожидаемую последовательность ворот;
- проверку projected left-right order.
