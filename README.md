# llama.cpp fork для SM8635 (Snapdragon 8s Gen 3, Hexagon v73)

Форк заточен только под SM8635 (Hexagon Arch v73, 4 MB VTCM, 4 HVX, 1 HMX).
Совместимость с v75+ не сохраняется.

Тестировалось на POCO F6 (Snapdragon 8s Gen 3).

## Что исправлено (коммит d6a0dcd90)

На v73 ломаются две группы операций. Патч в `ggml/src/ggml-hexagon/ggml-hexagon.cpp`:

1. Дефолты выключены (env-переменные всё равно переопределяют):
   - `opt_nhmx = 0` - HMX выключен, MUL_MAT/MUL_MAT_ID идут через HVX
   - `opt_opfusion = 0` - слияние операций выключено
   - `opt_fa_select = 0` - FLASH_ATTN_EXT на CPU
   - `opt_ssm_select = 0` - SSM_SCAN на CPU
   - `opt_gdn_select = 0` - GATED_DELTA_NET на CPU
   - `opt_li_select = 0` - LIGHTNING_INDEXER на CPU
2. В `supports_op` на CPU уходят операции, которые фейлят `test-backend-ops`
   даже при выключенных env: `ADD`, `SUB`, `DIV`, `SCALE`, `POOL_2D`, unary `CEIL`.

Причины фейлов разные:
- `MUL_MAT`/`MUL_MAT_ID` при `n >= 5` - включается HMX, выдаёт `ERR = inf`.
  Решено дефолтом `nhmx = 0` (уходят на HVX).
- `ADD`/`SUB`/`DIV`/`SCALE` - фейл по точности на тестах с `f16`
  (NMSE ~1e-7..3e-7 при пороге 1e-7, шум округления, не inf и не мусор).
  Реджект в `supports_op` жёсткий, без разделения по типам.
- `POOL_2D`/`CEIL` - фейлят, в LLM-инференсе не используются.

На HTP0 остаются рабочие операции: `MUL_MAT`/`MUL_MAT_ID` (HVX), `RMS_NORM`,
`ROPE`, `SOFT_MAX`, `MUL`, `SWIGLU`, `CPY`, `CONV_2D`, `CONV_3D` и др.

## Сборка

Сборка идёт через GitHub Actions, workflow `Build llama.cpp for SM8635`
(`.github/workflows/main.yml`), запускается на каждый push в master/main.

Локально (нужен Docker):

```sh
./scripts/snapdragon/build.py --target adb --push
```

Или вручную в контейнере:

```sh
docker run -it --rm -u $(id -u):$(id -g) --volume $(pwd):/workspace \
  --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-android:v0.8

# внутри контейнера
cp docs/backend/snapdragon/CMakeUserPresets.json .
cmake --preset arm64-android-snapdragon-release -B build-snapdragon
cmake --build build-snapdragon --parallel $(nproc)
cmake --install build-snapdragon --config Release --prefix pkg-android/llama.cpp
```

## Установка на устройство (Termux)

Скачай артефакт `llama-cpp-sm8635` из последнего запуска workflow,
затем на устройстве:

```sh
tar -xzf llama.cpp-sm8635-android.tar.gz -C ~
# получится ~/llama.cpp/{bin,lib,include,...}
```

Запускай бинарники с `LD_LIBRARY_PATH`:

```sh
export LD_LIBRARY_PATH=~/llama.cpp/lib
```

## Тестирование

Проверка бэкенда (ожидается: ни одного `FAIL`):

```sh
~/llama.cpp/bin/test-backend-ops test -b HTP0
```

Если хочешь увидеть только упавшие операции:

```sh
~/llama.cpp/bin/test-backend-ops test -b HTP0 2>&1 | grep FAIL
```

## Запуск модели

```sh
~/llama.cpp/bin/llama-server \
  -m ~/model.gguf \
  -dev HTP0 -ngl 99 --no-host \
  -c 4096 --host 0.0.0.0 --port 8080
```

Проверка: текст должен быть связным, без мусора, ни одной env-переменной
задавать не нужно. В логе загрузки ищи `hmx 0` - это подтверждение,
что HMX выключен:

```
ggml-hex: HTP0 hwinfo: threads 6, hvx 4, hmx 0, vtcm 4 MB
```

Если в логе `hmx 1` - где-то задан `GGML_HEXAGON_NHMX=1`.

## Env-переменные (все опциональны)

Для отладки и временного возврата старого поведения:

| Переменная | Дефолт после патча | Что делает |
|---|---|---|
| `GGML_HEXAGON_NHMX` | 0 | 1 = включить HMX (на v73 даёт inf) |
| `GGML_HEXAGON_OPFUSION` | 0 | 1 = включить слияние операций |
| `GGML_HEXAGON_FA_SELECT` | 0 | 2 = HMX->HVX->CPU, 1 = HVX->CPU, 0 = CPU |
| `GGML_HEXAGON_SSM_SELECT` | 0 | 2 = HMX->HVX, 1 = HVX, 0 = CPU |
| `GGML_HEXAGON_GDN_SELECT` | 0 | 2 = HMX->HVX, 1 = HVX, 0 = CPU |
| `GGML_HEXAGON_LI_SELECT` | 0 | 2 = HMX->HVX, 1 = HVX, 0 = CPU |
| `GGML_HEXAGON_MM_SELECT` | 2 | 2 = HMX->HVX->CPU, 1 = HVX->CPU, 0 = CPU |
| `GGML_HEXAGON_OPFILTER` | - | regex операций, которые НЕ брать на HTP |
| `GGML_HEXAGON_VERBOSE` | 0 | verbosity |
| `GGML_HEXAGON_POWER_MODE` | burst | питание DCVS |

Пример - принудительно включить всё старое поведение:

```sh
GGML_HEXAGON_NHMX=1 GGML_HEXAGON_OPFUSION=1 \
GGML_HEXAGON_FA_SELECT=2 GGML_HEXAGON_SSM_SELECT=2 \
GGML_HEXAGON_GDN_SELECT=2 GGML_HEXAGON_LI_SELECT=2 \
~/llama.cpp/bin/llama-server -m model.gguf -dev HTP0 -ngl 99
```

Реджект `ADD`/`SUB`/`DIV`/`SCALE`/`POOL_2D`/`CEIL` в `supports_op` жёсткий,
env-переменной нет - эти операции всегда идут на CPU.

## Диагностика проблем

1. NaN/inf в выводе -> проверь `hmx 0` в логе, проверь что не заданы
   `GGML_HEXAGON_*=1`.
2. `test-backend-ops` падает -> скинь вывод, сравни со списком операций
   из "Что исправлено". Новые фейлы - регрессия.
3. Медленно -> смотри, не ушли ли тяжёлые операции на CPU без нужды
   (`GGML_HEXAGON_VERBOSE=1` покажет, что именно на чём считается).

## Полезные документы в апстриме

- `docs/backend/snapdragon/README.md` - сборка для Snapdragon
- `docs/backend/snapdragon/developer.md` - устройство Hexagon-бэкенда
- `docs/build.md` - общие инструкции сборки
- `AGENTS.md` - правила вклада в апстрим (для этого форка в основном неактуальны)
