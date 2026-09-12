# СИНТЕЗ оператор-ИИ — дизайн

Дата: 2026-09-12  
Статус: бөлімдер бекітілді, пайдаланушы ревьюі күтіледі  
Жоба: интерактивті токамак (`tokomak-main`)

## Мақсат

Симуляцияға кеңесші-оператор қосу. ИИ диагностиканы IMAS тілінде оқиды, кеңес береді, растаудан кейін ғана актуатор қояды. Толық IMAS Data Dictionary каталогтан ізделеді. Сайтта модель атауы, провайдер атауы және API кілті көрсетілмейді.

## Шешімдер (бекітілген)

- Режим: кеңес + растаудан кейін басқару.
- Екі симуляция да: 3D пульт (0-D, `js/physics.js`) және Python бэкенд (1.5-D, `POST /simulate`).
- Екі режим бөлек: *тірі оператор* және *есеп*. Сандары бір жауапта салыстырылмайды.
- V1: HUD чат, актуатор карточкасы, бэкенд есебі, бар автопилот қалады, толық IDS каталогы ізделеді.
- Тәсіл: жергілікті IMAS каталог + құралды LLM. Эмбеддинг/fine-tune/RL жоқ.
- Провайдер: OpenRouter Chat Completions, серверде ғана. Кілт пайдаланушы берген; репозиторийге кірмейді.
- Модель идентификаторы тек сервер конфигінде. Браузер, HTML, CSS, пайдаланушыға көрінетін JSON, қате мәтіні — ешқайсысында модель/провайдер аты жоқ.
- Пульттегі көрінетін ат: **Кеңесші**.

## Архитектура

```
[HUD 3D + Кеңесші чат + растау карточкасы]
        | snapshot + хабар          | растау (браузер ғана)
        v                           v
[FastAPI: /ai/chat  /ai/imas  /ai/health  /simulate]
        |                    |
        |             [IMAS catalog.json]
        v
[OpenRouter Chat Completions, кілт тек env]
```

Кілтсіз: каталог пен пульт жұмыс істейді, чат жазуы жабық.  
ИИ құласа: қолмен NBI/газ/автопилот істейді.

## Компоненттер

### 1. IMAS каталог — `backend/imas/`

Жұмысы: Data Dictionary-ді рантайм іздеу индексіне айналдыру.

- Құрастыру: `tools/build_imas_catalog.py` GitHub-тан `iterorganization/IMAS-Data-Dictionary` ішіндегі `data_dictionary.xml` (немесе пакет ресурсы) оқиды, `backend/imas/catalog.json` жазады.
- Рантаймда XML жоқ, тек JSON. Railway образға JSON кіреді.
- Әр жазба: `ids`, `path`, `units`, `documentation` (400 символға дейін), `data_type`.
- Барлық IDS өрісі кіреді. Толық XML репозиторийге көшірілмейді.
- `search(q, limit=12)` — path/ids/documentation бойынша токенизация (`/`, `_`, `.`, бос орын), регистрсіз.
- `get(path)` — дәл жол.
- Атрибуция: ITER Organization, IMAS Data Dictionary, CC-BY-SA 4.0 — `backend/imas/README.md` және каталог метасында. Схеманы өзгертпейміз, индекс жасаймыз.

Тәуелділік: стандартты кітапхана. Рантаймда GitHub шақырылмайды.

### 2. Күй картасы — `backend/ai/state_map.py`

Жұмысы: симулятор өрісін IMAS жолына аудару. Картада жоқ өріс: «осы симуляцияда жоқ; IMAS-та мынадай» + каталог хиттері.

Live және compute картасы бөлек нысан. Араластыруға болмайды.

Live (HUD snapshot → IMAS). Төмендегі жолдар — мақсатты IDS отбасы. Құрастыру кезінде `catalog.json`-дағы нақты `path` қойылады; жоқ болса ең жақын `summary.*` / `nbi.*` өрісі алынады.

| HUD | IMAS жолы | бірлік |
|---|---|---|
| t | `pulse_schedule.time` | s |
| Ip | `summary.global_quantities.ip.value` | A (MA·1e6) |
| Bt | `summary.global_quantities.b0.value` | T |
| Te, Ti | `summary.global_quantities.t_e_volume_average` / `t_i_volume_average` | eV (keV·1e3) |
| ne | `summary.global_quantities.n_e_volume_average` | m^-3 (1e20) |
| fG, nG | `summary.global_quantities.greenwald_fraction` / `n_e_greenwald` | 1, m^-3 |
| Pfus | `summary.fusion.power` | W (MW·1e6) |
| Q | `summary.global_quantities.q_plus` | 1 |
| tauE | `summary.global_quantities.tau_energy` | s |
| H98 | `summary.global_quantities.h_98` | 1 |
| betaN | `summary.global_quantities.beta_normal` | 1 |
| q95 | `summary.global_quantities.q_95` | 1 |
| Zeff | `summary.global_quantities.zeff` | 1 |
| Pnbi / Picr / Pecr | `nbi.power_launched` / `ic_antennas.power_launched` / `ec_launchers.power_launched` | W |
| gasSet | `pulse_schedule.density_control.gas_puff` | 1 (клапан 0..1) |
| qDiv | `divertors.tungsten.target_heat_flux` | W/m^2 |
| neutronRate | `summary.fusion.neutron_power_total` | 1/s |
| hMode | `summary.global_quantities.h_mode` | bool |
| alarms | `summary.global_quantities.disruptions` мәтін | — |

Compute (`Simulator.scalars()` / `profiles()`) сол аттармен, бірақ `source: "compute"` және `t_end` бар. Профиль: `core_profiles.profiles_1d.{electrons.temperature, ion.temperature, electrons.density, q}`.

Бірлік конверсиясы картада тұрады. LLM-ге екі түр де беріледі: симулятор саны (HUD бірлігі) және IMAS SI.

### 3. Оркестр — `backend/ai/operator.py`

Жұмысы: хабарды модельге жіберу, құралдарды орындау, ұсыныс схемасын валидтеу.

Құралдар:

- `imas_lookup(query, path?)` — каталог.
- `read_mode_state()` — осы сұраудың snapshot/картасы. Басқа режимді оқымайды.
- `propose_actuators(items[])` — карточкаға ғана. Слайдерді қоймайды.
- `run_simulate(machine, t_end, ...)` — тек `mode=compute`. Ішінде бар `Simulator.run_scenario`. Live-да шақырылса қате.

Жауап (браузерге):

```json
{
  "reply": "қазақша мәтін",
  "proposals": [
    {
      "id": "uuid",
      "actuator": "p_nbi",
      "from": 16.0,
      "to": 33.0,
      "unit": "МВт",
      "reason": "...",
      "imas_path": "nbi.power_launched",
      "danger": false
    }
  ],
  "citations": [{"path": "nbi.power_launched", "units": "W", "documentation": "..."}],
  "mode": "live",
  "ai_available": true,
  "simulate": null
}
```

`model`, провайдер, кілт, raw tool dump — жоқ.

Актуатор allowlist: `p_nbi`, `p_icrf`, `p_ecrf`, `gas`, `ip`, `bt`.  
Қауіпті (екінші растау, `danger: true`): `disrupt`, `mitigate`.  
Пеллет/ELM: allowlist-те, `danger: false`, бір растау.

Лимит (қырқу, жауапта айтылады):

| актуатор | min | max |
|---|---|---|
| p_nbi | 0 | 50 МВт |
| p_icrf | 0 | 30 МВт |
| p_ecrf | 0 | 30 МВт |
| gas | 0 | 1 |
| ip | 0 | 17 МА |
| bt | 1 | 6 Тл |

Клиент `openai` пакеті, `base_url=https://openrouter.ai/api/v1`. Модель аты тек env `AI_MODEL`. Құрал қолдауы жоқ болса: сол схеманы JSON ретінде жазуды сұрау, сервер парсинг. Құрал циклі ең көбі 6 қадам, timeout 45 с.

Систем промпт (қысқаша): қазақша жауап; Кеңесші атың; модель/провайдерді атама; live мен compute-ты араластырма; актуаторды растаусыз қойма; IMAS жолын дәйексөз ет.

### 4. HTTP — `backend/api/main.py`

- `GET /ai/health` → `{ "ai_available": bool }`. Модель аты жоқ.
- `GET /ai/imas?q=` → каталог хиттері (path, units, documentation). Auth жоқ, кілтсіз жұмыс істейді.
- `POST /ai/chat` body: `{ "message", "mode": "live"|"compute", "snapshot?", "history?" }`. `mode=live` және snapshot жоқ → 400. Кілтсіз → 503 `{ "ai_available": false, "reply": "Кеңесші әлі қосылмаған." }`.
- `POST /simulate` өзгермейді, оркестр ішінен шақырылады.
- Статика: `GET /` және `/js/*`, `/css/*` осы FastAPI-дан.

CORS: қолданыстағы `allow_origins=["*"]` қалады; кілт браузерге берілмейді.

### 5. HUD — `js/ai.js`, `index.html`, `css/style.css`, `js/main.js`

- Симуляция сатысында `#hud` ішінде жиналатын панель, тақырып **Кеңесші**.
- Инпут, хабарлар, IMAS citation (path + units), карточка Растау / Жоқ.
- Растау: `phys` сетпойнттары (`PnbiSet`, `PicrSet`, `PecrSet`, `gasSet`, `IpSet`, `Bt`), автопилот off, слайдер DOM-мен синхрон. Серверге apply шақырылмайды.
- Жоқ: proposal drop.
- `danger: true` — қызыл карточка, екінші Растау.
- Есеп нәтижесі (`simulate`) чатта; HUD күйіне жазылмайды.
- Кілтсіз health: инпут disabled, «Кеңесші әлі қосылмаған».
- `js/physics.js` теңдеулеріне тиіспеу. Автопилот сценарийі қалады.

Көрінетін жолдарда тыйым (HTML, CSS, JS жолдары, чат, қате, `/ai/*` JSON): `deepseek`, `DeepSeek`, `openrouter`, `OpenRouter`, `flash`, `sk-or`, `AI_MODEL` мәні, кез келген модель id. Сервер логында модель id болуы мүмкін — сайт емес.

### 6. Конфиг және құпия

Env (сервер):

| айнымалы | мәні |
|---|---|
| `OPENROUTER_API_KEY` | пайдаланушы кілті |
| `AI_BASE_URL` | `https://openrouter.ai/api/v1` |
| `AI_MODEL` | сервердегі модель id (сайтқа шықпайды) |
| `AI_HTTP_REFERER` | қосымша, OpenRouter рейтингі; бос қалдыруға болады |
| `AI_APP_TITLE` | `СИНТЕЗ` — модель аты емес |

`.env` gitignore. `.env.example` кілтсіз. Кілт HTML/JS бандліне кірмейді.

Кілт чатта берілген. Оны файлға немесе Railway variable-ға ғана қоямыз, коммитқа емес. Чатта ашық тұрғандықтан, пайдаланушыға кейін ауыстыруға болады.

## Дерек ағыны

**Live**

1. Оператор жазады. Браузер `mode=live` + ағымдағы snapshot жібереді.
2. Сервер карталайды, модель + құрал.
3. `proposals` карточка. Слайдер қозғалмайды.
4. Растау → тек браузер сетпойнт қояды, автопилот өшеді.
5. Snapshot ескірсе (proposal `from` қазіргі мәнмен сәйкес емес) — қолданылмайды, қайта сұрау.

**Compute**

1. «ITER-ді 120 с есепте» → `mode=compute`, `run_simulate`.
2. Бар solver. Жауап IMAS аттарымен қысқарады.
3. Чатта ғана. Live слайдерге жазылмайды.

Бір HTTP сұрауда бір `mode`.

## Қателер

| Жағдай | Мінез-құлық |
|---|---|
| кілт жоқ | health false, chat 503, IMAS іздеу тірі |
| OpenRouter 4xx/5xx/timeout | «Кеңесші уақытша қолжетімсіз», пульт тірі, модель аты жоқ |
| каталог бос/жоқ | health-та `catalog: false`, іздеу бос массив |
| өріс табылмады | «IMAS-та жоқ / осы модельде жоқ» (симулятор мағынасында) |
| simulate exception | есеп қатесі чатта, HUD өзгермейді |
| лимиттен тыс | қырқу + себеп |
| snapshot mismatch | proposal қолданылмайды |
| құрал циклі бітті | сол кездегі мәтін, proposalсыз болуы мүмкін |

## Тест

Кілтсіз pytest (`backend/tests/`):

1. Каталог: `summary`, `nbi`, `core_profiles`, `equilibrium` іздеуден шығады.
2. Карта: `Pfus` → `summary.fusion.power`.
3. `/ai/chat` мок: `propose_actuators` HTTP жағында Tokamak күйін өзгертпейді.
4. Лимит қырқу.
5. Live chat-тан `run_simulate` шақыру — құрал қатесі.
6. Compute жауабында live snapshot жоқ.
7. Кілтсіз `/ai/health` және `/ai/chat`.
8. Chat JSON-да `model` / провайдер жолдары жоқ.
9. Бар физика тесті (51+) өтеді.

Мок: `operator` OpenRouter-ді httpx/openai stub-пен алмастырады. Тірі кілт CI-ға міндетті емес.

## Railway

- Бір сервис. Түбірден: FastAPI статика + API.
- Start: `uvicorn` порт `$PORT`.
- `OPENROUTER_API_KEY`, `AI_MODEL`, `AI_BASE_URL` — Railway variables.
- Каталог JSON образбен. Билдде XML жүктеу міндетті емес, егер JSON репода болса.
- Локальде `railway link` жоқ; бар аккаунтқа (biba.ermek@gmail.com) бар жобаға линк, жаңа жоба ашпау.
- Деплой алдында health: `ai_available` кілт қойылған соң true.

## Файлдар

Жаңа:

- `backend/imas/__init__.py`, `catalog.py`, `catalog.json`, `README.md`
- `backend/ai/__init__.py`, `operator.py`, `state_map.py`, `prompts.py`
- `tools/build_imas_catalog.py`
- `js/ai.js`
- `backend/tests/test_ai.py`, `test_imas_catalog.py`
- `.env.example`
- `docs/superpowers/specs/2026-09-12-tokamak-ai-operator-design.md` (осы файл)

Өзгереді:

- `backend/api/main.py` — `/ai/*`, статика
- `backend/requirements.txt` — `openai`, `httpx`
- `index.html`, `css/style.css`, `js/main.js` — Кеңесші панелі
- `.gitignore` — `.env`
- Railway start (Dockerfile немесе nixpacks/`Procfile`) — бір процесс

Тиіспейді: `js/physics.js` физикасы, `tokamak/*.py` теңдеулері, автопилот `autoSequence`.

## V1-ге кірмейді

- HUD пен 1.5-D-ны бір плазма ету
- Fine-tune, RL, эмбеддинг RAG
- IMAS HDF5/IDS файл экспорт
- Көп қолданушы сессия/логин
- Сайтта модель атын көрсету немесе ауыстыру UI

## Сәттілік критерийі

1. Симуляцияда Кеңесші чаты ашылады; модель аты еш жерде жоқ.
2. Кілтпен: IMAS өрісін сұрау → path+units; «NBI 33 МВт» → карточка → Растау → слайдер мен плазма өзгереді, автопилот өшеді.
3. Жоқ → өзгеріс жоқ.
4. Compute сұрау → `/simulate` нәтижесі чатта, HUD слайдері сол күйінде.
5. Кілтсіз: пульт пен `GET /ai/imas` тірі.
6. Ескі валидация/pytest өтеді.
