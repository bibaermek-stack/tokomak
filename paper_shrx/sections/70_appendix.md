# Қосымша А. SHRX гиперпараметрлері мен қайта өндіру {-}

:::table hyper
SHRX желісі мен оқыту параметрлері.

| Параметр | Мәні |
|:------------------|:--------------------------------------------|
| Кіріс | $X\in\mathbb{R}^{10}$ ($R/L_{T_i},R/L_{T_e},R/L_{n_e},R/L_{n_i},q,\hat s,x,T_i/T_e,\log_{10}\nu^{*},n_i/n_e$) |
| Шығыс | $(q_i,q_e,\Gamma_e)$, гиро-Бом бірлігі |
| Архитектура | {{= S['model']['depth'] }} жасырын қабат $\times$ {{= S['model']['width'] }} нейрон, ReLU |
| Параметр саны | {{= thou(S['model']['n_parameters']) }} (QLKNN: {{= thou(S['qlknn_parameters']) }}) |
| Оқу жолдары | {{= thou(S['model']['n_train']) }} |
| Оқудан тыс машиналар | {{= ', '.join(m.upper() for m in S['model']['holdout_machines']) }} |
| Оптимизатор | Adam, бастапқы қадам $2\cdot10^{-3}$, косинустық азаю |
| Пакет / дәуір | 4096 / {{= S['model']['epochs'] }} |
| Ансамбль | 5 желі, әр түрлі сид, жолдардың 85 % іріктемесі; $\tau=0{,}20$ |
| Мұғалім | QLKNN `qlknn_7_11_v1` (Apache-2.0), TORAX 1.4.3 арқылы |
:::

**Бағдарламалық орта.** NumPy [@numpy] және JAX [@jax] пайдаланылған. Python {{= S['versions']['python'] }}, NumPy {{= S['versions']['numpy'] }}; TORAX {{= S['versions']['torax'] }} (JAX {{= S['versions']['jax'] }}, CPU). Барлық есептеулер бір 4 ядролы x86‑64 виртуалды машинада (16 ГБ жад) жүргізілді; уақыт өлшемдері бір ағында алынды.

**Қайта өндіру.** Мақаладағы әр сан `paper_shrx/data/*.json` файлдарынан макроспен алынады; суреттер `make_figures.py`, мәтін `build.py` арқылы құрастырылады. Эксперименттер `backend/ml/shrx/` және `backend/ml/shrx/db5/` ішіндегі скрипттермен жүргізіледі (оқыту — `train_shrx.py`, табалдырық — `threshold_eval.py`, ансамбль — `make_ensemble.py`, `uncertainty_eval.py`, тәжірибелік салыстыру — `db5/sintez_runner.py`, `db5/torax_runner.py`, `db5/analyse.py`, `db5/core_analyse.py`, сфералық токамактар — `db5/st_frozen.py`, `st/collect_tglf_states.py`). Сынақтар: `backend/tests/test_shrx.py`, `test_solver_regression.py`, `test_speed_equivalence.py`.

**Деректер мен код қолжетімділігі.** SHRX салмақтары (`backend/tokamak/shrx/weights.npz`, `ensemble.npz`) жоба репозиторийінде; тәжірибелік деректер қорының таңдалған жазбалары `docs/shrx/db5_selection/` ішінде (бастапқы дерек — ITPA H-режим қорының FUSE.jl-мен таратылатын ашық үлгісі). TGLFNN-STEP салмақтары репозиторийге қосылмаған: оларды UKAEA жариялаған (LGPL) және `TGLFNN_STEP_PKL` айнымалысымен көрсетіледі.
