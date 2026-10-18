# SINTEZ нейрожелілік суррогаты ↔ Google DeepMind TORAX

**Нәтиже:** [`SINTEZ_vs_TORAX_report.pdf`](SINTEZ_vs_TORAX_report.pdf) (6 бет, қазақша).

Бірдей ITER режимінде (15 МА, 5,3 Тл, 50 МВт, бірдей n_e(ρ) мен педестал) екі тасымал моделі салыстырылды:

| | P_fus | Q |
|---|---|---|
| ITER жобасы | 500 МВт | 10 |
| TORAX · QLKNN (нағыз DeepMind коды, 1.4.3) | 532 МВт | 10,6 |
| SINTEZ · нейрожелі суррогат (`transport_model="neural_surrogate"`) | 67 МВт | 1,3 |

Қорытынды: қазіргі суррогат QLKNN-нің орнын баса алмайды — оның χ мәні оқу деректерінің «мұғалімінен»
(жобаның өз формулалары) мұраланған, QLKNN-нен әлдеқайда жоғары және критикалық градиент табалдырығы жоқ.
Толық талдау, шектеулер мен келесі қадамдар PDF-те.

## Файлдар

| Файл | Міндеті | Қай ортада |
|---|---|---|
| `run_sintez.py` | SINTEZ ITER разряды (120 с), бір тасымал моделі | жоба ортасы |
| `run_torax.py` | TORAX (QLKNN) сол ITER режимінде; `--no-sawteeth`, `--line-radiation-fraction` | **TORAX ортасы** |
| `torax_frozen.py` | QLKNN-ді SINTEZ-тің қатырылған күйінде есептеу (В тәжірибесі) | **TORAX ортасы** |
| `export_torax.py` | TORAX нәтижесін JSON-ға шығару + QLKNN латенттілігі | **TORAX ортасы** |
| `analyse.py` | бәрін `results.json`-ға біріктіру, желі↔QLKNN метрикалары, латенттілік | жоба ортасы |
| `build_report.py` | `results.json` → суреттер → PDF (барлық сан `results.json`-нан алынады) | matplotlib + reportlab |
| `results.json` | нәтижелердің жинағы (PDF осыдан жасалады) | |

TORAX жоба тәуелділіктеріне **қосылмаған**: оны бөлек виртуалды ортаға орнатыңыз.
`build_report.py` үшін matplotlib пен reportlab керек (`requirements.txt`-те жоқ), қаріп — DejaVu Sans.

## Қайта жүргізу

```bash
cd docs/torax_comparison
mkdir -p work

# 1) SINTEZ (жоба ортасы), әр модельге ~1 мин; нәтиже детерминді
for m in scaling_anchored cgm neural_surrogate bohm_gyrobohm gyrobohm; do
  python run_sintez.py $m work/sintez_$m.json
done

# 2) TORAX (бөлек орта)
python -m venv torax_env && torax_env/bin/pip install torax==1.4.3
T=torax_env/bin/python
$T run_torax.py work/sintez_neural_surrogate.json work/torax_main.json
$T run_torax.py work/sintez_neural_surrogate.json work/torax_nosaw.json --no-sawteeth
$T run_torax.py work/sintez_neural_surrogate.json work/torax_rad.json --line-radiation-fraction 0.11
for m in neural_surrogate scaling_anchored; do
  NU=1.2 $T torax_frozen.py work/sintez_$m.json work/frozen_$m.json
done
$T export_torax.py work

# 3) талдау және PDF
python analyse.py work          # -> results.json
python build_report.py          # -> SINTEZ_vs_TORAX_report.pdf
```

## Ескертпелер

* Бұл — екі әртүрлі кодты бірдей енгізулермен салыстыру, оның ішінде TORAX-қа SINTEZ-тің педесталы мен
  тығыздық профилі берілді (ядро тасымалының әсерін бөліп көру үшін). Бір режим (ITER H-режимі), бір рет жүргізу.
* `resistivity_multiplier = 200` (TORAX мысалындағыдай) омдық қызуды 200 есе үрлейді, сондықтан `run_torax.py`
  омдық көзді қоспайды (SINTEZ-те P_ohm ≈ 1,7 МВт).
* TORAX нәтижесі пилообразный (sawtooth) моделіне сезімтал: онсыз q₀ ≈ 0,4 және P_fus ≈ 816 МВт (физикалық емес).
* Латенттілік өлшемдері бір CPU-да шулы (диапазоны PDF-те көрсетілген); тек реті маңызды.
