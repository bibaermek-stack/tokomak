# SHRX — SINTEZ нейрожелілік тасымал моделі

**Есеп:** [`SHRX_vs_TORAX_report.pdf`](SHRX_vs_TORAX_report.pdf) (7 бет, қазақша) · **жоспар:** [`PLAN.md`](PLAN.md) ·
**ғылыми мақала** (қазақша, ақ-қара, KaTeX; TORAX-пен салыстыру, ансамбль, DB5, сфералық токамактар): [`paper_shrx/`](../../paper_shrx/README.md)

SHRX — бұрынғы «TORAX стиліндегі суррогаттың» (қазір `shrx_v0`) орнына келген модель. Ол SINTEZ-тің тасымал
теңдеуіне χ_e және χ_i береді және **нағыз QLKNN-нен** (DeepMind TORAX-тың нейрожелісі) дистилляцияланған:
QLKNN-ді SINTEZ жүргізген күйлерде белгілеп, 29 283 параметрлі кішкентай ReLU желісін оқытты. Желі тек NumPy-да жұмыс істейді.

| ITER, бірдей режим | P_fus | Q |
|---|---|---|
| TORAX · QLKNN (жеке код) | 532 МВт | 10,6 |
| SINTEZ + QLKNN | 478 МВт | 9,6 |
| **SINTEZ + SHRX** | **492 МВт** | **9,8** |
| SINTEZ + SHRX v0 (бұрынғы) | 67 МВт | 1,3 |

* QLKNN-нен жылдамырақ (желі 55/100 мкс vs 135/236 мкс float32, 229/491 мкс float64; 26/65 нүкте), параметрі 2,5 есе аз, JAX керек емес.
* Оқудан тыс машиналарда (NSTX, DEMO) QLKNN-мен R² ≈ 0,98–0,99.
* **Шегі:** оқытушысы QLKNN болғандықтан, гирокинетикалық дәлдікте QLKNN-нен асып түспейді. Басқа 10 машинаның скрипттелген
  разрядтарында сәйкестік дәлелденбеді (нәтиже салмақ нұсқасына сезімтал) — есептің 5-бөлімін қараңыз.

## Қолдану

```python
SolverConfig(machine="iter", transport_model="shrx")        # 'neural_surrogate' — ескі ат, осыған апарады
SolverConfig(machine="iter", transport_model="shrx_v0")     # бұрынғы желі
```
HTTP: `POST /ml/shrx` (QLKNN айнымалылары → χ_e, χ_i). Салмақтар: `backend/tokamak/shrx/weights.npz`.
Тесттер: `backend/tests/test_shrx.py` (JAX/TORAX керек емес; эталон мәндер `backend/tests/data/` ішінде).

## Қайта оқыту / қайта өлшеу

Оқыту мен тексеру **TORAX ортасын** қажет етеді (`python -m venv torax_env && torax_env/bin/pip install torax==1.4.3`);
SINTEZ-тің өзі тек NumPy. `backend/ml/shrx/` ішінде:

| Скрипт | Міндеті |
|---|---|
| `collect_states.py` | SINTEZ разрядынан күйлерді жинау (QLKNN кірісі түрінде) |
| `label_qlknn.py` | нағыз QLKNN-мен белгілеу (+ пертурбация, біркелкі үлгі) |
| `train_shrx.py` | оқыту (JAX/optax) → NumPy салмақтары |
| `validate_bridge.py` | көпірді TORAX-тың өз кірістерімен салыстыру |
| `eval_inloop.py`, `compare_machines.py` | SINTEZ ішінде SHRX vs QLKNN (бір/12 машина) |
| `local_compare.py`, `accuracy.py` | бірдей күйлерде жергілікті дәлдік |
| `bench_latency.py` | жылдамдық |
| `make_golden.py` | тест үшін эталон мәндер |

Есепті қайта жасау: `python collect_results.py <workdir> && python build_report.py` (`docs/shrx/`).
`transport_model="qlknn_ref"` — нағыз QLKNN SINTEZ ішінде (тек бағалау үшін, JAX керек).
