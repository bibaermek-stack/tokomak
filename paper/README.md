# Мақала: токамак плазмасын магниттік басқаруға арналған нейрожелілік контроллер

`article_kk.docx` — қазақша ғылыми мақала (A4, Times New Roman 14, 1,5 интервал,
формулалар — Word-тың өз формулалары). Мақаладағы **барлық сан** `results.json`-нан
автоматты түрде алынады; қолмен жазылған сан жоқ.

| Файл | Мазмұны |
|---|---|
| `part0_front.md` … `part4_discussion.md` | мақала мәтіні (Markdown + LaTeX, `{{…}}` — нәтиже макростары) |
| `refs.py` | әдебиеттер тізімі (ГОСТ), нөмірлеу — мәтіндегі ретпен |
| `build_paper.py` | мәтін + `results.json` → `article_kk.docx` (pandoc + python-docx) |
| `make_figures.py` | суреттер → `figures/` |
| `collect_results.py` | `tokamak.control.experiments` шығыстарын және оқу тарихын `results.json`-ға біріктіру |
| `policies/` | 3 нұсқа × 3 оқыту сиді, мақала нәтижелерін қайта өндіру үшін |

## Қайта құру

```bash
pip install pypandoc_binary python-docx matplotlib scipy
cd paper
python make_figures.py results.json figures \
    PPO-DR-AC=../backend/tokamak/control/training/ppo_dr_ac_seed0.json \
    PPO-DR=../backend/tokamak/control/training/ppo_dr_seed0.json \
    PPO-0=../backend/tokamak/control/training/ppo0_seed0.json
python build_paper.py results.json article_kk.docx
```

Эксперименттерді қайта жүргізу (~1 сағат, 3 процесс; `backend/`-тен):

```bash
P="--policy PPO-0=../paper/policies/ppo0_seed0.npz --policy PPO-DR=../paper/policies/ppo_dr_seed0.npz --policy PPO-DR-AC=../paper/policies/ppo_dr_ac_seed0.npz"
python -m tokamak.control.experiments --out r1.json $P --only device,draws,main,traces,cost
python -m tokamak.control.experiments --out r2.json $P --only ood
python -m tokamak.control.experiments --out r3.json $P --only delay,safety
python -m tokamak.control.experiments --out r4.json --only growth
# сидтер бойынша: --policy PPO-0#0=… --policy PPO-0#1=… (т.б.) --only seeds_check
```

## Орындалмаған және тексерілмеген тұстар

* **Авторлар мен ұйым** — мақалада `[авторлардың аты-жөні]` деп қалдырылған.
* **Әдебиеттер тізімі** жадтан жазылған, жарияланушылардың сайттарына қол жеткізусіз
  (желі саясаты тыйым салады). Том, бет, DOI дәлдігін жіберер алдында тексеру керек.
* **Kaggle деректері қолданылған жоқ**: Kaggle-ге қолжетімділік жоқ болды
  (`kaggle.com` — 403) және API кілті берілмеді. Контроллер симуляторда күшейтілген
  оқытумен оқытылады, ол үшін кестелік дерек жиыны жоқ; Kaggle деректері басқа
  міндетке (мысалы, дизрупцияны болжау немесе ұстау уақытының скейлингін
  регрессиялау) ғана жарайды.
* Салыстыру кестесіндегі (1-кесте) басқа авторлардың сандары қайта өндірілген жоқ.
* Нәтижелер **тек симуляцияға** қатысты.

## Kaggle (J-TEXT) бөлімі — деректер тәуелді

`part3b_jtext.md` (3.9-бөлім) мақалаға **тек** `paper/kaggle_results.json` болғанда қосылады.
Бұл файл — `backend/ml/train_jtext.py` шығаратын `jtext_report.json`-ның көшірмесі
(`backend/ml/models/jtext_report.json`). Оны қойып, `python build_paper.py results.json article_kk.docx`
қайта іске қосыңыз; сандардың бәрі сол файлдан алынады. Файл жоқ болса, бөлім құрылмайды.
