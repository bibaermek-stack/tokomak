# SHRX-тің QLKNN-ге сәйкестігі {#sec-shrx-quality}

Бұл бөлім SHRX желісін оның мұғаліміне — QLKNN-ге — қарсы өлшейді. Мұнда «дұрыс жауап» тәжірибе емес, QLKNN болып табылады: сондықтан SHRX қанша жақсы болғанымен гирокинетикалық дәлдікте QLKNN-нен асып түспейді, тек оны SINTEZ-ке ыңғайлы түрде қайталайды. Тәжірибемен салыстыру @sec:torax-exp бөлімінде.

## Жергілікті дәлдік {#sec-local}

@tbl:accuracy SHRX мен QLKNN ағындарының сәйкестігін береді. Оқуға қатыспаған машиналарда (NSTX, DEMO) электрондық және иондық ағындар үшін $R^2\approx{{= c(S['accuracy']['groups']['held_out_solver_states']['r2_flux_e'], 3) }}$ және ${{= c(S['accuracy']['groups']['held_out_solver_states']['r2_flux_i'], 3) }}$; турбуленттілік қосулы нүктелерде $\chi$-дің медианалық салыстырмалы қатесі {{= pct(S['accuracy']['groups']['held_out_solver_states']['median_rel_err_e'], 1) }} (электрон) және {{= pct(S['accuracy']['groups']['held_out_solver_states']['median_rel_err_i'], 1) }} (ион), ал нүктелердің {{= pct(S['accuracy']['groups']['held_out_solver_states']['within25_e']) }} (электрон) және {{= pct(S['accuracy']['groups']['held_out_solver_states']['within25_i']) }} (ион) бөлігі QLKNN-нен 25 %-дан аз ауытқиды. Оқу машиналарында қате {{= c(S['accuracy']['groups']['held_out_solver_states']['median_rel_err_e']/S['accuracy']['groups']['train_machines_solver_states']['median_rel_err_e'], 1) }} есе кіші ({{= pct(S['accuracy']['groups']['train_machines_solver_states']['median_rel_err_e'], 1) }}): желі жаңа машинаға жалпылайды, бірақ оқу машиналарындай дәл емес. 25 % коридордан тыс нүктелер (@fig:parity) көбіне әлсіз ағын аймағында (гиро-Бом бірлігінде $q_e\lesssim1$), яғни табалдырық маңында жатыр.

:::table accuracy
SHRX мен QLKNN ағындарының сәйкестігі. $R^2$ — гиро-Бом ағындары бойынша; қате медианасы мен $\pm25$ % үлесі — $\chi$ бойынша, QLKNN-нің турбуленттілігі қосулы нүктелерде (@sec:method-train бөлімі).

| Жиын | Нүкте саны | $R^2$, $q_e$ | $R^2$, $q_i$ | Қате, $\chi_e$ | Қате, $\chi_i$ | $\pm25$ %, $\chi_e$ |
|:----------------------------|---------:|------:|------:|------:|------:|---------:|
{{= chr(10).join(f"| {lab} | {thou(g['n'])} | {c(g['r2_flux_e'], 3)} | {c(g['r2_flux_i'], 3)} | {pct(g['median_rel_err_e'], 1)} | {pct(g['median_rel_err_i'], 1)} | {pct(g['within25_e'])} |" for lab, g in [(l, S['accuracy']['groups'][k]) for l, k in (('Оқудан тыс (NSTX, DEMO), SINTEZ күйлері', 'held_out_solver_states'), ('Оқудан тыс, ауытқытылған күйлер', 'held_out_perturbed'), ('Оқу машиналары, SINTEZ күйлері', 'train_machines_solver_states'), ('Бүкіл қораптағы біркелкі нүктелер', 'box_uniform'))]) }}
:::

:::fig parity figures/parity.png
SHRX мен QLKNN электрондық жылу ағыны ($q_e$) оқудан тыс машиналарда: (а) DEMO, (б) NSTX. Тұтас сызық — $y=x$, үзік сызықтар — $\pm25$ %. Әр нүкте — симулятор барған бір күй.
:::

## Табалдырық бойынша қате {#sec-thr}

Қатаң тасымал үшін ағын бойынша орташа қате интегралды нәтижені болжамайды: 1-раундтағы желі ITER есебінде 267 МВт берді, ал QLKNN — {{= c(V['iter']['qlknn']['Pfus'], 0) }} МВт (@tbl:data). Бұл $\delta_f$ метрикасын @eq:deltaf формуласы енгізуге түрткі болды. Табалдырық аймағындағы үлгілерді қосқаннан кейін оқудан тыс машиналарда $\delta_f$ медианасы {{= pct(np.median(np.abs(E['threshold_held_out']['err_single'])), 1) }}, оқу машиналарында — {{= pct(np.median(np.abs(E['threshold_train']['err_single'])), 1) }} құрады (@fig:threshold); оқудан тыс машиналарда күйлердің {{= pct(np.mean(np.abs(E['threshold_held_out']['err_single']) < 0.10)) }}-інде $|\delta_f|<10$ % (ансамбльде {{= pct(np.mean(np.abs(E['threshold_held_out']['err_ensemble']) < 0.10)) }}). Оқудан тыс машиналарда $|\delta_f|<3$ % күйлер бөлігі ансамбльге өткенде {{= pct(np.mean(np.abs(E['threshold_held_out']['err_single']) < 0.03)) }}-дан {{= pct(np.mean(np.abs(E['threshold_held_out']['err_ensemble']) < 0.03)) }}-ға дейін артады.

:::fig threshold figures/threshold.svg
Критикалық градиент факторының қатесі $\delta_f=f_c^\mathrm{NN}/f_c^\mathrm{QLKNN}-1$ @eq:deltaf: (а) оқу, (б) оқудан тыс машиналар. Үзік сызық — жалғыз желі (ансамбльдің нөлінші мүшесі), тұтас сызық — бес мүшенің ансамблі. Күйлер саны: {{= thou(E['threshold_train']['n_both']) }} (а) және {{= thou(E['threshold_held_out']['n_both']) }} (б).
:::

## Ансамбль және белгісіздік {#sec-ens}

Бес мүшенің орташасы жалғыз желіден дәлірек (@fig:ensemble, а): NSTX-те $\chi_e$-нің медианалық қатесі {{= pct(E['nstx']['median_err_ensemble'], 1) }} (жалғыз желіде {{= pct(E['nstx']['median_err_single'], 1) }}), DEMO-да — {{= pct(E['demo']['median_err_ensemble'], 1) }} ({{= pct(E['demo']['median_err_single'], 1) }}); 25 %-дан кіші қатесі бар нүктелер үлесі {{= pct(E['nstx']['within25_single']) }}-дан {{= pct(E['nstx']['within25_ensemble']) }}-ға (NSTX) және {{= pct(E['demo']['within25_single']) }}-дан {{= pct(E['demo']['within25_ensemble']) }}-ға (DEMO) артады. Мүшелер арасындағы алшақтық $s$ қатемен үйлеседі: оның децильдері өскен сайын қате медианасы өседі (@fig:ensemble, б), ал «қате >25 %» оқиғасы үшін AUROC NSTX-те {{= c(E['nstx']['auroc'], 2) }}, DEMO-да {{= c(E['demo']['auroc'], 2) }}; Спирмен корреляциясы {{= c(E['nstx']['spearman'], 2) }} және {{= c(E['demo']['spearman'], 2) }}.

Алшақтық табалдырығы $\tau$ бір машинада таңдалып, екіншісінде тексерілді (@tbl:unc): таңдалған мәндер ({{= c(U['calibration']['calibrated_on_nstx_tested_on_demo']['threshold'], 2) }} және {{= c(U['calibration']['calibrated_on_demo_tested_on_nstx']['threshold'], 2) }}) бір-біріне жақын, сондықтан жеткізілетін ансамбльде $\tau=0{,}20$ алынды. Белгіленген нүктелердің қатесінің медианасы белгіленбегендерден {{= c(min(v['median_err_flagged']/v['median_err_unflagged'] for v in U['calibration'].values()), 1) }}–{{= c(max(v['median_err_flagged']/v['median_err_unflagged'] for v in U['calibration'].values()), 1) }} есе үлкен, бірақ дәлдік (precision) төмен: белгіленгеннің {{= pct(U['calibration']['calibrated_on_demo_tested_on_nstx']['precision']) }}–{{= pct(U['calibration']['calibrated_on_nstx_tested_on_demo']['precision']) }} ғана шын үлкен қатеге жатады. Сондықтан белгісіздік «қараңыз» белгісі ретінде пайдалы, ал «сенбеңіз» үкімі ретінде емес.

:::table unc
Алшақтық табалдырығының айқас калибровкасы (бір машинада таңдалып, екіншісінде тексерілді); қате ($\chi_e$, ансамбль-орташа) 25 %-дан жоғары болса, «үлкен».

| Таңдау → тексеру | $\tau$ | Белгіленген үлес | Дәлдік | Толықтық | Қате медианасы: белгіленген / белгіленбеген |
|:-----------|----:|---------:|--------:|--------:|---------------------:|
{{= chr(10).join(f"| {lab} | {c(v['threshold'], 2)} | {pct(v['flagged_share'])} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['median_err_flagged'], 1)} / {pct(v['median_err_unflagged'], 1)} |" for lab, v in (('NSTX → DEMO', U['calibration']['calibrated_on_nstx_tested_on_demo']), ('DEMO → NSTX', U['calibration']['calibrated_on_demo_tested_on_nstx']))) }}
:::

:::fig ensemble figures/ensemble.png
Ансамбль және белгісіздік: (а) $\chi_e$ салыстырмалы қатесінің жинақталған таралуы (жалғыз желі — үзік/нүктелі, ансамбль — тұтас/штрих-нүктелі); (б) мүшелер алшақтығының децилі бойынша қате медианасы; (в) әр нүкте үшін алшақтық пен қате (екі машина), тік үзік сызық $\tau=0{,}20$, көлденең нүктелі сызық — 25 % қате.
:::

Ансамбльдің интегралды нәтижеге әсері өлшеуге қиын (@sec:sensitivity бөлімі): ITER-де бес мүше $P_\mathrm{fus}$ үшін {{= c(V['spread']['iter_min'], 0) }}–{{= c(V['spread']['iter_max'], 0) }} МВт береді (орта шамадан ±{{= pct(V['spread']['iter_rel'], 1) }}), ансамбль — {{= c(V['spread']['iter_ens'], 0) }} МВт, жеткізілетін желі — {{= c(V['spread']['iter_ship'], 0) }} МВт, ал QLKNN — {{= c(V['iter']['qlknn']['Pfus'], 0) }} МВт.

## Жылдамдық {#sec-speed}

SHRX желісінің бір шақыруы 65 радиалды түйінде {{= c(V['lat'][65]['shrx_network'], 0) }} мкс (бір ағында, float32), ал JAX-та jit-пен құрастырылған QLKNN үшін float32 дәлдігінде {{= c(V['lat'][65]['qlknn_jit_float32'], 0) }} мкс және float64 дәлдігінде {{= c(V['lat'][65]['qlknn_jit_float64'], 0) }} мкс, яғни SHRX {{= c(V['lat'][65]['speed32'], 1) }} және {{= c(V['lat'][65]['speed64'], 1) }} есе жылдам (26 түйінде — {{= c(V['lat'][26]['speed32'], 1) }} және {{= c(V['lat'][26]['speed64'], 1) }} есе); толық шақыру (көпір + желі + $\chi$-ге айналдыру) үшін {{= c(V['lat'][65]['speed_full'], 1) }} есе ({{= c(V['lat'][65]['shrx_evaluate_full'], 0) }} және {{= c(V['lat'][65]['qlknn_evaluate_full'], 0) }} мкс). Желі QLKNN-нен {{= c(S['qlknn_parameters']/S['model']['n_parameters'], 1) }} есе аз параметрмен және JAX-сыз жұмыс істейді. Өлшем JAX-тың бір ағындағы CPU орындалуына қатысты; GPU/TPU-да QLKNN-нің басқа нәтиже беруі мүмкін. Басты нәтиже жылдамдықтың өзі емес, SHRX-ті JAX тәуелділігінсіз симулятордың ішіне енгізуге болатыны.

:::fig latency figures/latency.svg
Бір шақырудың уақыты (бір ағын, CPU; медиана, жолақтар — ауқым). Қатарлар: SHRX желісі; QLKNN (JAX, jit, float32 және float64); SHRX мен QLKNN толық шақыруы.
:::

## Бірдей симулятордағы желілер {#sec-machines}

SHRX-ті 12 машинаның скрипттелген разрядында (100 с) SINTEZ ішінде нағыз QLKNN-мен салыстырдық (@fig:machines). Кәдімгі токамактарда (ITER, DEMO, SPARC, JET, DIII-D, EAST, KSTAR, T-15MD) $W_\mathrm{th}$ үшін 8 машинаның {{= V['machines']['conv_Wth_within30'] }}-інде айырма 30 %-дан аз (ITER-де {{= pct(abs(V['machines']['Wth']['ratios']['iter'] - 1), 1) }}, JET-те {{= pct(abs(V['machines']['Wth']['ratios']['jet'] - 1), 1) }}). Қалған үшеуінде — DIII-D ({{= pct(V['machines']['Wth']['ratios']['diiid'] - 1) }}), EAST ({{= pct(V['machines']['Wth']['ratios']['east'] - 1) }}), KSTAR ({{= pct(V['machines']['Wth']['ratios']['kstar'] - 1) }}) — скрипттелген разряд өзі қатаң жүйе: бастапқы күйдің $10^{-6}$ шамасындағы өзгерісі нәтижені ондаған пайызға ығыстырады (@sec:sensitivity бөлімі), сондықтан айырманы желіге таңуға болмайды. Сфералық токамактардағы айырма да осындай (@sec:st бөлімі).

:::fig machines figures/machines.svg
SINTEZ ішіндегі SHRX пен нағыз QLKNN: 12 машинаның скрипттелген разрядынан (а) сақталған энергия, (б) орташа электрондық температура, (в) энергияны ұстау уақыты. Тұтас сызық — $y=x$, үзіктер — $\pm30$ %.
:::

## Симулятордың жылдамдатылуы {#sec-simspeed}

@sec:method-speed бөлімінде сипатталған өзгерістер SINTEZ-тің бірдей разрядын жылдамдатады (@tbl:simspeed, @fig:speedup): ITER-де SHRX моделімен {{= c(D['speed']['cases']['iter:shrx']['median_old'], 0) }} с орнына {{= c(D['speed']['cases']['iter:shrx']['median_new'], 0) }} с ({{= c(D['speed']['cases']['iter:shrx']['speedup'], 1) }} есе), барлық төрт жағдайда {{= c(min(v['speedup'] for v in D['speed']['cases'].values()), 1) }}–{{= c(max(v['speedup'] for v in D['speed']['cases'].values()), 1) }} есе. Өлшем бір ағында (`OMP_NUM_THREADS=1`), бірақ машинада басқа есептеулер қатар жүрді, сондықтан абсолют уақыттар жоғары, ал «бұрын/қазір» қатынасы бір шарттарда кезектесіп алынды (әр жағдай {{= len(D['speed']['cases']['iter:shrx']['old']) }} рет). Басты талап — нәтиженің бірдей болуы: төрт жағдайдың {{= sum(1 for v in D['speed']['cases'].values() if v['identical']) }}-інде екі нұсқаның соңғы $P_\mathrm{fus}$, $Q$, $T_e$, $W_\mathrm{th}$, $\beta_N$ мәндері ондық таңбаға дейін бірдей, ал алты эталондық жағдайда ауытқу нақты нөлге тең.

:::table simspeed
Бір разрядтың жұмыс уақыты (медиана, с) жылдамдатуға дейін және одан кейін. «Бірдей» — екі нұсқаның басылған нәтижелері дәл сәйкес.

| Жағдай | Бұрын, с | Кейін, с | Жылдамдау | Нәтиже бірдей |
|:-------------------|---------:|---------:|----------:|:-------------|
{{= chr(10).join(f"| {k.replace(':', ', ')} | {c(v['median_old'], 1)} | {c(v['median_new'], 1)} | {c(v['speedup'], 2)}× | {'иә' if v['identical'] else 'жоқ'} |" for k, v in D['speed']['cases'].items()) }}
:::

:::fig speedup figures/speedup.svg
Симулятор разрядының жұмыс уақыты жылдамдатуға дейін және одан кейін (120 с плазма уақыты, бір ағын); бағандар үстінде — жылдамдау.
:::
