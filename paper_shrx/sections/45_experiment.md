## C деңгейі: ITPA H-режим деректер қоры {#sec-torax-exp}

### Болжамды пьедестал {#sec-exp-predictive}

SINTEZ + SHRX {{= B['n_conv'] }} жазбаның бәрінде жұмыс істеді, бірақ тек {{= len([d for d in B['records'] if d.get('valid')]) }}-і ({{= pct(len([d for d in B['records'] if d.get('valid')]) / B['n_conv']) }}) тексерілген жүгіру критерийіне (@sec:method-db5 бөлімі) сай келді. Қалғандарының басым бөлігі жылулық жүгіріске кетті (@tbl:bymachine): {{= len([d for d in B['records'] if d.get('ok') and not d.get('no_runaway')]) }} жүгіруде $\beta_N\ge3{,}5$ немесе өздігінен туатын ток үлесі $\ge0{,}8$. Мұндай жүгірудегі пьедестал төбесінің температурасының медианасы {{= c(np.median([d['T_ped'] for d in B['records'] if d.get('ok') and not d.get('no_runaway')]), 1) }} кэВ, ал жарамдыларда — {{= c(np.median([d['T_ped'] for d in B['records'] if d.get('valid')]), 2) }} кэВ. Бұл EPED-ке ұқсас пьедестал моделінің кәдімгі мүмкіндігінен тыс режимдердегі (төмен $I_p$, жоғары $\beta_p$, ескі шектегіш құрылғылар) қателігі: сыртқы шекаралық шарт болғандықтан сақталған энергия негізінен $T_\mathrm{ped}$-пен анықталады, ядролық тасымал оны тек сәл өзгертеді. Бірде-бір жарамды жүгірісі жоқ құрылғылар: {{= ', '.join(t for t, m in B['by_machine'].items() if m['n_valid'] == 0) }}.

:::table bymachine
Құрылғылар бойынша нәтижелер (кәдімгі аспект қатынасы). «Жарамды» — үш критерийдің үшеуіне де сай жүгіру саны; «тығыздық», «қуат», «жүгіріссіз» — әр критерийге жеке сай жүгіру саны. Қатынастар $\tilde r=W^\mathrm{pred}/W^\mathrm{exp}$ — құрылғы бойынша медиана; $1/H_{98}$ — деректер қорындағы IPB98(y,2) қатынасы (AUG-те $H_{98}$ жазбаларда жоқ).

| Құрылғы | $a/R$ | $N$ | Жарамды | Тығыздық | Қуат | Жүгіріссіз | $\tilde r$, SHRX (бәрі) | $\tilde r$, SHRX (жарамды) | $\tilde r$, TORAX | $1/H_{98}$ |
|:--------|-----:|----:|--------:|---------:|-----:|----------:|-----------:|-----------:|-----------:|-------:|
{{= chr(10).join(f"| {t} | {c(m['eps'], 2)} | {m['n']} | {m['n_valid']} | {m['dens_ok']} | {m['pow_ok']} | {m['no_runaway']} | {c(m['shrx_all_median'], 2)} | {c(m['shrx_valid_median'], 2)} | {c(m['torax_median'], 2)} | {c(1 / m['H98_median'], 2) if m['H98_median'] else '—'} |" for t, m in B['by_machine'].items()) }}
:::

@tbl:db5stats қорытынды көрсеткіштерді береді. Тексерілген {{= B['stats']['shrx_valid']['n'] }} жүгіруде SINTEZ + SHRX сақталған энергияны медианамен {{= pct(B['stats']['shrx_valid']['median'] - 1) }} асыра бағалайды, $\ln r$-дің ауытқуы $\sigma_{\ln}={{= c(B['stats']['shrx_valid']['sigma_ln'], 2) }}$ және жазбалардың {{= pct(B['stats']['shrx_valid']['f30']) }}-і ғана ±30 % ішінде. Дәл осы деректерге сәйкестендірілген IPB98(y,2) регрессиясы сол жазбаларда {{= c(B['stats']['ipb_on_valid']['sigma_ln'], 2) }} ауытқу және {{= pct(B['stats']['ipb_on_valid']['f30']) }} (±30 %) береді. Яғни **болжамды пьедесталмен SINTEZ + SHRX регрессиядан әлсіз**: ол деректі «білмейді» және пьедесталды дәл болжай алмайды (@fig:db5parity, @fig:db5cdf).

:::table db5stats
Сақталған энергияның болжамы: $\tilde r=W^\mathrm{pred}/W^\mathrm{exp}$ @eq:ratio. Соңғы үш қатар — TORAX-пен салыстыру үшін ортақ жазбалар жиыны (SINTEZ + SHRX тексерілген жүгіруінің TORAX да есептелгені).

| Болжам | $N$ | $\tilde r$ | $\sigma_{\ln}$ | $f_{30}$ |
|:----------------------------------|----:|------:|---------:|------:|
{{= chr(10).join(f"| {lab} | {st['n']} | {c(st['median'], 2)} | {c(st['sigma_ln'], 2)} | {pct(st['f30'])} |" for lab, st in (('SINTEZ + SHRX, барлық жүгіру', B['stats']['shrx_all']), ('SINTEZ + SHRX, тексерілген жүгіру', B['stats']['shrx_valid']), ('IPB98(y,2) регрессиясы, сол жазбалар', B['stats']['ipb_on_valid']), ('Ортақ жиында: SINTEZ + SHRX', B['stats']['shrx_on_torax_set']), ('Ортақ жиында: TORAX', B['stats']['torax_valid']), ('Ортақ жиында: IPB98(y,2)', B['stats']['ipb_on_torax_set']))) }}
:::

:::fig db5parity figures/db5_parity.png
Сақталған энергия: болжам мен тәжірибе. (а) SINTEZ + SHRX: кресттер — тексеруден өтпеген жүгірулер, маркерлер — тексерілгендер (құрылғылар бойынша); (б) TORAX және (в) IPB98(y,2) — сол тексерілген жазбалар үшін. Тұтас сызық — $y=x$, үзік сызықтар — $\pm30$ %.
:::

:::fig db5cdf figures/db5_cdf.svg
Болжам қателігінің жинақталған таралуы: $|\ln(W^\mathrm{pred}/W^\mathrm{exp})|$. Нүктелі сызық — барлық жүгіру, қалғандары — тексерілген жазбалар. Тік нүктелі сызық — ±30 % ($\ln1{,}3=0{,}26$).
:::

**TORAX-пен салыстыру.** TORAX сол SINTEZ-тің қойған тығыздығы, пьедесталы және қуатымен, бірақ өз кодымен (дөңгелек қима, тіркелген $q$, ток диффузиясыз) {{= B['stats']['torax_valid']['n'] }} жазбада жүргізілді. Тәжірибеге қатысты TORAX $\tilde r={{= c(B['stats']['torax_valid']['median'], 2) }}$, $\sigma_{\ln}={{= c(B['stats']['torax_valid']['sigma_ln'], 2) }}$, $f_{30}={{= pctm(B['stats']['torax_valid']['f30']) }}$, ал сол жиында SINTEZ + SHRX үшін {{= c(B['stats']['shrx_on_torax_set']['median'], 2) }}, {{= c(B['stats']['shrx_on_torax_set']['sigma_ln'], 2) }} және {{= pct(B['stats']['shrx_on_torax_set']['f30']) }}. Екі код бір-біріне тәжірибеден әлдеқайда жақын (@fig:db5pair): $\ln W$ бойынша корреляция $r={{= c(B['shrx_vs_torax']['corr_ln'], 3) }}$, қатынастың медианасы {{= c(B['shrx_vs_torax']['median_ratio'], 2) }}, $\sigma_{\ln}={{= c(B['shrx_vs_torax']['sigma_ln'], 2) }}$ ($N={{= B['shrx_vs_torax']['n'] }}$), яғни екі модельдің арасындағы айырма ($\sigma_{\ln}={{= c(B['shrx_vs_torax']['sigma_ln'], 2) }}$) әрқайсысының тәжірибеден айырмасынан ({{= c(B['stats']['shrx_on_torax_set']['sigma_ln'], 2) }} және {{= c(B['stats']['torax_valid']['sigma_ln'], 2) }}) екі есе кіші. Кіріс деректер бірдей болғандықтан, тәжірибеден ауытқудың басым бөлігі екі кодқа ортақ нәрседен — пьедесталдан, тығыздық профилінен, қыздыру көздерінің таралуынан — келеді, ал тасымал желісінен емес.

:::fig db5pair figures/db5_pair.svg
Бірдей кіріс деректермен SINTEZ + SHRX және TORAX болжаған сақталған энергия (тексерілген жазбалар). Тұтас сызық — $y=x$, үзік — $\pm30$ %.
:::

### «Тек ядро» режимі {#sec-exp-core}

Пьедестал қателігін тасымалдың қателігінен бөлу үшін пьедестал төбесінің температурасы өлшенген $W_\mathrm{th}$ қайта шығарылатындай таңдалды @eq:calib. Орталық температурасы өлшенген {{= B['core_counts']['n_runs'] }} жазбадан {{= B['core_counts']['n_good'] }}-ы тазалық критерийлерінен өтті (тығыздық {{= B['core_counts']['dens_ok'] }}, қуат {{= B['core_counts']['pow_ok'] }}, жүгіріссіз {{= B['core_counts']['no_runaway'] }}, пьедестал іздеу ауқымынан шықпады) және TORAX-пен {{= B['core_counts']['n_torax'] }} жазбада салыстырылды. Дәл сол SINTEZ-пьедесталмен TORAX сақталған энергияны да жақсы береді ($W^\mathrm{TORAX}/W^\mathrm{exp}$ медианасы {{= c(B['core_pair']['W_torax_over_exp']['median'], 2) }}, ауытқу {{= c(B['core_pair']['W_torax_over_exp']['sigma_ln'], 2) }}, {{= pct(B['core_pair']['W_torax_over_exp']['f30']) }} жазба ±30 % ішінде): энергияның көп бөлігін пьедестал анықтайды.

Ядро профилінің пішінін орталық температуралар көрсетеді (@tbl:core, @fig:core). Екі код та орталық электрондық температураны асыра бағалайды: SINTEZ + SHRX үшін $T_{e0}^\mathrm{pred}/T_{e0}^\mathrm{exp}$ медианасы {{= c(B['core_stats']['Te0_shrx']['median'], 2) }} ($\sigma_{\ln}={{= c(B['core_stats']['Te0_shrx']['sigma_ln'], 2) }}$), TORAX үшін {{= c(B['core_stats']['Te0_torax']['median'], 2) }} ({{= c(B['core_stats']['Te0_torax']['sigma_ln'], 2) }}). Иондық орталық температура үшін медиана SHRX-те {{= c(B['core_stats']['Ti0_shrx']['median'], 2) }}, TORAX-та {{= c(B['core_stats']['Ti0_torax']['median'], 2) }}, бірақ шашырау үлкен ($\sigma_{\ln}={{= c(B['core_stats']['Ti0_shrx']['sigma_ln'], 2) }}$ және {{= c(B['core_stats']['Ti0_torax']['sigma_ln'], 2) }}). Айқын жүйелі қателік — $T_i/T_e$ қатынасы: өлшенген орташа $T_{i0}/T_{e0}\approx{{= c(B['core_ti_te']['exp_median'], 2) }}$, ал SINTEZ + SHRX {{= c(B['core_ti_te']['pred_median'], 2) }} береді ($N={{= B['core_ti_te']['n'] }}$). Екі код бір-біріне жақын: $T_{e0}$ бойынша SHRX/TORAX қатынасы {{= c(B['core_pair']['Te0_shrx_over_torax']['median'], 2) }} ($\sigma_{\ln}={{= c(B['core_pair']['Te0_shrx_over_torax']['sigma_ln'], 2) }}$), $T_{i0}$ бойынша — {{= c(B['core_pair']['Ti0_shrx_over_torax']['median'], 2) }} ($\sigma_{\ln}={{= c(B['core_pair']['Ti0_shrx_over_torax']['sigma_ln'], 2) }}$). Сондықтан $T_e$ мен $T_i$ арасындағы тепе-теңсіздікті желінің ғана қателігі деп атау қате: ол QLKNN-ге де TORAX-қа да ортақ, демек симулятор (қыздыру көздері, электрон–ион алмасуы) мен QLKNN-нің электрондық арнасына байланысты болуы мүмкін; бұл гипотезаны бөлек тексермедік.

:::table core
Орталық температуралардың болжамы «тек ядро» режимінде (пьедестал өлшенген $W_\mathrm{th}$-ге келтірілген): болжам/өлшем.

| Шама | Код | $N$ | Медиана | $\sigma_{\ln}$ | $f_{30}$ |
|:----------|:--------------|----:|--------:|--------:|------:|
{{= chr(10).join(f"| {q} | {code} | {st['n']} | {c(st['median'], 2)} | {c(st['sigma_ln'], 2)} | {pct(st['f30'])} |" for q, code, st in (('$T_{e0}$', 'SINTEZ + SHRX', B['core_stats']['Te0_shrx']), ('$T_{e0}$', 'TORAX', B['core_stats']['Te0_torax']), ('$T_{i0}$', 'SINTEZ + SHRX', B['core_stats']['Ti0_shrx']), ('$T_{i0}$', 'TORAX', B['core_stats']['Ti0_torax']))) }}
:::

:::fig core figures/core.png
«Тек ядро» режимі: орталық температуралардың болжамы мен өлшемі: (а, б) $T_{e0}$, (в, г) $T_{i0}$; сол жақта SINTEZ + SHRX, оң жақта TORAX. Тұтас сызық — $y=x$, үзік — $\pm30$ %. Деректер қорындағы 0,2 кэВ-тан төмен мәндер өлшем емес (толтырғыш), сондықтан алынбаған.
:::
