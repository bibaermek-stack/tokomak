"""Kazakh -> English labels for the figures.  Used by figstyle.py when FIG_LANG=en (the Kazakh figures are the default and unchanged).

Every text a figure draws goes through tr() (figstyle patches matplotlib's Text.set_text), so the figure code itself has no English strings.
Substring replacement, longest phrase first, then the unit rules; decimal commas become decimal points.
"""
import re

PHRASES = {
    # pipeline
    "Жылу тасымалы теңдеуі": "Heat transport equation",
    "Жаттығу (желіден тыс)": "Training (outside the network)",
    "QLKNN-мұғалім": "QLKNN teacher",
    "белгі $y$": "label $y$",
    "-шығын, он-полиси қайта белгілеу": " loss, on-policy relabelling",
    "күй жаңарады (әр уақыт қадамында)": "state update (every time step)",
    "салмақтар": "weights",
    "Көпір": "Bridge",
    "Ағын": "Flux",
    "күйі": "state",
    "желісі": "network",
    "29 283": "29,283",
    # local accuracy, parity
    "оқудан тыс машиналар": "machines held out of training",
    "оқудан тыс машина": "machine held out of training",
    "оқу машиналары": "training machines",
    "гиро-Бом бірлігі": "gyro-Bohm units",
    "көпір арқылы": "through the bridge",
    "өз есебі": "own calculation",
    # ITER traces and profiles
    "IPB98-ге бекітілген": "anchored to IPB98",
    "жобасы": "design",
    "модельдеу уақыты": "simulation time",
    "жазық қиыр": "flat top",
    # latency
    "толық шақыру": "full call",
    "бір шақыру уақыты": "time per call",
    "түйін": "nodes",
    "мкс": "µs",
    # ensemble and threshold
    "жалғыз желі": "single network",
    "ансамбль": "ensemble",
    "(жинақталған)": "(cumulative)",
    "жазбалар үлесі": "fraction of records",
    "нүктелер үлесі": "fraction of points",
    "күйлер үлесі": "share of states",
    "үлесі": "fraction",
    "алшақтық децилі ($s$ бойынша)": "spread decile (by $s$)",
    "қатенің медианасы": "median error",
    "|ε| медианасы": "|ε| median",
    "мүшелер арасындағы алшақтық": "spread between members",
    "салыстырмалы қате": "relative error",
    # in-loop
    "жеткіз.": "delivered",
    "жеткізілетін желі": "delivered network",
    "алты желі мүшесі": "five network members",   # the data have five members per machine (the Kazakh label said six)
    "бастапқы": "initial",
    "өзгертілген": "perturbed",
    "№": "#",
    "машиналық эпсилон": "machine epsilon",
    # database
    "жарамсыз жүгіру": "runs failing validation",
    "барлық жүгіру": "all runs",
    "жарамды жүгіру": "validated runs",
    "регрессиясы": "regression",
    # spherical tokamaks
    "оқу ауқымы": "training range",
    "$c_s/a$ бірлігінде": "in units of $c_s/a$",
    "соқтығысу жиілігі": "collision frequency",
    "ауқымнан тыс нүктелер": "points outside the range",
    "(сценарий)": "(scripted)",
    "(жазбалар)": "(records)",
    "сценарий ": "scripted ",
    "жазбалар ": "records ",
    "КТМ": "KTM",
    # speed-up
    "жылдамдатуға дейін": "before the speed-up",
    "жылдамдатудан кейін": "after the speed-up",
    "жүгіру уақыты": "run time",
    "(120 с плазма)": "(120 s of plasma)",
    # units
    "м$^2$/с": "m$^2$/s",
    "МВт": "MW",
    "кэВ": "keV",
    "МДж": "MJ",
}
_ORDERED = sorted(PHRASES.items(), key=lambda kv: -len(kv[0]))
_CYR = re.compile("[Ѐ-ӿ]")
PANEL = str.maketrans("абвгде", "abcdef")


def tr(s):
    for k, v in _ORDERED:
        s = s.replace(k, v)
    s = re.sub(r", с(?=$| \()", ", s", s)      # "$t$, с" -> "$t$, s"
    s = s.replace("{,}", ".")
    return re.sub(r"(?<=\d),(?=\d)", ".", s)


def leftover(s):
    """Cyrillic letters that tr() did not translate (the build refuses to save such a figure)."""
    return "".join(_CYR.findall(s))
