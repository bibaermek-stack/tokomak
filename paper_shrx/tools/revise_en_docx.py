#!/usr/bin/env python3
"""Bring the English manuscript .docx closer to the Journal of Fusion Energy article layout.

usage: revise_en_docx.py INPUT.docx OUTPUT.docx [--eqnum native|tabs]

Standard library only.  Edits word/document.xml, word/_rels/document.xml.rels and docProps/core.xml:

  1. display equations: the 24 borderless 3-column "layout" tables (empty | equation | (n)) become one numbered
     display equation each.  --eqnum native  uses Word's own equation numbering (equation array, '#' separator);
     --eqnum tabs    uses a centre tab + right tab in an ordinary paragraph (fallback; renders everywhere).
  2. equations: invisible / exotic spaces (U+200B, U+2001, U+2004, U+2005, U+2009) are removed or turned into plain spaces.
  3. back matter in the order of the journal sample: Appendix, Author Contributions, Data Availability, Code Availability,
     Declarations (Funding, Competing Interests, Ethics), References.  Bold run-in labels, draft markers removed.
  4. references in the sample's style (article titles, bold volume, italic book titles, DOIs as hyperlinks).
  5. (optional, --figures-kk/--figures-en) figures whose labels are in Kazakh are replaced by the English-labelled ones
     built with FIG_LANG=en make_figures.py; the match is by file content, width kept, height follows the new aspect ratio.
  6. keywords separated by a middle dot, ORCID as a hyperlink, alt text on every figure, empty spacer paragraphs and the
     manual page break removed, author name in the document properties.
"""
import argparse
import hashlib
import html
import re
import struct
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

EMPTY_SPACER = '<w:p><w:pPr><w:pStyle w:val="Compact"/></w:pPr></w:p>'
PAGE_BREAK = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
AUTHOR = "Beibarys Ermekuly"


def plain(xml):
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


def esc(s):
    return html.escape(s, quote=False)


# ---------------------------------------------------------------- 1. equations

def equation_paragraph(inner, number, mode):
    if mode == "native":
        num = ('<m:r><m:t>#</m:t></m:r><m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr>'
               '<m:e><m:r><m:t>%s</m:t></m:r></m:e></m:d>' % number)
        return ('<w:p><w:pPr><w:keepLines/><w:spacing w:before="120" w:after="120"/><w:jc w:val="center"/></w:pPr>'
                '<m:oMathPara><m:oMath><m:eqArr><m:eqArrPr><m:maxDist m:val="1"/></m:eqArrPr><m:e>%s%s</m:e></m:eqArr>'
                '</m:oMath></m:oMathPara></w:p>' % (inner, num))
    return ('<w:p><w:pPr><w:keepLines/><w:tabs><w:tab w:val="center" w:pos="4536"/><w:tab w:val="right" w:pos="9072"/>'
            '</w:tabs><w:spacing w:before="120" w:after="120"/></w:pPr><w:r><w:tab/></w:r><m:oMath>%s</m:oMath>'
            '<w:r><w:tab/><w:t>(%s)</w:t></w:r></w:p>' % (inner, number))


def convert_equation_tables(d, mode):
    count = 0

    def repl(m):
        nonlocal count
        table = m.group(2)
        if "<m:oMathPara>" not in table:
            return m.group(0)
        cells = re.findall(r"<w:tc>.*?</w:tc>", table, re.S)
        maths = re.findall(r"<m:oMath>(.*?)</m:oMath>", cells[1], re.S) if len(cells) == 3 else []
        number = re.fullmatch(r"\((\d+)\)", plain(cells[2]).strip()) if len(cells) == 3 else None
        if len(maths) != 1 or not number:
            raise SystemExit("unexpected equation table layout")
        count += 1
        return equation_paragraph(maths[0], number.group(1), mode)

    d = re.sub(r"(%s)?(<w:tbl>.*?</w:tbl>)" % re.escape(EMPTY_SPACER), repl, d, flags=re.S)
    return d, count


# ---------------------------------------------------------------- 2. exotic spaces in equations

def clean_math_text(d):
    def fix(m):
        attrs, s = m.group(1) or "", m.group(2)
        s = s.replace("​", "")
        s = re.sub("[    ]", " ", s)
        if s != s.strip() or "  " in s:
            attrs = ' xml:space="preserve"'
        return "<m:t%s>%s</m:t>" % (attrs, s)

    d = re.sub(r"<m:t( [^>]*)?>([^<]*)</m:t>", fix, d)
    return re.sub(r"<m:r>(?:<m:rPr>.*?</m:rPr>)?<m:t(?: [^>]*)?></m:t></m:r>", "", d, flags=re.S)


# ---------------------------------------------------------------- 3. back matter

def heading_span(d, text):
    for m in re.finditer(r"<w:p>(?:(?!</w:p>).)*?</w:p>", d, re.S):
        p = m.group(0)
        if 'w:val="Heading1"' in p and plain(p).strip().startswith(text):
            return m.start(), m.end()
    raise SystemExit("heading not found: " + text)


def relabel(p, label):
    p = re.sub(r'<w:pStyle w:val="(?:FirstParagraph|BodyText)"/>', '<w:pStyle w:val="BodyText"/>', p, count=1)
    return re.sub(r'<w:r><w:rPr><w:iCs/><w:i/></w:rPr><w:t xml:space="preserve">[^<]*</w:t></w:r>',
                  '<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % label, p, count=1)


def drop_draft_markers(p):
    def f(m):
        t = m.group(1)
        if "Author to confirm" not in t:
            return m.group(0)
        t = re.sub(r"\s*Author to confirm( or replace)?\.", "", t)
        return '<w:r><w:t xml:space="preserve">%s</w:t></w:r>' % t
    return re.sub(r'<w:r><w:rPr><w:rStyle w:val="Placeholder"/><w:highlight w:val="yellow"/></w:rPr>'
                  r'<w:t xml:space="preserve">([^<]*)</w:t></w:r>', f, p)


def reorder_back_matter(d, references_xml):
    d_start, d_end = heading_span(d, "Declarations")
    a_start, _ = heading_span(d, "Appendix A")
    r_start, r_end = heading_span(d, "References")
    if not d_start < a_start < r_start:
        raise SystemExit("unexpected order of Declarations / Appendix / References")
    decl_heading = d[d_start:d_end]
    decl_paras = re.findall(r"<w:p>.*?</w:p>", d[d_end:a_start], re.S)
    by = {}
    for p in decl_paras:
        t = plain(p).strip()
        for key, label in (("Funding.", "Funding"), ("Competing interests.", "Competing Interests"),
                           ("Ethics approval", "Ethics Approval and Consent to Participate"),
                           ("Author contributions.", "Author Contributions"), ("Data availability.", "Data Availability"),
                           ("Code availability.", "Code Availability")):
            if t.startswith(key):
                by[label] = relabel(drop_draft_markers(p), label)
    expected = {"Funding", "Competing Interests", "Ethics Approval and Consent to Participate", "Author Contributions",
                "Data Availability", "Code Availability"}
    if set(by) != expected:
        raise SystemExit("declarations not recognised: " + repr(sorted(set(by) ^ expected)))
    appendix = d[a_start:r_start]
    new = (appendix + by["Author Contributions"] + by["Data Availability"] + by["Code Availability"] + decl_heading +
           by["Funding"] + by["Competing Interests"] + by["Ethics Approval and Consent to Participate"] +
           d[r_start:r_end] + references_xml)
    return d[:d_start] + new + d[d.rfind("<w:sectPr"):]


# ---------------------------------------------------------------- 4. references

def references(rel_ids):
    """Return (xml, [(rId, url)]).  Entries copy the existing text; titles of journal articles come from refs.py."""
    doi = "https://doi.org/"
    R = [
        ("J. Wesson, ", ("i", "Tokamaks"), ", 4th edn. (Oxford University Press, Oxford, 2011)"),
        ("ITER Physics Expert Group on Confinement and Transport, ITER Physics Expert Group on Confinement Modelling and "
         "Database, ITER Physics Basis Editors, ITER Physics Basis. Chapter 2: Plasma confinement and transport. Nucl. Fusion ",
         ("b", "39"), "(12), 2175 (1999). ", ("u", doi + "10.1088/0029-5515/39/12/302")),
        ("C. Bourdelle, J. Citrin, B. Baiocchi et al., Core turbulent transport in tokamak plasmas: bridging theory and "
         "experiment with QuaLiKiz. Plasma Phys. Control. Fusion ", ("b", "58"), "(1), 014036 (2016). ",
         ("u", doi + "10.1088/0741-3335/58/1/014036")),
        ("G.M. Staebler, J.E. Kinsey, R.E. Waltz, A theory-based transport model with comprehensive physics. Phys. Plasmas ",
         ("b", "14"), ", 055909 (2007). ", ("u", doi + "10.1063/1.2436852")),
        ("K.L. van de Plassche, J. Citrin, C. Bourdelle et al., Fast modeling of turbulent transport in fusion plasmas using "
         "neural networks. Phys. Plasmas ", ("b", "27"), "(2), 022310 (2020). ", ("u", doi + "10.1063/1.5134126"),
         "; arXiv:1911.05617"),
        ("O. Meneghini et al., Self-consistent core-pedestal transport simulations with neural network accelerated models. "
         "Nucl. Fusion ", ("b", "57"), ", 086034 (2017). ", ("u", doi + "10.1088/1741-4326/aa7776")),
        ("L. Zanisi, F. Casson, A. Snodin et al., Data-efficient digital twinning strategies and surrogate models of "
         "quasilinear turbulence in JET and STEP, in IAEA Fusion Energy Conference (FEC 2025), Chengdu, China, "
         "13–18 October 2025. ", ("u", "https://conferences.iaea.org/event/392/contributions/36059/")),
        ("J. Citrin, I. Goodfellow, A. Raju et al., TORAX: a fast and differentiable tokamak transport simulator in JAX. "
         "arXiv:2406.06718 (2024)",),
        ("O. Sauter, C. Angioni, Y.R. Lin-Liu, Neoclassical conductivity and bootstrap current formulas for general "
         "axisymmetric equilibria and arbitrary collisionality regime. Phys. Plasmas ", ("b", "6"), ", 2834 (1999). ",
         ("u", doi + "10.1063/1.873240"), "; erratum: Phys. Plasmas ", ("b", "9"), ", 5140 (2002)"),
        ("P.B. Snyder et al., A first-principles predictive model of the pedestal height and width: development, testing "
         "and ITER optimization with the EPED model. Nucl. Fusion ", ("b", "51"), "(10), 103016 (2011). ",
         ("u", doi + "10.1088/0029-5515/51/10/103016")),
        ("J.W. Connor, J.B. Taylor, Scaling laws for plasma confinement. Nucl. Fusion ", ("b", "17"), "(5), 1047 (1977). ",
         ("u", doi + "10.1088/0029-5515/17/5/015")),
        ("D.P. Kingma, J.L. Ba, Adam: a method for stochastic optimization, in International Conference on Learning "
         "Representations (ICLR 2015). arXiv:1412.6980",),
        ("K. He, X. Zhang, S. Ren, J. Sun, Delving deep into rectifiers: surpassing human-level performance on ImageNet "
         "classification, in Proceedings of the IEEE International Conference on Computer Vision (ICCV) (2015), "
         "pp. 1026–1034. ", ("u", doi + "10.1109/ICCV.2015.123")),
        ("G. Hinton, O. Vinyals, J. Dean, Distilling the knowledge in a neural network. arXiv:1503.02531 (2015)",),
        ("S. Ross, G. Gordon, D. Bagnell, A reduction of imitation learning and structured prediction to no-regret online "
         "learning, in Proceedings of the 14th International Conference on Artificial Intelligence and Statistics "
         "(AISTATS), PMLR 15 (2011), pp. 627–635",),
        ("B. Lakshminarayanan, A. Pritzel, C. Blundell, Simple and scalable predictive uncertainty estimation using deep "
         "ensembles, in Advances in Neural Information Processing Systems 30 (NeurIPS 2017) (2017), pp. 6402–6413",),
        ("T. Fawcett, An introduction to ROC analysis. Pattern Recognit. Lett. ", ("b", "27"), "(8), 861–874 (2006). ",
         ("u", doi + "10.1016/j.patrec.2005.10.010")),
        ("E.N. Lorenz, Deterministic nonperiodic flow. J. Atmos. Sci. ", ("b", "20"), ", 130–141 (1963). ",
         ("u", doi + "10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2")),
        ("G. Verdoolaege, S.M. Kaye, C. Angioni et al., The updated ITPA global H-mode confinement database: description "
         "and analysis. Nucl. Fusion ", ("b", "61"), ", 076006 (2021). ", ("u", doi + "10.1088/1741-4326/abdb91"),
         ". The open sample used in this work is distributed with FUSE.jl. ",
         ("u", "https://github.com/ProjectTorreyPines/FUSE.jl"),
         " (file sample/HDB5_compressed.csv, accessed 19 October 2026). The database is also archived at ",
         ("u", doi + "10.11578/1814950")),
        ("UKAEA, tglfnn-ukaea: neural network surrogates of TGLF. ", ("u", "https://github.com/ukaea/tglfnn-ukaea"),
         " (accessed 19 October 2026)"),
        ("C.R. Harris, K.J. Millman, S.J. van der Walt et al., Array programming with NumPy. Nature ", ("b", "585"),
         ", 357–362 (2020). ", ("u", doi + "10.1038/s41586-020-2649-2")),
        ("J. Bradbury, R. Frostig, P. Hawkins et al., JAX: composable transformations of Python+NumPy programs (2018). ",
         ("u", "https://github.com/jax-ml/jax"), " (accessed 19 October 2026)"),
    ]
    out, links = [], []
    ppr = ('<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1004"/></w:numPr><w:pStyle w:val="Compact"/>'
           '<w:jc w:val="left"/></w:pPr>')
    for entry in R:
        runs = []
        for part in entry:
            if isinstance(part, str):
                runs.append('<w:r><w:t xml:space="preserve">%s</w:t></w:r>' % esc(part))
                continue
            kind, text = part
            if kind == "b":
                runs.append('<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(text))
            elif kind == "i":
                runs.append('<w:r><w:rPr><w:i/><w:iCs/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(text))
            else:
                rid = "rIdLink%d" % (len(links) + 1)
                links.append((rid, text))
                runs.append('<w:hyperlink r:id="%s"><w:r><w:rPr><w:rStyle w:val="Hyperlink"/></w:rPr>'
                            '<w:t xml:space="preserve">%s</w:t></w:r></w:hyperlink>' % (rid, esc(text)))
        out.append("<w:p>%s%s</w:p>" % (ppr, "".join(runs)))
    return "".join(out), links


# ---------------------------------------------------------------- 5. small things

def keywords_and_orcid(d, links):
    def kw(m):
        words = [w.strip() for w in plain(m.group(0)).split(":", 1)[1].split(",")]
        return ('<w:p><w:pPr><w:pStyle w:val="KeywordsPara"/></w:pPr><w:r><w:rPr><w:b/><w:bCs/></w:rPr>'
                '<w:t xml:space="preserve">Keywords</w:t></w:r><w:r><w:t xml:space="preserve">  %s</w:t></w:r></w:p>'
                % esc(" · ".join(words)))
    d, n = re.subn(r'<w:p><w:pPr><w:pStyle w:val="KeywordsPara"/></w:pPr>.*?</w:p>', kw, d, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("keywords paragraph not found")

    url = "https://orcid.org/0009-0005-0506-4120"
    old = '<w:r><w:t xml:space="preserve">ORCID: %s</w:t></w:r>' % url
    if old not in d:
        raise SystemExit("ORCID run not found")
    rid = "rIdLink%d" % (len(links) + 1)
    links.append((rid, url))
    d = d.replace(old, '<w:r><w:t xml:space="preserve">ORCID: </w:t></w:r><w:hyperlink r:id="%s"><w:r><w:rPr>'
                       '<w:rStyle w:val="Hyperlink"/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r></w:hyperlink>'
                  % (rid, url))
    return d


def alt_texts(d):
    captions = []
    for m in re.finditer(r"<w:p>(?:(?!</w:p>).)*?</w:p>", d, re.S):
        p = m.group(0)
        if 'w:val="FigCaption"' in p:
            text = plain(re.sub(r"<m:oMath>.*?</m:oMath>", "", p, flags=re.S)).strip()
            text = re.sub(r"^Fig\.\s*\d+\s*", "", text)
            text = re.sub(r"\s+", " ", text)
            if len(text) > 240:
                text = text[:240].rsplit(" ", 1)[0] + " …"
            captions.append(text)
    pics = re.findall(r'<wp:docPr descr="" ', d)
    if len(captions) != len(pics):
        raise SystemExit("figures (%d) and captions (%d) differ" % (len(pics), len(captions)))
    it = iter(captions)
    return re.sub(r'<wp:docPr descr="" ', lambda m: '<wp:docPr descr="%s" ' % html.escape(next(it), quote=True), d)


def swap_figures(d, rels, src, kk_dir, en_dir):
    """Replace figure images that have an English twin.  Returns (document xml, {media path: bytes}, [names])."""
    kk_dir, en_dir = Path(kk_dir), Path(en_dir)
    known = {}
    for f in kk_dir.rglob("*.png"):
        known[hashlib.md5(f.read_bytes()).hexdigest()] = f.relative_to(kk_dir)
    targets = {}
    for tag in re.findall(r"<Relationship [^>]*/>", rels):
        i, t = re.search(r'Id="([^"]+)"', tag), re.search(r'Target="([^"]+)"', tag)
        if i and t:
            targets[i.group(1)] = t.group(1)
    media, done = {}, []
    with zipfile.ZipFile(src) as z:
        def one(m):
            blk = m.group(0)
            target = targets[re.search(r'r:embed="([^"]+)"', blk).group(1)]
            rel = known.get(hashlib.md5(z.read("word/" + target)).hexdigest())
            if rel is None or not (en_dir / rel).exists():
                return blk
            new = (en_dir / rel).read_bytes()
            w, h = struct.unpack(">II", new[16:24])
            cx = int(re.search(r'<wp:extent cx="(\d+)"', blk).group(1))
            cy = round(cx * h / w)
            blk = re.sub(r'(<wp:extent cx="\d+" cy=")\d+(")', r"\g<1>%d\2" % cy, blk)
            blk = re.sub(r'(<a:ext cx="\d+" cy=")\d+(")', r"\g<1>%d\2" % cy, blk)
            media["word/" + target] = new
            done.append(str(rel))
            return blk
        d = re.sub(r"<w:drawing>.*?</w:drawing>", one, d, flags=re.S)
    return d, media, done


def write(src, dst, replace, extra):
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in replace:
                data = replace[item.filename]
                data = data.encode("utf-8") if isinstance(data, str) else data
            zout.writestr(item, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--eqnum", choices=("native", "tabs"), default="native")
    ap.add_argument("--figures-kk", help="folder with the Kazakh-labelled figures (figures/)")
    ap.add_argument("--figures-en", help="folder with the English-labelled twins (figures_en/)")
    a = ap.parse_args()

    with zipfile.ZipFile(a.src) as z:
        d = z.read("word/document.xml").decode("utf-8")
        rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
        core = z.read("docProps/core.xml").decode("utf-8")

    d, n_eq = convert_equation_tables(d, a.eqnum)
    d = clean_math_text(d)
    d = d.replace(PAGE_BREAK, "")
    refs_xml, links = references(None)
    d = reorder_back_matter(d, refs_xml)
    d = keywords_and_orcid(d, links)
    d = alt_texts(d)
    media, swapped = {}, []
    if a.figures_kk and a.figures_en:
        d, media, swapped = swap_figures(d, rels, a.src, a.figures_kk, a.figures_en)

    rels = rels.replace("</Relationships>", "".join(
        '<Relationship Id="%s" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
        'Target="%s" TargetMode="External"/>' % (rid, html.escape(url, quote=True)) for rid, url in links) +
        "</Relationships>")
    if "<dc:creator>" not in core:
        core = core.replace("<dc:language>", "<dc:creator>%s</dc:creator><dc:language>" % AUTHOR)

    for name, xml in (("document.xml", d), ("rels", rels), ("core", core)):
        ET.fromstring(xml.encode("utf-8"))  # well-formedness check
    write(a.src, a.dst, {"word/document.xml": d, "word/_rels/document.xml.rels": rels, "docProps/core.xml": core, **media}, None)
    print("equations converted: %d | hyperlinks: %d | figures swapped: %d | mode: %s -> %s"
          % (n_eq, len(links), len(swapped), a.eqnum, a.dst))


if __name__ == "__main__":
    sys.exit(main())
