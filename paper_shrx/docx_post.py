"""Page layout, fonts and table rules of the Word version (python-docx on the file pandoc wrote)."""
import re

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"
MONO = "Courier New"


def _strip_theme(rfonts):
    for a in list(rfonts.attrib):
        if a.endswith("Theme"):
            del rfonts.attrib[a]


def _set_font(style, name=FONT, size=None, bold=None, italic=None, color=True):
    f = style.font
    f.name = name
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    _strip_theme(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), name)
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color:
        f.color.rgb = RGBColor(0, 0, 0)
        c = rpr.find(qn("w:color"))
        for a in list(c.attrib):
            if a.endswith("Theme") or a.endswith("themeColor") or a.endswith("themeShade") or a.endswith("themeTint"):
                del c.attrib[a]


def _style(doc, name):
    """Style by name (python-docx maps 'Heading 1' to the internal 'heading 1', which pandoc's file does not use)."""
    for st in doc.styles:
        if st.name == name:
            return st
    return None


def _para(style, align=None, before=None, after=None, line=None, keep_next=None, keep_lines=None, indent=None):
    pf = style.paragraph_format
    if align is not None:
        pf.alignment = align
    if before is not None:
        pf.space_before = Pt(before)
    if after is not None:
        pf.space_after = Pt(after)
    if line is not None:
        pf.line_spacing = line
    if keep_next is not None:
        pf.keep_with_next = keep_next
    if keep_lines is not None:
        pf.keep_together = keep_lines
    if indent is not None:
        pf.first_line_indent = indent


def _doc_defaults(doc):
    styles = doc.styles.element
    dd = styles.find(qn("w:docDefaults"))
    if dd is None:
        return
    rpr = dd.find(qn("w:rPrDefault") + "/" + qn("w:rPr"))
    if rpr is None:
        return
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    _strip_theme(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONT)
    lang = rpr.find(qn("w:lang"))
    if lang is None:
        lang = OxmlElement("w:lang")
        rpr.append(lang)
    lang.set(qn("w:val"), "kk-KZ")
    lang.set(qn("w:eastAsia"), "kk-KZ")
    lang.set(qn("w:bidi"), "ar-SA")


def styles(doc):
    _doc_defaults(doc)
    for n in ("Normal", "Body Text", "First Paragraph"):
        s = _style(doc, n)
        if s:
            _set_font(s, size=11)
            _para(s, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=4, line=1.12)
    s = _style(doc, "Compact")
    if s:
        _set_font(s, size=9)
        _para(s, align=WD_ALIGN_PARAGRAPH.LEFT, before=1, after=1, line=1.0)
    for n, size, ital in (("Heading 1", 14, False), ("Heading 2", 12, False), ("Heading 3", 11, True), ("Heading 4", 11, True)):
        s = _style(doc, n)
        if s:
            _set_font(s, size=size, bold=True, italic=ital)
            _para(s, align=WD_ALIGN_PARAGRAPH.LEFT, before=12 if n == "Heading 1" else 9, after=4, keep_next=True, keep_lines=True)
    for n, size, bold, ital, after in (("Title", 15, True, False, 6), ("Subtitle", 11, False, True, 8), ("Authors", 11, True, False, 2),
                                       ("Affiliation", 10, False, True, 6)):
        s = _style(doc, n)
        if s:
            _set_font(s, size=size, bold=bold, italic=ital)
            _para(s, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=after, line=1.05)
    s = _style(doc, "UDK")
    if s:
        _set_font(s, size=10)
        _para(s, align=WD_ALIGN_PARAGRAPH.LEFT, before=4, after=6)
    for n in ("Image Caption", "Table Caption", "Caption"):
        s = _style(doc, n)
        if s:
            _set_font(s, size=9.5, bold=False, italic=False)
            _para(s, align=WD_ALIGN_PARAGRAPH.JUSTIFY if n != "Table Caption" else WD_ALIGN_PARAGRAPH.LEFT, before=3 if n != "Table Caption" else 8,
                  after=8 if n != "Table Caption" else 3, line=1.0, keep_next=(n == "Table Caption"), keep_lines=True)
    for n in ("Figure", "Captioned Figure"):
        s = _style(doc, n)
        if s:
            _para(s, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=0, keep_next=True)
    for n in ("Source Code", "Verbatim Char"):
        s = _style(doc, n)
        if s:
            _set_font(s, name=MONO, size=8.5 if n == "Source Code" else 9.5)
    s = _style(doc, "Source Code")
    if s:
        _para(s, align=WD_ALIGN_PARAGRAPH.LEFT, before=2, after=2, line=1.0)
    for n in ("Footnote Text", "Block Text", "Definition", "Definition Term"):
        s = _style(doc, n)
        if s:
            _set_font(s, size=10)
    # numbered list of references: tighter
    s = _style(doc, "Compact")


def page(doc):
    for sec in doc.sections:
        sec.orientation = WD_ORIENT.PORTRAIT
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.0)
        sec.top_margin = sec.bottom_margin = Cm(2.0)
        sec.footer_distance = Cm(1.0)
        foot = sec.footer
        foot.is_linked_to_previous = False
        p = foot.paragraphs[0] if foot.paragraphs else foot.add_paragraph()
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        for typ, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if typ:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), typ)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)
        run.font.size = Pt(10)
        run.font.name = FONT


def _borders(el_parent, tag, edges):
    b = el_parent.find(qn(tag))
    if b is not None:
        el_parent.remove(b)
    b = OxmlElement(tag)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        if edge in edges:
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(edges[edge]))
            e.set(qn("w:space"), "0")
            e.set(qn("w:color"), "000000")
        else:
            e.set(qn("w:val"), "nil")
        b.append(e)
    el_parent.append(b)


def _is_eq_table(t):
    if len(t.columns) != 3 or len(t.rows) != 1:
        return False
    return re.fullmatch(r"\(\d+\)", t.rows[0].cells[2].text.strip()) is not None and not t.rows[0].cells[0].text.strip()


def tables(doc):
    for t in doc.tables:
        tblPr = t._tbl.tblPr
        for st in tblPr.findall(qn("w:tblStyle")):
            tblPr.remove(st)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        if _is_eq_table(t):
            _borders(tblPr, "w:tblBorders", {})
            for row in t.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.space_before = Pt(2)
                        p.paragraph_format.space_after = Pt(2)
                        if cell is row.cells[2]:
                            for r in p.runs:
                                r.font.size = Pt(11)
            continue
        _borders(tblPr, "w:tblBorders", {"top": 8, "bottom": 8})
        ncol = len(t.columns)
        size = 9 if ncol <= 6 else (8 if ncol <= 8 else 7.5)
        for i, row in enumerate(t.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(size)
                        r.font.name = FONT
                        if i == 0:
                            r.font.bold = True
                if i == 0:
                    tcPr = cell._tc.get_or_add_tcPr()
                    for b in tcPr.findall(qn("w:tcBorders")):
                        tcPr.remove(b)
                    tb = OxmlElement("w:tcBorders")
                    e = OxmlElement("w:bottom")
                    e.set(qn("w:val"), "single")
                    e.set(qn("w:sz"), "4")
                    e.set(qn("w:space"), "0")
                    e.set(qn("w:color"), "000000")
                    tb.append(e)
                    tcPr.append(tb)
            if i == 0:
                trPr = row._tr.get_or_add_trPr()
                h = OxmlElement("w:tblHeader")
                trPr.append(h)
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            trPr.append(cs)
        # cell margins
        mar = tblPr.find(qn("w:tblCellMar"))
        if mar is not None:
            tblPr.remove(mar)
        mar = OxmlElement("w:tblCellMar")
        for side, w in (("top", 20), ("left", 60), ("bottom", 20), ("right", 60)):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:w"), str(w))
            e.set(qn("w:type"), "dxa")
            mar.append(e)
        tblPr.append(mar)


def keep_figure_with_caption(doc):
    """A figure paragraph (a picture and nothing else) stays on the page of the caption that follows it."""
    for p in doc.paragraphs:
        if p._p.findall(".//" + qn("w:drawing")) and not p.text.strip():
            p.paragraph_format.keep_with_next = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def finish(src, dst):
    doc = Document(str(src))
    styles(doc)
    page(doc)
    tables(doc)
    keep_figure_with_caption(doc)
    cp = doc.core_properties
    cp.title = "SHRX: токамак плазмасының турбуленттік тасымалының нейрожелілік моделі және оны TORAX-пен салыстыру"
    cp.language = "kk-KZ"
    cp.author = ""
    doc.save(str(dst))
