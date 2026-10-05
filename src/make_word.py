"""把 results/圖表/ 的 24 張圖與 results/圖表說明.md 的文字，排成「一頁一張圖配說明」的 Word 檔，再用 Word 轉成 PDF。

版面：A4 橫向；第 1 頁是封面與目錄，之後每張圖從新的一頁開始：圖號與標題 → 滿版的圖 → 說明。
執行：python src/make_word.py（需先跑過 figure_book.py；轉 PDF 需要電腦裝有 Microsoft Word）
輸出：results/CO2研究圖表與說明.docx、results/CO2研究圖表與說明.pdf
"""
from __future__ import annotations

import re
import sys

import numpy as np

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

import data as D

R = D.ROOT / "results"
FIG_DIR = R / "圖表"
MD = R / "圖表說明.md"
OUT = R / "CO2研究圖表與說明.docx"

FONT_ZH = "標楷體"            # 中文
FONT_EN = "Times New Roman"   # 英文與數字
INK = RGBColor(0x17, 0x21, 0x2B)
MUTED = RGBColor(0x5B, 0x66, 0x70)
ACCENT = RGBColor(0x0D, 0x74, 0x68)

PAGE_W, PAGE_H, MARGIN = 29.7, 21.0, 1.5      # A4 橫向（公分）
MAX_IMG_W = PAGE_W - 2 * MARGIN
MAX_IMG_H = 11.5                                # 留空間給下方說明
LINE_PT = 15                                    # 說明文字的固定行高


# ---------------------------------------------------------------- 解析說明檔
def parse_md(text: str) -> tuple[list[str], list[dict]]:
    """回傳 (前言段落, 圖的清單)。每張圖：section, num, title, file, items[(label, text, [子項目])]。"""
    intro, figs, section = [], [], ""
    cur = None
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("### 圖"):
            m = re.match(r"### 圖 (\d+)\s+(.*)", line)
            cur = {"section": section, "num": m.group(1), "title": m.group(2).strip(), "file": None, "items": []}
            figs.append(cur)
        elif cur is None:
            if line.strip() and not line.startswith("#") and line.strip() != "---":
                intro.append(line.strip())
        elif line.startswith("**檔名**"):
            cur["file"] = re.search(r"`(.+?)`", line).group(1)
        elif line.startswith("- "):
            m = re.match(r"- \*\*(.+?)\*\*：?(.*)", line)
            label, body = (m.group(1), m.group(2).strip()) if m else ("", line[2:].strip())
            cur["items"].append([label, body, []])
        elif re.match(r"\s{2,}- ", line) and cur["items"]:
            cur["items"][-1][2].append(line.strip()[2:])
        elif line.startswith("  ") and line.strip() and cur["items"]:
            # 子清單後的續行（例如判斷規則的補充句）
            cur["items"][-1][2].append("　" + line.strip())
    return intro, figs


# ---------------------------------------------------------------- 排版工具
def _apply_fonts(rpr_fonts):
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rpr_fonts.set(qn(attr), FONT_EN)
    rpr_fonts.set(qn("w:eastAsia"), FONT_ZH)


def set_font(run, size=None, bold=None, color=None):
    run.font.name = FONT_EN
    _apply_fonts(run._element.rPr.rFonts)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def add_rich(par, text, size=11, color=INK, bold_all=False):
    """把 **粗體** 與 `程式字` 轉成 Word 格式。"""
    for part in re.split(r"(\*\*.+?\*\*)", text):
        if not part:
            continue
        bold = bold_all or part.startswith("**")
        part = part.strip("*").replace("`", "")
        set_font(par.add_run(part), size=size, bold=bold, color=color)


def spacing(par, before=0, after=4, line=1.25, exact=None):
    pf = par.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = Pt(exact) if exact else line   # 固定行高（pt）才能準確估算版面


def setup(doc: Document):
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Cm(PAGE_W), Cm(PAGE_H)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(MARGIN))
    normal = doc.styles["Normal"]
    normal.font.name = FONT_EN
    normal.font.size = Pt(11)
    _apply_fonts(normal.element.rPr.rFonts)
    for name in ("Heading 1", "Title"):
        st = doc.styles[name]
        st.font.name = FONT_EN
        _apply_fonts(st.element.rPr.rFonts)
    # 頁尾頁碼
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = fp.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = run._element.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): tag})
        else:
            el = run._element.makeelement(qn("w:instrText"), {qn("xml:space"): "preserve"})
            el.text = txt
        run._element.append(el)
    set_font(run, size=9, color=MUTED)


def _lines(text: str, cpl: float) -> int:
    return max(1, int(np.ceil(len(text) / cpl)))


def estimate_height(items, chars_per_line: float) -> float:
    """說明文字需要的高度（公分）：固定行高 LINE_PT，加上段後間距。"""
    n, paras = 0, 0
    for label, body, subs in items:
        n += _lines(label + body, chars_per_line)
        n += sum(_lines(x, chars_per_line - 3) for x in subs)
        paras += 1 + len(subs)
    return n * LINE_PT * 0.03528 + paras * 2 * 0.03528 + 0.3


def add_items(container, items, first=False, chars_per_line=None):
    for label, body, subs in items:
        p = container.paragraphs[0] if first else container.add_paragraph()
        first = False
        spacing(p, after=2, exact=LINE_PT)
        if label:
            add_rich(p, f"{label}：", size=10.5, bold_all=True, color=ACCENT)
        add_rich(p, body, size=10.5)
        for x in subs:
            q = container.add_paragraph()
            spacing(q, after=1, exact=LINE_PT)
            q.paragraph_format.left_indent = Cm(0.9)
            q.paragraph_format.first_line_indent = Cm(-0.4)
            add_rich(q, x if x.startswith("　") else "・" + x, size=10)


def keep_full_resolution(doc: Document) -> None:
    """等同 Word 選項「不要壓縮檔案中的影像」＋「預設解析度：高傳真」，轉 PDF 時保留 300 dpi 原圖。

    兩個設定都必須放在 settings.xml 規定的位置，放錯位置 Word 會直接忽略。
    """
    st = doc.settings.element
    w14 = "http://schemas.microsoft.com/office/word/2010/wordml"
    if st.find(qn("w:doNotAutoCompressPictures")) is None:
        el = st.makeelement(qn("w:doNotAutoCompressPictures"), {})
        anchor = next((st.find(qn(t)) for t in ("w:forceUpgrade", "w:captions", "w:readModeInkLockDown",
                                                  "w:smartTagType", "w:shapeDefaults", "w:doNotEmbedSmartTags",
                                                  "w:decimalSymbol", "w:listSeparator")
                       if st.find(qn(t)) is not None), None)
        anchor.addprevious(el) if anchor is not None else st.append(el)
    if st.find(f"{{{w14}}}defaultImageDpi") is None:
        el = st.makeelement(f"{{{w14}}}defaultImageDpi", {f"{{{w14}}}val": "32767"})   # 32767 = 高傳真
        lst = st.find(qn("w:listSeparator"))
        lst.addnext(el) if lst is not None else st.append(el)


def image_size(path) -> tuple[float, float]:
    w, h = Image.open(path).size
    width = MAX_IMG_W
    height = width * h / w
    if height > MAX_IMG_H:
        height = MAX_IMG_H
        width = height * w / h
    return width, height


# ---------------------------------------------------------------- 主流程
def build() -> None:
    intro, figs = parse_md(MD.read_text(encoding="utf-8"))
    doc = Document()
    setup(doc)

    # 封面與目錄
    p = doc.add_paragraph()
    spacing(p, before=10, after=4)
    add_rich(p, "高中教室 CO₂ 對學生認知表現影響之預測模型", size=24, bold_all=True)
    p = doc.add_paragraph()
    spacing(p, after=10)
    add_rich(p, "圖表與說明（給指導教授）", size=15, color=MUTED)
    for line in intro:
        if line.startswith("圖檔位置"):
            continue  # 圖已經放在這份文件裡，不需要檔案路徑
        p = doc.add_paragraph()
        spacing(p, after=2)
        add_rich(p, line, size=11)
    p = doc.add_paragraph()
    spacing(p, before=8, after=2)
    add_rich(p, "目錄", size=13, bold_all=True, color=ACCENT)
    # 目錄分兩欄：用一個無框表格
    sections = list(dict.fromkeys(f["section"] for f in figs))
    sizes = [1 + sum(f["section"] == x for f in figs) for x in sections]   # 每節佔的行數
    cut = min(range(1, len(sections)), key=lambda k: abs(sum(sizes[:k]) - sum(sizes[k:])))
    table = doc.add_table(rows=1, cols=2)
    for col, secs in zip(table.rows[0].cells, (sections[:cut], sections[cut:])):
        col.paragraphs[0].text = ""
        first = True
        for s in secs:
            par = col.paragraphs[0] if first else col.add_paragraph()
            first = False
            spacing(par, before=5, after=0, exact=15)
            add_rich(par, s, size=10.5, bold_all=True, color=MUTED)
            for f in (x for x in figs if x["section"] == s):
                par = col.add_paragraph()
                spacing(par, after=0, exact=15)
                par.paragraph_format.left_indent = Cm(0.5)
                add_rich(par, f"圖 {f['num']}　{f['title']}", size=10.5)

    # 每張圖一頁
    for f in figs:
        p = doc.add_paragraph()
        p.paragraph_format.page_break_before = True       # 從新的一頁開始，不留空白頁
        spacing(p, after=0)
        add_rich(p, f["section"], size=9.5, color=MUTED)
        h = doc.add_heading(level=1)
        spacing(h, before=0, after=4)
        add_rich(h, f"圖 {f['num']}　{f['title']}", size=16, bold_all=True, color=INK)
        img = FIG_DIR / f["file"]
        iw, ih = Image.open(img).size
        aspect = iw / ih
        if aspect < 1.8:
            # 偏方形：左圖右文
            t = doc.add_table(rows=1, cols=2)
            t.autofit = False
            left_w = min(16.5, 15.5 * aspect)
            right_w = MAX_IMG_W - left_w - 0.4
            cl, cr = t.rows[0].cells
            cl.width, cr.width = Cm(left_w), Cm(right_w)
            pl = cl.paragraphs[0]
            spacing(pl, after=0, line=1.0)
            pl.add_run().add_picture(str(img), width=Cm(left_w - 0.2))
            add_items(cr, f["items"], first=True, chars_per_line=right_w / 0.37)
        else:
            text_h = estimate_height(f["items"], MAX_IMG_W / 0.37)
            avail = PAGE_H - 2 * MARGIN - 2.3 - text_h
            w = MAX_IMG_W
            hgt = w / aspect
            if hgt > avail:
                hgt = max(avail, 6.5)
                w = hgt * aspect
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            spacing(p, after=6, line=1.0)
            p.add_run().add_picture(str(img), width=Cm(w), height=Cm(hgt))
            add_items(doc, f["items"])
    keep_full_resolution(doc)
    doc.save(OUT)
    print(OUT, f"{len(figs)} 張圖")


def to_pdf() -> int:
    """用電腦上的 Microsoft Word 轉 PDF（版面與在 Word 中看到的一致），回傳頁數。"""
    import subprocess
    pdf = OUT.with_suffix(".pdf")
    ps = (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0; "
        f"try {{ $d = $w.Documents.Open('{OUT}', $false, $true); "
        f"$d.ExportAsFixedFormat('{pdf}', 17, $false, 0); "      # 17 = PDF；0 = 列印品質，不壓縮圖片
        "$d.ComputeStatistics(2); $d.Close($false) } finally { $w.Quit() }"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, check=True)
    pages = int(out.stdout.strip().splitlines()[-1])
    restore_full_resolution(pdf)
    print(pdf, f"{pages} 頁")
    return pages


def restore_full_resolution(pdf) -> None:
    """Word 匯出 PDF 時會把圖重新取樣到約 150 dpi；這裡把每頁的圖原位置換回 300 dpi 原圖，版面不變。"""
    import shutil
    import tempfile

    import pymupdf
    _, figs = parse_md(MD.read_text(encoding="utf-8"))
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    doc = pymupdf.open(pdf)
    for i, f in enumerate(figs):
        page = doc[i + 1]                         # 第 1 頁是封面
        imgs = page.get_images(full=True)
        assert len(imgs) == 1, f"第 {i + 2} 頁應該只有一張圖"
        page.replace_image(imgs[0][0], filename=str(FIG_DIR / f["file"]))
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    shutil.move(tmp, pdf)


if __name__ == "__main__":
    build()
    if "--no-pdf" not in sys.argv:
        to_pdf()
