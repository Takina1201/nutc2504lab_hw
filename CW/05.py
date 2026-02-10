"""
CW-05：純文字 PDF 提取，轉成 Markdown
======================================
使用三種工具：pdfplumber、Docling、Markitdown
輸入：example.pdf
輸出：output_pdfplumber.md / output_docling.md / output_markitdown.md

安裝：pip install pdfplumber docling markitdown
"""

import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PDF = os.path.join(SCRIPT_DIR, "example.pdf")


# ============================================================
# 工具一：pdfplumber
# ============================================================
def run_pdfplumber(pdf_path: str) -> str:
    import pdfplumber

    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        print(f"  📄 PDF 共 {len(pdf.pages)} 頁")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                all_text.append(text)

    raw = "\n\n".join(all_text)

    # 簡易 Markdown 格式化
    lines = raw.split("\n")
    md_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            md_lines.append("")
        elif "畢業資格審查作業要點" in s and len(s) < 50:
            md_lines.append(f"# {s}")
        elif "各類專業技術證照表" in s:
            md_lines.append(f"## {s}")
        elif re.match(r'^[一二三四五六七八九十]+、', s):
            md_lines.append(f"\n### {s}")
        elif re.match(r'^\([一二三四五六七八九十]+\)', s):
            md_lines.append(f"\n**{s}**")
        elif re.match(r'^\d+\.', s):
            md_lines.append(f"- {s}")
        elif re.match(r'^\d{2,3}/\d{2}/\d{2}', s):
            md_lines.append(f"- {s}")
        else:
            md_lines.append(s)

    return "\n".join(md_lines)


# ============================================================
# 工具二：Docling
# ============================================================
def run_docling(pdf_path: str) -> str:
    from docling.document_converter import DocumentConverter

    print(f"  🔄 Docling 處理中...")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


# ============================================================
# 工具三：Markitdown
# ============================================================
def run_markitdown(pdf_path: str) -> str:
    from markitdown import MarkItDown

    print(f"  🔄 Markitdown 處理中...")
    md = MarkItDown()
    result = md.convert(pdf_path)
    return result.text_content


# ============================================================
# 主程式
# ============================================================
def main():
    print("=" * 60)
    print("CW-05：純文字 PDF → Markdown（三種工具）")
    print("=" * 60)

    if not os.path.exists(INPUT_PDF):
        print(f"❌ 找不到 {INPUT_PDF}")
        return

    tools = {
        "pdfplumber":  run_pdfplumber,
        "docling":     run_docling,
        "markitdown":  run_markitdown,
    }

    for name, func in tools.items():
        out_path = os.path.join(SCRIPT_DIR, f"output_{name}.md")
        print(f"\n{'─'*40}")
        print(f"📌 {name}")
        try:
            md = func(INPUT_PDF)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"  ✅ {len(md)} 字元 → {out_path}")
        except ImportError as e:
            print(f"  ⚠️  未安裝：{e}")
        except Exception as e:
            print(f"  ❌ 錯誤：{e}")

    print(f"\n{'='*60}")
    print("完成！")


if __name__ == "__main__":
    main()