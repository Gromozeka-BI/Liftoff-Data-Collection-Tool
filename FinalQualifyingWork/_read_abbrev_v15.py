#!/usr/bin/env python3
"""Извлечь перечень сокращений из v1.5."""
from pathlib import Path
from docx import Document

import sys
name = sys.argv[1] if len(sys.argv) > 1 else "БояриновИР_Дипломная v1.5.docx"
doc = Document(Path(__file__).parent / name)
lines = []
capture = False
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if "ПЕРЕЧЕНЬ СОКРАЩЕНИЙ" in t.upper():
        capture = True
        lines.append(f"--- START p{i} ---")
        lines.append(t)
        continue
    if capture:
        if t.upper().startswith("ВВЕДЕНИЕ") or (p.style and p.style.name == "Heading 1" and "ВВЕДЕНИЕ" in t.upper()):
            lines.append(f"--- END before p{i}: {t} ---")
            break
        if t:
            lines.append(f"p{i}: {t}")

out = Path(__file__).parent / ("_abbrev_v16.txt" if "v1.6" in name else "_abbrev_v15.txt")
out.write_text("\n".join(lines), encoding="utf-8")
print(out.read_text(encoding="utf-8"))
