import re
from pathlib import Path
from docx import Document

doc = Document(Path(__file__).parent / "БояриновИР_Дипломная v1.5.docx")
out = []
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if not t:
        continue
    st = p.style.name if p.style else ""
    if st in ("Title", "Heading 1", "Heading 2", "Heading 3") or re.match(r"^[1-4]\.", t):
        out.append(f"{i} [{st}] {t[:110]}")
Path(__file__).parent.joinpath("_v15_outline.txt").write_text("\n".join(out), encoding="utf-8")
print(len(out))
