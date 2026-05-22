import re
from pathlib import Path
from docx import Document

doc = Document(Path(__file__).parent / "БояриновИР_Дипломная v1.5.docx")
text = "\n".join(p.text for p in doc.paragraphs)
refs = sorted(set(int(x) for x in re.findall(r"рис\.\s*(\d+)", text, re.I)))
print("count", len(refs))
print(refs)
missing = [i for i in range(1, 47) if i not in refs]
print("missing 1-46:", missing)
