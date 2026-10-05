#!/usr/bin/env python3
"""Assemble results/aidb/READING_GUIDE.md from the deep section files (stdlib only).
usage: assemble_guide.py [--qa path-to-previous-guide-for-section-8]"""
import pathlib, re, sys, datetime
HERE = pathlib.Path(__file__).resolve().parent; AIDB = HERE.parent
parts = sorted(p for p in HERE.glob("0[0-7]_*.md"))
qa_src = AIDB / "READING_GUIDE_SHORT.md"
out = [f"# Reading guide (deep): NFL, CSV, the AIDB hardness protocol, our implementations, the exact protocol and the results\n",
       f"Assembled {datetime.date.today().isoformat()} from the section files in results/aidb/guide_deep/ (each section was written from the papers, the code and the result files, then checked by a source-fidelity verifier and a student critic, corrected, and cross-checked for consistency). The previous short guide is results/aidb/READING_GUIDE_SHORT.md; the verified meeting notes are results/aidb/MEETING_NOTES.md. Paths are relative to experimental/scaleli/ unless absolute.\n",
       "## Contents\n"]
toc = []
bodies = []
for i, p in enumerate(parts):
    text = p.read_text().strip("\n")
    m = re.match(r"#\s+(.*)", text)
    title = m.group(1).strip() if m else p.stem
    anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    toc.append(f"{i}. [{title}](#{anchor})")
    if m: text = text[m.end():].lstrip("\n")
    # demote headings by one level so the section title is ##
    text = re.sub(r"^(#{1,5})\s", lambda mm: "#" * (len(mm.group(1)) + 1) + " ", text, flags=re.M)
    bodies.append(f"\n---\n\n## {title}\n\n{text}\n")
if qa_src.exists():
    sq = qa_src.read_text(); mq = re.search(r"^## 8\. Caveats and likely questions\n(.*)", sq, re.S | re.M)
    if mq:
        toc.append(f"{len(parts)}. [Caveats and likely questions (from the short guide)](#caveats-and-likely-questions-from-the-short-guide)")
        bodies.append("\n---\n\n## Caveats and likely questions (from the short guide)\n\n" + mq.group(1).strip() + "\n")
out += [ "\n".join(toc) + "\n"] + bodies
dst = AIDB / "READING_GUIDE.md"; dst.write_text("\n".join(out)); print(dst, sum(1 for _ in dst.read_text().splitlines()), "lines from", len(parts), "sections")
