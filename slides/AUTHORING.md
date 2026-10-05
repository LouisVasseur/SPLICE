# Optional presentation rebuilding

The supplied PPTX and PDF are ready to open. Editing or presenting them does not
require Node, Python plotting packages or LibreOffice.

To rebuild from the saved data, the optional authoring dependencies are:
Python + Matplotlib; Node.js + pptxgenjs; LibreOffice only for automatic PDF export.

```bash
python3 slides/build_figures.py
node slides/build_deck.js
```

The source reads results/saved, not your machine's results/local. This avoids
silently changing source figures during a meeting. To publish updated benchmark
figures, deliberately archive a new verified run and update the source data.

notes_text.json contains editable speaker notes. slide_index.json maps slides to
lesson IDs and source references. SPEAKER_NOTES.md is a readable export.
