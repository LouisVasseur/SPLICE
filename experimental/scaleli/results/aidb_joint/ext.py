import fitz, sys
src, out = sys.argv[1], sys.argv[2]
d = fitz.open(src)
t = '\n'.join(p.get_text() for p in d)
open(out,'w').write(t)
print(out, len(t), d.page_count)
