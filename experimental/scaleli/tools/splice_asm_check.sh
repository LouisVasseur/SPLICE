#!/usr/bin/env bash
# Fairness check of the SPLICE-H read path: compiles tests/splice_get_asm.cpp as GRE compiles its TU
# (C++17, -O3, -march=goldmont) and fails if splice_get_probe contains a call, an indirect jump that is
# not a local jump table, a division, an int->fp conversion, x87, FMA, BSF/BSR/LZCNT/TZCNT or a string
# store. Every memory-destination store is listed with its base register; a store passes only if its
# base is the stack (%rsp, or %rbp when the function sets up a frame pointer) or a register that holds
# the out parameter (%rdx, third SysV argument) on EVERY path to it (a must-dataflow over the CFG, with
# spills and reloads followed through stack slots).
#   tools/splice_asm_check.sh [CXX]      (CXX defaults to c++; on macOS -arch x86_64 is added)
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
CXX=${1:-${CXX:-c++}}
OUT=$(mktemp -d "${TMPDIR:-/tmp}/splice_asm.XXXXXX")
trap 'rm -rf "$OUT"' EXIT
ARCH=()
[ "$(uname -s)" = Darwin ] && ARCH=(-arch x86_64)
SRC=${SPLICE_ASM_SRC:-$HERE/tests/splice_get_asm.cpp}   # override only to self-test the checker
"$CXX" "${ARCH[@]}" -std=c++17 -O3 -march=goldmont -fno-asynchronous-unwind-tables -S -I "$HERE/include" \
  "$SRC" -o "$OUT/get.s"
echo "compiler: $CXX ($(${CXX} --version | head -1))"
status=0
python3 - "$OUT/get.s" <<'PY' || status=$?
import re, sys
lines = [l.rstrip("\n") for l in open(sys.argv[1])]
LABEL = re.compile(r"^([._A-Za-z$][\w.$]*):")
# ---- the function body, labels kept (they are the CFG's block heads)
body, on = [], False
for l in lines:
    m = LABEL.match(l)
    if not on:
        if m and m.group(1) in ("splice_get_probe", "_splice_get_probe"): on = True
        continue
    if m and not re.match(r"^\.?L", m.group(1)): break
    s = l.split("#")[0].strip()
    if re.search(r"\.cfi_endproc|^\.size", s): break
    if m: body.append(("label", m.group(1)))
    elif s and not s.startswith("."): body.append(("ins", s))
if not body: print("splice_asm_check: splice_get_probe not found"); sys.exit(1)
# ---- jump tables anywhere in the file: a label followed by .long/.quad entries naming local labels
alias = {}
for l in lines:
    m = re.match(r"^\s*\.set\s+([\w.$]+),\s*([\w.$]+)", l)
    if m: alias[m.group(1)] = m.group(2)
tables, cur = {}, None
for l in lines:
    m = LABEL.match(l)
    if m: cur = m.group(1); continue
    m = re.match(r"^\s*\.(long|quad)\s+([\w.$]+)", l)
    if m and cur:
        t = alias.get(m.group(2), m.group(2))
        tables.setdefault(cur, []).append(t)
    elif l.strip() and not l.strip().startswith((".p2align", ".align", ".data_region", ".end_data_region", "#", "##")):
        cur = None
PREFIX = ("rep", "repz", "repe", "repnz", "repne", "lock", "notrack", "bnd")
def split(s):
    parts = s.split(None, 1)
    while parts and parts[0] in PREFIX and len(parts) > 1: parts = parts[1].split(None, 1)
    op = parts[0] if parts else ""
    a = [o.strip() for o in re.split(r",(?![^(]*\))", parts[1])] if len(parts) > 1 else []
    return op, a
# ---- forbidden opcodes (prefixes stripped, so `rep bsf` = TZCNT is caught)
FORBID = re.compile(r"^(call|div|idiv|cvtsi2s|vcvtsi2s|cvtusi|vcvtusi|f(?!ence)[a-z]+|vfmadd|vfnmadd|vfmsub|vfnmsub|bsf|bsr|lzcnt|tzcnt|stos|movs[bwlq]?$|cmps[bwlq]?$|scas|syscall|int\b)")
ins = [(k, v) for k, v in body if k == "ins"]
print("splice_get_probe: %d instructions" % len(ins))
bad = []
for i, (k, s) in enumerate(body):
    if k != "ins": continue
    op, a = split(s)
    if op.startswith("nop"): continue
    if FORBID.match(op): bad.append(s); continue
    if op.startswith("jmp") and a and a[0].startswith("*"):
        # allowed only as a local jump table: a table label used earlier in the same basic block
        j = i
        while j > 0 and body[j - 1][0] == "ins": j -= 1
        near = " ".join(v for kk, v in body[j:i])
        if not any(re.search(r"(?<![\w.$])" + re.escape(t) + r"(?![\w.$])", near) for t in tables): bad.append(s + "    (indirect jump, no jump table)")
    elif op.startswith("j") and a and not re.match(r"^\.?L", a[0]): bad.append(s + "    (jump out of the function)")
status = 0
if bad:
    status = 1; print("FORBIDDEN instructions:")
    for s in bad: print("  " + s)
# ---- registers: canonical 64-bit names
R64 = {}
for r in ("ax", "bx", "cx", "dx"):
    for n in ("r" + r, "e" + r, r, r[0] + "l", r[0] + "h"): R64["%" + n] = "%r" + r
for r in ("si", "di", "bp", "sp"):
    for n in ("r" + r, "e" + r, r, r + "l"): R64["%" + n] = "%r" + r
for i in range(8, 16):
    for suf in ("", "d", "w", "b"): R64["%%r%d%s" % (i, suf)] = "%%r%d" % i
MEM = re.compile(r"^(-?[\w.$+-]*)\((%\w+)?(?:,(%\w+))?(?:,\d)?\)$")
frame = any(re.sub(r"\s+", "", s) == "movq%rsp,%rbp" for k, s in ins)
STACK = ("%rsp", "%rbp") if frame else ("%rsp",)
def slot(o):
    m = MEM.match(o)
    return o if (m and m.group(2) in STACK and not m.group(3)) else None
READONLY = ("cmp", "test", "bt", "ucomi", "comi", "ptest", "prefetch", "nop", "push", "mul", "div", "idiv", "jmp", "call", "j")
def written(op, a):
    """GPRs (canonical) and stack slots this instruction may write."""
    regs, slots = set(), set()
    if op.startswith(("cmp", "test", "bt", "ucomi", "comi", "ptest", "prefetch", "nop", "j")) and not op.startswith(("bts", "btr", "btc", "cmov", "cmpxchg")):
        return regs, slots
    if op.startswith(("mul", "div", "idiv", "cqt", "cqo", "cltd", "cdq", "cpuid", "rdtsc")) or (op.startswith("imul") and len(a) == 1):
        regs |= {"%rax", "%rdx"}
    if op.startswith(("push", "pop", "call", "ret", "leave", "enter")): regs.add("%rsp")
    if op.startswith(("xchg", "xadd", "cmpxchg")):
        for o in a: regs.add(R64.get(o, "")); slots.add(slot(o))
    if a and not op.startswith("push"):
        d = a[-1]
        if d in R64: regs.add(R64[d])
        elif slot(d): slots.add(d)
    regs.discard(""); slots.discard(None)
    return regs, slots
# ---- CFG over blocks
blocks, labels, curb = [], {}, None
for k, v in body:
    if k == "label":
        curb = {"label": v, "ins": [], "succ": []}; blocks.append(curb); labels[v] = len(blocks) - 1; continue
    if curb is None or (curb["ins"] and split(curb["ins"][-1])[0].startswith(("j", "ret"))):
        curb = {"label": None, "ins": [], "succ": []}; blocks.append(curb)
    curb["ins"].append(v)
for bi, b in enumerate(blocks):
    last = split(b["ins"][-1]) if b["ins"] else ("", [])
    op, a = last
    falls = not (op.startswith("jmp") or op.startswith("ret"))
    if op.startswith("j") and a:
        if a[0].startswith("*"):
            for t, ent in tables.items():
                if any(re.search(r"(?<![\w.$])" + re.escape(t) + r"(?![\w.$])", s) for s in b["ins"]):
                    b["succ"] += [labels[e] for e in ent if e in labels]
        elif a[0] in labels: b["succ"].append(labels[a[0]])
    if falls and bi + 1 < len(blocks): b["succ"].append(bi + 1)
TOP = None   # unvisited
def transfer(state, s):
    regs, slots = state
    op, a = split(s)
    if op in ("movq",) and len(a) == 2:
        src, dst = a
        has = src in regs or src in slots
        if dst in R64 and R64[dst] == dst:
            regs = (regs | {dst}) if has else (regs - {dst})
            if dst == "%rsp": slots = frozenset() if "%rbp" not in STACK else slots
            return frozenset(regs), frozenset(slots)
        if slot(dst):
            slots = (slots | {dst}) if has else (slots - {dst})
            return frozenset(regs), frozenset(slots)
    wr, ws = written(op, a)
    regs = regs - wr; slots = slots - ws
    if "%rsp" in wr and "%rbp" not in STACK: slots = frozenset()   # %rsp-relative slots move
    return frozenset(regs), frozenset(slots)
IN = [TOP] * len(blocks); IN[0] = (frozenset(["%rdx"]), frozenset())
work = [0]
while work:
    bi = work.pop()
    st = IN[bi]
    for s in blocks[bi]["ins"]: st = transfer(st, s)
    for sb in blocks[bi]["succ"]:
        new = st if IN[sb] is TOP else (IN[sb][0] & st[0], IN[sb][1] & st[1])
        if new != IN[sb]: IN[sb] = new; work.append(sb)
print("memory-destination stores (base register):")
nbad = nout = shown = 0
for bi, b in enumerate(blocks):
    st = IN[bi] if IN[bi] is not TOP else (frozenset(), frozenset())
    for s in b["ins"]:
        op, a = split(s)
        d = a[-1] if a else ""
        m = MEM.match(d)
        if m and not op.startswith(READONLY) and not (op.startswith("imul") and len(a) == 1):
            base = m.group(2) or "(none)"
            if base in STACK and not m.group(3): kind = "stack"
            elif base in st[0] and not m.group(3) and m.group(1) in ("", "0"): kind = "out"; nout += 1
            else: kind = "OTHER"; nbad += 1
            print("  %s    [%s %s]" % (s, kind, base)); shown += 1
        elif op.startswith("push"):
            print("  %s    [stack]" % s); shown += 1
        st = transfer(st, s)
if not shown: print("  (none)")
print("  out-parameter stores: %d; other stores: %d; frame pointer: %s; jump tables: %s" % (nout, nbad, "yes" if frame else "no", " ".join(sorted(tables)) or "none"))
if nbad: status = 1
sys.exit(status)
PY
[ $status = 0 ] && echo "splice_asm_check: PASS" || echo "splice_asm_check: FAIL"
exit $status
