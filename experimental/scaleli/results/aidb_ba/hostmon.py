import os, subprocess, time, json
f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hostmon.jsonl'), 'a')
while True:
    ps = subprocess.run(['ps', '-Ao', 'pcpu=,comm='], capture_output=True, text=True).stdout.splitlines()
    bench = other = 0.0; top = []
    for l in ps:
        l = l.strip()
        if not l: continue
        c, _, name = l.partition(' ')
        try: c = float(c)
        except: continue
        if 'scaleli_bench' in name: bench += c
        else: other += c; top.append((c, name.strip()[-60:]))
    top.sort(reverse=True)
    f.write(json.dumps({'t': time.time(), 'la': os.getloadavg()[0], 'bench_cpu': bench, 'other_cpu': other, 'top': top[:3]}) + '\n'); f.flush()
    time.sleep(5)
