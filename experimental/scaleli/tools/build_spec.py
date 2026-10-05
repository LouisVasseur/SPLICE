#!/usr/bin/env python3
"""Render the editable specification to HTML/PDF. Optional: pandoc + weasyprint.
No network retrieval is required; figures must exist locally (plot_results.py).
"""
from __future__ import annotations
import argparse,pathlib,shutil,subprocess

def main() -> None:
    root=pathlib.Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,default=root/'docs/TECHNICAL_SPEC.pdf');a=p.parse_args()
    for program in ['pandoc','weasyprint']:
        if not shutil.which(program):p.error(f'install optional document dependency: {program}')
    source=root/'docs/TECHNICAL_SPEC.md';html=root/'docs/TECHNICAL_SPEC.html'
    subprocess.run(['pandoc',str(source),'-s','--toc','--toc-depth=1','--css=print.css','--metadata=pagetitle:SCALE-LI Technical Specification','-o',str(html)],check=True)
    subprocess.run(['weasyprint',str(html),str(a.output.resolve())],check=True)
    print(a.output)
if __name__=='__main__':main()
