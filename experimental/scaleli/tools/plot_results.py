#!/usr/bin/env python3
"""Regenerate documentation figures from the included, actually executed experiments.
Requires optional matplotlib; does not run or invent benchmark results.
"""
from __future__ import annotations
import argparse, csv, json, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main() -> None:
    root=pathlib.Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',type=pathlib.Path,default=root/'results/example/layout/summary.json')
    p.add_argument('--summary',type=pathlib.Path,default=root/'results/example/suite/summary.csv')
    p.add_argument('--output',type=pathlib.Path,default=root/'docs/figures')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads(a.layout.read_text())['results']
    labels={'raw':'Raw','min_bytes':'Minimum bytes','smooth':'Density-target smoothing'}
    fig,ax=plt.subplots(figsize=(9.6,5.7))
    for row in rows:
        x=row['bytes_per_key'];y=row['byte_normalized_mse']*1e5
        ax.scatter([x],[y],s=85)
        offset={'raw':(-135,-35),'min_bytes':(12,14),'smooth':(12,15)}[row['policy']]
        text=f"{labels[row['policy']]}\nMean block error: {row['byte_mean_block_error']:.3f}"
        ax.annotate(text,(x,y),xytext=offset,textcoords='offset points',fontsize=11)
    ax.set(xlabel='Encoded key-arena bytes per key (includes block headers)',
           ylabel='Normalized byte-position MSE (× 10⁻⁵)',xlim=(0.8,9),ylim=(0,11))
    ax.set_title('Straighter byte geometry does not guarantee better block prediction',pad=17,fontsize=13)
    ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(a.output/'layout_tradeoff.png',dpi=180);plt.close(fig)
    with a.summary.open() as f:all_rows=list(csv.DictReader(f))
    names={'sorted_vector':'Sorted vector','ordered_map':'Ordered map','raw_rank':'Raw / rank',
           'packed_rank':'Packed / rank','packed_byte':'Packed / byte','smooth_byte':'Smooth / byte'}
    rows=[r for r in all_rows if r['dataset']=='dense_sparse' and r['profile']=='read_only' and r['variant'] in names]
    fig,ax=plt.subplots(figsize=(9.6,5.8))
    offsets={'sorted_vector':(12,15),'ordered_map':(-100,15),'raw_rank':(12,-24),
             'packed_rank':(-75,-36),'packed_byte':(15,18),'smooth_byte':(12,-10)}
    for row in rows:
        x=float(row['initial_bytes_per_key']);y=float(row['throughput_ops_s_median'])/1e6
        ax.scatter([x],[y],s=80)
        ax.annotate(names[row['variant']],(x,y),xytext=offsets[row['variant']],textcoords='offset points',
                    fontsize=11,arrowprops={'arrowstyle':'-','lw':.7})
    ax.set(xlabel='Full accounted bytes per initial key (lower is smaller)',ylabel='Throughput (million operations/second)',
           xlim=(6,53),ylim=(0,max(float(r['throughput_ops_s_median'])/1e6 for r in rows)*1.22))
    ax.set_title('Read-only dense/sparse controls: real savings, no established speedup',pad=17,fontsize=13)
    ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(a.output/'read_only_tradeoff.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
