#!/usr/bin/env python
"""Run IDEA on the bundled datasets, record wall time, ARI/NMI, and optionally
compare outputs against a reference run.

usage: python bench/run_bench.py CODE_DIR OUT_DIR [-j N] [--ref REF_DIR] [--only A01,B02]
"""
import argparse, os, subprocess, sys, time, json
from concurrent.futures import ThreadPoolExecutor

DATASETS = [
    ('A01', 15, 'datapoints'), ('A02', 31, 'datapoints'), ('A03', 35, 'datapoints'),
    ('A04', 15, 'datapoints'), ('A05', 15, 'datapoints'), ('A06', 15, 'datapoints'),
    ('B01', 3, 'datapoints'), ('B02', 6, 'datapoints'), ('B03', 9, 'datapoints'),
    ('B04', 8, 'datapoints'), ('B05', 10, 'datapoints'), ('B06', 9, 'datapoints'),
    ('C01', 7, 'datapoints'), ('C02', 6, 'datapoints'), ('C03', 9, 'datapoints'),
    ('C04', 8, 'datapoints'), ('C05', 10, 'datapoints'), ('C06', 9, 'datapoints'),
    ('D01', 11, 'dissimilarity'), ('D02', 5, 'dissimilarity'),
]
EXTRA = []  # extra CLI args, set via --extra

def datafile(name, dtype):
    if dtype == 'datapoints':
        return '%s.data.points' % name
    return {'D01': 'D01.data.dist.fullmatrix', 'D02': 'D02.data.dist.sparsematrix'}[name]

def run_one(code_dir, data_dir, out_dir, name, k, dtype):
    out = os.path.join(out_dir, name)
    cmd = [sys.executable, os.path.join(code_dir, 'IDEA.py'), os.path.join(data_dir, datafile(name, dtype)),
           str(k), '-dataType', dtype] + EXTRA + ['-o', out + '.cluster', '-o2', out + '.linkage']
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True,
                       env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    el = time.time() - t
    if p.returncode != 0:
        return name, {'error': '\n'.join(p.stderr.strip().split('\n')[-3:]), 'time': el}
    return name, {'time': el}

def score(data_dir, out_dir, name):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    lab = dict(l.split() for l in open(os.path.join(data_dir, name + '.label')))
    pred = dict(l.split() for l in open(os.path.join(out_dir, name + '.cluster')))
    keys = [k for k in pred if k in lab]
    y, p = [lab[k] for k in keys], [pred[k] for k in keys]
    return adjusted_rand_score(y, p), normalized_mutual_info_score(y, p)

def same(a, b):
    return os.path.exists(a) and os.path.exists(b) and open(a).read() == open(b).read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code_dir'); ap.add_argument('out_dir')
    ap.add_argument('-j', type=int, default=20)
    ap.add_argument('--ref', default=None)
    ap.add_argument('--only', default=None)
    ap.add_argument('--extra', default='', help='extra args passed to IDEA.py, e.g. "-outlierRatio 0.09"')
    a = ap.parse_args()
    EXTRA[:] = a.extra.split()
    code_dir = os.path.abspath(a.code_dir)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(a.out_dir, exist_ok=True)
    ds = DATASETS if a.only is None else [d for d in DATASETS if d[0] in a.only.split(',')]
    with ThreadPoolExecutor(a.j) as ex:
        res = dict(ex.map(lambda d: run_one(code_dir, data_dir, a.out_dir, *d), ds))
    tot = 0.0
    print('%-4s %8s %7s %7s  %s' % ('data', 'time(s)', 'ARI', 'NMI', 'vs-ref(cluster/linkage)'))
    for name, _, _ in ds:
        r = res[name]; tot += r['time']
        if 'error' in r:
            print('%-4s %8.2f ERROR\n%s' % (name, r['time'], r['error'])); continue
        r['ari'], r['nmi'] = score(data_dir, a.out_dir, name)
        cmp = ''
        if a.ref:
            c = same(os.path.join(a.out_dir, name + '.cluster'), os.path.join(a.ref, name + '.cluster'))
            l = same(os.path.join(a.out_dir, name + '.linkage'), os.path.join(a.ref, name + '.linkage'))
            r['same_cluster'], r['same_linkage'] = c, l
            cmp = '%s/%s' % ('same' if c else 'DIFF', 'same' if l else 'DIFF')
        print('%-4s %8.2f %7.4f %7.4f  %s' % (name, r['time'], r['ari'], r['nmi'], cmp))
    print('total time %.1fs' % tot)
    json.dump(res, open(os.path.join(a.out_dir, 'summary.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
