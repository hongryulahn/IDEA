#!/usr/bin/env python
"""Print a markdown ARI/NMI/time table from several run_bench.py output dirs.

usage: python bench/compare_methods.py LABEL=DIR [LABEL=DIR ...]
"""
import json, sys

cols = [a.split('=', 1) for a in sys.argv[1:]]
R = dict((c, json.load(open(d + '/summary.json'))) for c, d in cols)
names = list(R[cols[0][0]])
print('| data | ' + ' | '.join(c for c, _ in cols) + ' |')
print('|---|' + '---|' * len(cols))
for n in names:
    print('| %s | ' % n + ' | '.join('%.3f' % R[c][n]['ari'] for c, _ in cols) + ' |')
mean = lambda c, k: sum(R[c][n][k] for n in names) / len(names)
print('| **mean ARI** | ' + ' | '.join('**%.4f**' % mean(c, 'ari') for c, _ in cols) + ' |')
print('| mean NMI | ' + ' | '.join('%.4f' % mean(c, 'nmi') for c, _ in cols) + ' |')
print('| total time (s) | ' + ' | '.join('%.0f' % sum(R[c][n]['time'] for n in names) for c, _ in cols) + ' |')
