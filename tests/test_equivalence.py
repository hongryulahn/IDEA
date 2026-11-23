#!/usr/bin/env python
"""Randomised equivalence tests: optimised code paths vs. the original algorithm.

    python tests/test_equivalence.py [--orig PATH_TO_ORIGINAL_IDEA.py] [-n N]

* _reassignShatteredNodes  vs. the original re-sort loop (copied below verbatim)
* hierarchicalClusteringAgent (compiled kernel and pure-Python paths) vs. the
  agent of the original IDEA.py, for all 11 'auto' linkage methods, with merge
  prefixes and forbidden pairs.  Integer-valued weights are used on purpose so
  that many exact ties occur.
"""
import argparse, copy, importlib.util, os, random, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import IDEA as NEW

AUTO = ['average','weighted','single','complete','deltaRev','deltaModularity','minPossibleEdgeCountNormalized',
        'minNodeNormalized','minEdgeCountNormalized','minEdgeWeightNormalized','minModularityWeightNormalized']


def orig_reassign(set_shatteredNode, dic_node2cluster, dic_cluster2nodes, g):
    dic_shatteredNode2clusterWeightSum={}
    for node in set_shatteredNode:
        dic_c2w={}
        for n2, weight in g[node].items():
            c2 = dic_node2cluster[n2]
            if c2 == -1:
                continue
            if c2 not in dic_c2w:
                dic_c2w[c2]=0.0
            dic_c2w[c2]+=weight
        if len(dic_c2w) == 0:
            weightSum = 0.0
        else:
            weightSum = max(dic_c2w.values())
        dic_shatteredNode2clusterWeightSum[node]=[weightSum, dic_c2w]
    lst_shatteredNode=list(set_shatteredNode)
    order=[]
    while True:
        if len(lst_shatteredNode) == 0:
            break
        lst_shatteredNode.sort(key=lambda x:dic_shatteredNode2clusterWeightSum[x][0],reverse=True)
        node = lst_shatteredNode.pop(0)
        order.append(node)
        weightSum, dic_c2w = dic_shatteredNode2clusterWeightSum[node]
        cluster = sorted(dic_c2w.items(),key=lambda x:x[1], reverse=True)[0][0]
        dic_cluster2nodes[cluster].add(node)
        dic_node2cluster[node]=cluster
        for n2, weight in g[node].items():
            c2 = dic_node2cluster[n2]
            if c2 != -1:
                continue
            weightSum, dic_c2w = dic_shatteredNode2clusterWeightSum[n2]
            if cluster not in dic_c2w:
                dic_c2w[cluster]=0.0
            dic_c2w[cluster]+=weight
            weightSum = max(dic_c2w.values())
            dic_shatteredNode2clusterWeightSum[n2] = weightSum, dic_c2w
    return order


def random_graph(rng, n, deg, wlo, whi):
    g = dict((i, {}) for i in range(n))
    for i in range(n):
        for j in rng.sample(range(n), min(deg, n-1)):
            if i != j:
                w = float(rng.randint(wlo, whi))
                g[i][j] = w
                g[j][i] = w
    return g


def test_shattered(rng, n_cases):
    for case in range(n_cases):
        n = rng.randint(5, 300)
        g = random_graph(rng, n, rng.randint(2, 8), -2 if case % 3 == 0 else 1, rng.choice([2, 4, 50]))
        nC = rng.randint(1, 6)
        dic_node2cluster = dict((i, rng.randrange(nC)) for i in range(n))
        shattered = set(rng.sample(range(n), rng.randint(1, max(1, n // 2))))
        # every shattered node needs at least one non-shattered neighbour eventually; skip bad cases
        dic_cluster2nodes = dict((c, set()) for c in range(nC))
        for i, c in dic_node2cluster.items():
            if i not in shattered:
                dic_cluster2nodes[c].add(i)
        for i in shattered:
            dic_node2cluster[i] = -1
        a = (copy.deepcopy(dic_node2cluster), copy.deepcopy(dic_cluster2nodes))
        b = (copy.deepcopy(dic_node2cluster), copy.deepcopy(dic_cluster2nodes))
        try:
            orig_reassign(shattered, a[0], a[1], g)
            ok_a = True
        except IndexError:
            ok_a = False
        try:
            NEW._reassignShatteredNodes(shattered, b[0], b[1], g)
            ok_b = True
        except IndexError:
            ok_b = False
        assert ok_a == ok_b, ('exception mismatch', case)
        if ok_a:
            assert a == b, ('shattered mismatch', case)
    print('shattered-node reassignment: %d random cases identical' % n_cases)


def chunk_inputs(rng, n0, deg, integer):
    dic_n = dict((c, rng.randint(1, 30)) for c in range(n0))
    dic_cnt = dict((c, float(rng.randint(0, 20))) for c in range(n0))
    dic_sw = dict((c, float(rng.randint(0, 20)) if integer else rng.random()*10) for c in range(n0))
    emax, emin, ecnt, esum = {}, {}, {}, {}
    for a in range(n0):
        for b in rng.sample(range(n0), min(deg, n0-1)):
            if a == b:
                continue
            e = (min(a, b), max(a, b))
            if e in esum:
                continue
            k = rng.randint(1, 5)
            ws = [float(rng.randint(1, 3)) if integer else rng.random() for _ in range(k)]
            emax[e], emin[e], ecnt[e], esum[e] = max(ws), min(ws), float(k), sum(ws)
    nTotal = sum(dic_n.values())
    return dict(dic_edge2maxWeight=emax, dic_edge2minWeight=emin, dic_edge2countWeight=ecnt, dic_edge2sumWeight=esum,
                dic_node2n=dic_n, dic_node2countWeight=dic_cnt, dic_node2sumWeight=dic_sw, nTotalNode=nTotal)


def lm_key(lm):
    return [(x[0], x[1], x[2], tuple(sorted(x[3].items()))) for x in lm]


def run_agent(mod, n0, method, kw, prefix, forb, use_kernel):
    ag = mod.hierarchicalClusteringAgent(list(range(n0)), method, **copy.deepcopy(kw))
    if use_kernel is not None:
        ag.useKernel = use_kernel
    try:
        ag.fit(prefix, set_forbiddenPair=forb) if forb else ag.fit(prefix)
    except ZeroDivisionError:
        return 'ZeroDivisionError'
    return lm_key(ag.linkageMatrix)


def test_agents(rng, ORIG, n_cases):
    stats = {}
    for case in range(n_cases):
        n0 = rng.randint(2, 60)
        integer = case % 2 == 0
        kw = chunk_inputs(rng, n0, rng.randint(1, 6), integer)
        for method in AUTO:
            ref_full = run_agent(ORIG, n0, method, kw, [], None, None)
            if ref_full == 'ZeroDivisionError':
                prefix = []
            else:
                cut = rng.randint(0, len(ref_full))
                prefix = [[x[0], x[1], x[2], dict(x[3])] for x in ref_full[:cut]]
            forb = None
            if ref_full != 'ZeroDivisionError' and len(ref_full) > len(prefix) and rng.random() < 0.5:
                x = ref_full[len(prefix)]
                forb = {(min(x[0], x[1]), max(x[0], x[1]))}
            ref = run_agent(ORIG, n0, method, kw, prefix, forb, None)
            for use_kernel in (True, False):
                got = run_agent(NEW, n0, method, kw, prefix, forb, use_kernel)
                assert got == ref, ('agent mismatch', case, method, use_kernel, n0)
            stats[method] = stats.get(method, 0) + 1
    print('agents: %d random graphs x %d methods identical (kernel=%s)' % (n_cases, len(AUTO), NEW._kernel is not None))


def test_node_level(rng, ORIG, n_cases):
    """fullLinkageMatrix-style agents: arbitrary node ids, default (float 1.0) sizes."""
    for case in range(n_cases):
        n = rng.randint(2, 80)
        ids = sorted(rng.sample(range(100000), n))
        rng.shuffle(ids)
        integer = case % 2 == 0
        ew = {}
        for a in ids:
            for b in rng.sample(ids, min(rng.randint(1, 6), n)):
                if a != b:
                    ew[(min(a, b), max(a, b))] = float(rng.randint(1, 3)) if integer else rng.random()
        cnt = dict((e, 1.0) for e in ew)
        for method in AUTO:
            def run(mod, use_kernel):
                ag = mod.hierarchicalClusteringAgent(set(ids), method, dic_edge2maxWeight=ew, dic_edge2minWeight=ew,
                        dic_edge2countWeight=cnt, dic_edge2sumWeight=ew, nTotalNode=n)
                if use_kernel is not None:
                    ag.useKernel = use_kernel
                try:
                    ag.fit()
                except (ZeroDivisionError, ValueError):
                    return 'error'
                return [(x[0], x[1], x[2], tuple((k, v, type(v).__name__) for k, v in sorted(x[3].items()))) for x in ag.linkageMatrix]
            ref = run(ORIG, None)
            for uk in (True, False):
                assert run(NEW, uk) == ref, ('node-level mismatch', case, method, uk)
    print('node-level agents: %d random graphs x %d methods identical (incl. value types)' % (n_cases, len(AUTO)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig', default=os.path.join(os.path.dirname(HERE), 'IDEA_orig', 'IDEA.py'))
    ap.add_argument('-n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('IDEA_orig_mod', a.orig)
    ORIG = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.orig)))
    spec.loader.exec_module(ORIG)
    rng = random.Random(a.seed)
    test_shattered(rng, a.n * 5)
    test_agents(rng, ORIG, a.n)
    test_node_level(rng, ORIG, a.n)
    print('OK')

if __name__ == '__main__':
    main()
