#!/usr/bin/env python
"""Alternative graph chunking (partitioning) methods for IDEA.

IDEA's default chunker is rMETIS (recursive METIS bisection + silhouette based
shatter / reassignment, ``ChunkTree.chunking(chunkingMethod='metis')``).  This
module adds community-detection and spectral partitioners that can be chosen
with ``-chunkingMethod``.  Ported from paper_RLclustering/experiment_model/core/
partition.py and adapted to IDEA's dict-of-dicts WNN graph.

  leiden_cpm  Leiden, Constant Potts Model; resolution searched so that the
              number of communities is close to nChunk          (needs igraph)
  leiden_mod  Leiden, modularity with resolution parameter, searched likewise
                                                                (needs igraph)
  louvain     Louvain (multilevel) modularity, resolution searched likewise
                                                                (needs igraph)
  spectral    recursive spectral bisection (Fiedler vector of the normalised
              Laplacian), always splitting the largest chunk

Common post-processing (all methods):
  * every chunk is split into its connected components;
  * if there are more than nChunk pieces, the smallest piece is merged into
    the neighbouring piece it shares the most edge weight with;
  * if there are fewer than nChunk chunks, the largest one is bisected with
    METIS until nChunk is reached (as in rMETIS stage 1).
All methods are deterministic for a fixed ``seed``.
"""
import random

import numpy as np
import scipy.sparse as sp

METHODS = ('leiden_cpm', 'leiden_mod', 'louvain', 'spectral')


def _toCSR(dic_node2node_weight, lst_node):
	idx = dict((node, i) for i, node in enumerate(lst_node))
	rows, cols, vals = [], [], []
	for node in lst_node:
		i = idx[node]
		for n2, w in dic_node2node_weight[node].items():
			j = idx.get(n2)
			if j is None or j == i:
				continue
			rows.append(i); cols.append(j); vals.append(float(w))
	n = len(lst_node)
	W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
	return W.maximum(W.T).tocsr()


def _toIgraph(W):
	import igraph as ig
	coo = sp.triu(W, k=1).tocoo()
	g = ig.Graph(n=W.shape[0], edges=list(zip(coo.row.tolist(), coo.col.tolist())))
	g.es['weight'] = coo.data.tolist()
	return g


def _searchResolution(run, K, lo, hi, r0, maxIter=20):
	"""Find a resolution whose community count is closest to K (log-secant + bisection)."""
	tol = max(1, K // 50)
	best, prev, r = None, None, r0
	for _ in range(maxIter):
		nk, part = run(r)
		if best is None or abs(nk - K) < abs(best[0] - K):
			best = (nk, part, r)
		if abs(nk - K) <= tol:
			break
		lo, hi = (r, hi) if nk < K else (lo, r)
		slope = None
		if prev is not None and prev[1] != nk and prev[0] != r and nk > 0 and prev[1] > 0:
			slope = (np.log(nk) - np.log(prev[1])) / (np.log(r) - np.log(prev[0]))
		prev = (r, nk)
		rNew = r * (K / max(nk, 1)) ** (1.0 / slope) if slope is not None and slope > 1e-3 else np.sqrt(lo * hi)
		r = rNew if lo < rNew < hi else float(np.sqrt(lo * hi))
	return best[1]


def _leiden(W, K, objective, seed):
	import igraph as ig
	ig.set_random_number_generator(random.Random(seed))
	g = _toIgraph(W)
	def run(r):
		p = g.community_leiden(objective_function=objective, weights='weight', resolution=r, n_iterations=3)
		m = np.asarray(p.membership)
		return len(np.unique(m)), m
	if objective == 'CPM':
		# CPM resolution is on the scale of edge weights
		w = np.asarray(g.es['weight']) if g.ecount() else np.ones(1)
		return _searchResolution(run, K, 1e-6 * w.max(), 10.0 * w.max(), float(np.median(w)) * 0.1)
	return _searchResolution(run, K, 1e-3, 1e4, 1.0)


def _louvain(W, K, seed):
	import igraph as ig
	ig.set_random_number_generator(random.Random(seed))
	g = _toIgraph(W)
	def run(r):
		m = np.asarray(g.community_multilevel(weights='weight', resolution=r).membership)
		return len(np.unique(m)), m
	return _searchResolution(run, K, 1e-3, 1e4, 1.0)


def _fiedlerBisect(W, nodes):
	sub = W[nodes][:, nodes].tocsr()
	m = len(nodes)
	if m < 2:
		return nodes, nodes[:0]
	if m <= 3:
		return nodes[:1], nodes[1:]
	deg = np.asarray(sub.sum(axis=1)).ravel()
	deg[deg <= 0] = 1e-12
	Dm = sp.diags(1.0 / np.sqrt(deg))
	L = (sp.eye(m) - Dm @ sub @ Dm).tocsc()
	try:
		from scipy.sparse.linalg import eigsh
		if m <= 200:
			vals, vecs = np.linalg.eigh(L.toarray())
		else:
			# smallest eigenpairs of L == largest of (2I - L): plain Lanczos, no factorisation
			vals, vecs = eigsh((2.0 * sp.eye(m) - L).tocsr(), k=2, which='LA', v0=np.ones(m), tol=1e-6, maxiter=5000)
			vals = 2.0 - vals
		f = vecs[:, np.argsort(vals)[1]]
		left = f <= np.median(f)
		if 0 < left.sum() < m:
			return nodes[left], nodes[~left]
	except Exception:
		pass
	half = m // 2
	return nodes[:half], nodes[half:]


def _spectral(W, K):
	parts = [np.arange(W.shape[0])]
	while len(parts) < K:
		i = int(np.argmax([len(p) for p in parts]))
		if len(parts[i]) < 2:
			break
		a, b = _fiedlerBisect(W, parts[i])
		if len(a) == 0 or len(b) == 0:
			break
		parts.pop(i)
		parts.extend([a, b])
	m = np.empty(W.shape[0], dtype=np.int64)
	for c, p in enumerate(parts):
		m[p] = c
	return m


def _mergeSmallest(dic_node2node_weight, lst_set, nChunk):
	node2c = {}
	for c, nodes in enumerate(lst_set):
		for node in nodes:
			node2c[node] = c
	alive = dict((c, set(nodes)) for c, nodes in enumerate(lst_set))
	while len(alive) > nChunk:
		c = min(alive, key=lambda x: (len(alive[x]), x))
		w = {}
		for node in alive[c]:
			for n2, wt in dic_node2node_weight[node].items():
				c2 = node2c.get(n2)
				if c2 is not None and c2 != c:
					w[c2] = w.get(c2, 0.0) + wt
		if not w:
			break  # isolated piece (cannot happen inside a connected subgraph)
		target = max(sorted(w), key=lambda x: w[x])
		for node in alive[c]:
			node2c[node] = target
		alive[target] |= alive.pop(c)
	return [alive[c] for c in sorted(alive)]


def partition(wKNNgraph, set_node, nChunk, method, seed=0, bisect=None, connectedComponent=None):
	"""Split ``set_node`` of ``wKNNgraph`` into about nChunk node sets (list of sets).

	bisect(set_node) -> list of node sets   (METIS bisection used to reach nChunk)
	connectedComponent(wKNNgraph, set_node) -> list of node sets
	"""
	if method not in METHODS:
		raise ValueError('unknown chunking method %r; choose from %s' % (method, METHODS))
	lst_node = sorted(set_node)
	W = _toCSR(wKNNgraph.dic_node2node_weight, lst_node)
	if method == 'leiden_cpm':
		m = _leiden(W, nChunk, 'CPM', seed)
	elif method == 'leiden_mod':
		m = _leiden(W, nChunk, 'modularity', seed)
	elif method == 'louvain':
		m = _louvain(W, nChunk, seed)
	else:
		m = _spectral(W, nChunk)
	groups = {}
	for node, c in zip(lst_node, m.tolist()):
		groups.setdefault(c, set()).add(node)
	lst_set = []
	for c in sorted(groups):
		lst_set += connectedComponent(wKNNgraph, groups[c])
	# too many pieces (e.g. disconnected spectral halves): merge the smallest piece
	# into the neighbouring piece it is most strongly connected to
	if len(lst_set) > nChunk:
		lst_set = _mergeSmallest(wKNNgraph.dic_node2node_weight, lst_set, nChunk)
	# reach nChunk by bisecting the largest chunk (deterministic order)
	while len(lst_set) < nChunk:
		i = max(range(len(lst_set)), key=lambda j: (len(lst_set[j]), -j))
		if len(lst_set[i]) < 2:
			break
		pieces = [p for p in bisect(lst_set[i]) if len(p) > 0]
		if len(pieces) < 2:
			break
		lst_set.pop(i)
		lst_set.extend(pieces)
	return lst_set
