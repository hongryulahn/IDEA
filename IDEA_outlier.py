#!/usr/bin/env python
"""Optional outlier post-processing for IDEA (not part of the published algorithm).

IDEA's DynamicOutlierCutTree (``cutTree``) returns k clusters plus outlier
points labelled -1. ``assignOutliers`` assigns every -1 point to one of the k
clusters, so that the result has exactly k clusters. It runs after
``cutTree`` and leaves the original algorithm unchanged; it is enabled with
``-outlierAssignment greedy`` (default ``none`` keeps the -1 labels).

Units ("outlier leaf chunks"): maximal subtrees of the clustering tree whose
leaves are all labelled -1. Each chunk is assigned as a whole.

Closeness: sum of WNN edge weights between the chunk and a cluster (the same
connectivity criterion IDEA uses to reassign shattered nodes during chunking).

Greedy, one chunk at a time: repeatedly attach the (chunk, cluster) pair with
the largest connecting weight; after a chunk is attached, the chunks adjacent
to it gain weight towards that cluster. This handles chains (a chunk that is
only connected to other outlier chunks becomes attachable once a neighbour
has been attached). It is implemented with a priority queue, which gives the
same result as recomputing all scores and attaching the single best pair at
every step. Ties: larger weight first, then smaller chunk index (chunks are
ordered by their smallest leaf id), then smaller cluster label.

Chunks with no graph connection to any cluster (e.g. a separate connected
component) are attached through the tree: to the cluster that shares the
deepest common ancestor with the chunk (most points under that ancestor,
then smaller label).
"""
import heapq


def _tree(linkageMatrix, maxLeafID=None):
	if maxLeafID is None:
		maxLeafID = len(linkageMatrix)
	children, parent = {}, {}
	for c, x in enumerate(linkageMatrix, maxLeafID + 1):
		c1, c2 = int(x[0]), int(x[1])
		children[c] = (c1, c2)
		parent[c1] = c
		parent[c2] = c
	return children, parent


def _leaves(node, children):
	out, stack = [], [node]
	while stack:
		v = stack.pop()
		if v in children:
			stack.extend(children[v])
		else:
			out.append(v)
	return out


def outlierChunks(linkageMatrix, dic_node2cluster, maxLeafID=None):
	"""Maximal subtrees whose leaves are all -1, as a list of (root, sorted leaves), ordered by smallest leaf."""
	children, parent = _tree(linkageMatrix, maxLeafID)
	allOut = {}
	for leaf, c in dic_node2cluster.items():
		allOut[leaf] = (c == -1)
	for c in sorted(children):                 # children are created before parents
		c1, c2 = children[c]
		allOut[c] = allOut[c1] and allOut[c2]
	chunks = []
	for v, flag in allOut.items():
		if flag and not allOut.get(parent.get(v), False):
			chunks.append((v, sorted(_leaves(v, children))))
	chunks.sort(key=lambda x: x[1][0])
	return chunks


def assignOutliers(linkageMatrix, dic_node2cluster, dic_node2node_weight, maxLeafID=None, returnStats=False):
	"""Return a copy of dic_node2cluster in which every -1 point belongs to a cluster."""
	out = dict(dic_node2cluster)
	chunks = outlierChunks(linkageMatrix, dic_node2cluster, maxLeafID)
	stats = {'chunks': len(chunks), 'points': sum(len(l) for _, l in chunks), 'direct': 0, 'chain': 0, 'tree': 0}
	if not chunks:
		return (out, stats) if returnStats else out
	children, parent = _tree(linkageMatrix, maxLeafID)

	owner = {}                                  # node -> chunk index (for -1 points)
	for i, (_, leaves) in enumerate(chunks):
		for v in leaves:
			owner[v] = i
	score = [dict() for _ in chunks]            # chunk -> {cluster: weight}
	adj = [dict() for _ in chunks]              # chunk -> {chunk: weight}
	for i, (_, leaves) in enumerate(chunks):
		s, a = score[i], adj[i]
		for v in leaves:
			for n2, w in dic_node2node_weight.get(v, {}).items():
				j = owner.get(n2)
				if j is None:
					c = dic_node2cluster.get(n2, -1)
					if c != -1:
						s[c] = s.get(c, 0.0) + w
				elif j != i:
					a[j] = a.get(j, 0.0) + w
	direct = [bool(s) for s in score]

	def best(i):
		s = score[i]
		if not s:
			return 0.0, None
		c = min(s, key=lambda x: (-s[x], x))
		return s[c], c

	attached = [False] * len(chunks)
	heap = []
	for i in range(len(chunks)):
		w, c = best(i)
		if c is not None:
			heap.append((-w, i))
	heapq.heapify(heap)
	nLeft = len(chunks)

	def attach(i, c, how):
		nonlocal nLeft
		attached[i] = True
		nLeft -= 1
		stats[how] += 1
		for v in chunks[i][1]:
			out[v] = c
		for j, w in adj[i].items():
			if not attached[j]:
				score[j][c] = score[j].get(c, 0.0) + adj[j].get(i, w)
				bw, bc = best(j)
				heapq.heappush(heap, (-bw, j))

	while nLeft > 0:
		while heap:
			negw, i = heapq.heappop(heap)
			if attached[i]:
				continue
			w, c = best(i)
			if c is None or -negw != w:         # stale entry
				continue
			attach(i, c, 'direct' if direct[i] else 'chain')
		if nLeft == 0:
			break
		# no graph connection left: attach the first remaining chunk through the tree
		i = attached.index(False)
		attach(i, _treeNearest(chunks[i][0], children, parent, out), 'tree')
	return (out, stats) if returnStats else out


def _treeNearest(root, children, parent, labels):
	a = root
	while a in parent:
		a = parent[a]
		cnt = {}
		for v in _leaves(a, children):
			c = labels.get(v, -1)
			if c != -1:
				cnt[c] = cnt.get(c, 0) + 1
		if cnt:
			return min(cnt, key=lambda x: (-cnt[x], x))
	raise ValueError('no cluster found in the tree')
