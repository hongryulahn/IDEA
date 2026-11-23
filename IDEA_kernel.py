#!/usr/bin/env python
"""Compiled (numba) kernel for hierarchicalClusteringAgent.fit at the chunk level.

It reproduces the pure-Python agent exactly (same merge order, same tie
breaking, same floating point operations in the same order).  The few cases
whose outcome depends on CPython set iteration order -- equal linkage weights
inside one candidate list, an exhausted candidate heap (disconnected chunk
graph), NaN weights or a zero divisor -- are not decided here: the kernel
returns a non-zero status and the caller re-runs the exact Python code.

Cluster ids must be 0..n0-1 for the initial clusters; merged clusters get ids
n0, n0+1, ... (which is what the Python agent produces as well).
"""
import numpy as np
from numba import njit

METHOD_CODES = {
	'average': 0,
	'weighted': 1,
	'single': 2,
	'complete': 3,
	'deltaRev': 4,
	'deltaModularity': 5,
	'minPossibleEdgeCountNormalized': 6,
	'minNodeNormalized': 7,
	'minEdgeCountNormalized': 8,
	'minEdgeWeightNormalized': 9,
	'minModularityWeightNormalized': 10,
}

# Dense (M x M) bookkeeping: only worth it -- and memory safe -- for moderately
# sized problems. Larger agents use the pure-Python implementation.
MAX_CLUSTERS = 512

STATUS_OK = 0
STATUS_TIE = 1          # equal weights in a candidate list -> order depends on set iteration
STATUS_EMPTY_HEAP = 2   # no candidate edge left (disconnected) -> original picks by set order
STATUS_NAN = 3          # NaN weight -> comparisons are order dependent
STATUS_ZERODIV = 4      # Python would raise ZeroDivisionError
STATUS_BADPREFIX = 5    # prefix refers to a dead cluster / id out of range


@njit(cache=True, inline='always')
def _less(w1, i1, w2, i2):
	# tuple order (negweight, i) as used by heapq in the Python version
	if w1 < w2:
		return True
	if w1 == w2 and i1 < i2:
		return True
	return False


@njit(cache=True)
def _heappush(hw, hi, size, w, i):
	pos = size
	hw[pos] = w
	hi[pos] = i
	while pos > 0:
		parent = (pos - 1) >> 1
		if _less(hw[pos], hi[pos], hw[parent], hi[parent]):
			hw[pos], hw[parent] = hw[parent], hw[pos]
			hi[pos], hi[parent] = hi[parent], hi[pos]
			pos = parent
		else:
			break
	return size + 1


@njit(cache=True)
def _heappop(hw, hi, size):
	w = hw[0]
	i = hi[0]
	size -= 1
	if size > 0:
		hw[0] = hw[size]
		hi[0] = hi[size]
		pos = 0
		while True:
			l = 2 * pos + 1
			if l >= size:
				break
			r = l + 1
			m = l
			if r < size and _less(hw[r], hi[r], hw[l], hi[l]):
				m = r
			if _less(hw[m], hi[m], hw[pos], hi[pos]):
				hw[pos], hw[m] = hw[m], hw[pos]
				hi[pos], hi[m] = hi[m], hi[pos]
				pos = m
			else:
				break
	return w, i, size


@njit(cache=True, error_model='numpy')
def agglomerate(method, n0, nTotalNode, nNode0, sumW0, cost0, cstat0, totalW,
				edge_a, edge_b, edge_sum, edge_aux,
				init_a, init_b, init_w,
				prefix, forb_a, forb_b):
	"""Returns (status, nMerge, merges[int64 (n0-1,3)], nn[int64 (n0-1,2)], st[float64 (n0-1,5)]).

	st columns: sumWeight1, sumWeight2, sumWeight12, sumWeight, cost
	"""
	M = 2 * n0 + 1
	nMergeMax = max(n0 - 1, 0)
	merges = np.zeros((nMergeMax, 3), np.int64)
	nn = np.zeros((nMergeMax, 2), np.int64)
	st = np.zeros((nMergeMax, 5), np.float64)

	alive = np.zeros(M, np.bool_)
	nNode = np.zeros(M, np.int64)
	sumW = np.zeros(M, np.float64)
	cost = np.zeros(M, np.float64)
	cstat = np.zeros(M, np.float64)
	for c in range(n0):
		alive[c] = True
		nNode[c] = nNode0[c]
		sumW[c] = sumW0[c]
		cost[c] = cost0[c]
		cstat[c] = cstat0[c]
	nAlive = n0

	present = np.zeros((M, M), np.bool_)
	esw = np.zeros((M, M), np.float64)
	needAux = method == 1 or method == 2 or method == 3 or method == 8
	aux = np.zeros((M, M) if needAux else (1, 1), np.float64)
	for k in range(edge_a.shape[0]):
		a = edge_a[k]
		b = edge_b[k]
		present[a, b] = True
		present[b, a] = True
		esw[a, b] = edge_sum[k]
		esw[b, a] = edge_sum[k]
		if needAux:
			aux[a, b] = edge_aux[k]
			aux[b, a] = edge_aux[k]

	hasForb = forb_a.shape[0] > 0
	forb = np.zeros((M, M) if hasForb else (1, 1), np.bool_)
	for k in range(forb_a.shape[0]):
		a = forb_a[k]
		b = forb_b[k]
		if a >= 0 and b >= 0 and a < M and b < M:
			forb[a, b] = True
			forb[b, a] = True

	# candidate lists: list j occupies pool[lst_start[j]:lst_end[j]], sorted ascending by weight
	cap = init_a.shape[0] + n0 * M + 1
	pool_a = np.empty(cap, np.int64)
	pool_b = np.empty(cap, np.int64)
	pool_w = np.empty(cap, np.float64)
	lst_start = np.zeros(n0 + 2, np.int64)
	lst_end = np.zeros(n0 + 2, np.int64)
	hw = np.empty(n0 + 2, np.float64)
	hi = np.empty(n0 + 2, np.int64)
	hsize = 0
	poolTop = 0
	max_i = 0
	nInit = init_a.shape[0]
	if nInit > 0:
		for k in range(nInit):
			if init_w[k] != init_w[k]:
				return STATUS_NAN, 0, merges, nn, st
			pool_a[k] = init_a[k]
			pool_b[k] = init_b[k]
			pool_w[k] = init_w[k]
		lst_start[0] = 0
		lst_end[0] = nInit
		poolTop = nInit
		hsize = _heappush(hw, hi, hsize, -pool_w[nInit - 1], 0)
		max_i = 1

	nMerge = 0
	nPrefix = prefix.shape[0]
	newc = 0
	maxAlive = n0 - 1
	step = 0
	while True:
		if step < nPrefix:
			c1 = prefix[step, 0]
			c2 = prefix[step, 1]
			newc = prefix[step, 2]
			replay = True
			if c1 < 0 or c2 < 0 or newc < 0 or c1 >= M or c2 >= M or newc >= M or not alive[c1] or not alive[c2] or alive[newc] or c1 == c2:
				return STATUS_BADPREFIX, nMerge, merges, nn, st
		else:
			replay = False
			if nAlive <= 1:
				break
			if step == nPrefix:
				newc = maxAlive + 1
			found = False
			while hsize > 0:
				negw, i, hsize = _heappop(hw, hi, hsize)
				lst_end[i] -= 1
				top = lst_end[i]
				c1 = pool_a[top]
				c2 = pool_b[top]
				if not (alive[c1] and alive[c2]):
					while lst_end[i] > lst_start[i]:
						lst_end[i] -= 1
						top = lst_end[i]
						if alive[pool_a[top]] and alive[pool_b[top]]:
							lst_end[i] += 1
							break
					if lst_end[i] > lst_start[i]:
						hsize = _heappush(hw, hi, hsize, -pool_w[lst_end[i] - 1], i)
					continue
				if lst_end[i] > lst_start[i]:
					hsize = _heappush(hw, hi, hsize, -pool_w[lst_end[i] - 1], i)
				if hasForb and forb[c1, c2]:
					continue
				found = True
				break
			if not found:
				return STATUS_EMPTY_HEAP, nMerge, merges, nn, st
			if newc >= M:
				return STATUS_BADPREFIX, nMerge, merges, nn, st

		# ---------------- merge c1, c2 -> newc
		n1 = nNode[c1]
		n2 = nNode[c2]
		n = n1 + n2
		s1 = sumW[c1]
		s2 = sumW[c2]
		s12 = esw[c1, c2] if present[c1, c2] else 0.0
		s = s1 + s2 + s12
		cst = cost[c1] + cost[c2] + s12 * n * n
		nNode[newc] = n
		sumW[newc] = s
		cost[newc] = cst
		merges[nMerge, 0] = c1
		merges[nMerge, 1] = c2
		merges[nMerge, 2] = newc
		nn[nMerge, 0] = n1
		nn[nMerge, 1] = n2
		st[nMerge, 0] = s1
		st[nMerge, 1] = s2
		st[nMerge, 2] = s12
		st[nMerge, 3] = s
		st[nMerge, 4] = cst
		nMerge += 1

		if method == 5 or method == 10:
			cstat[newc] = cstat[c1] + cstat[c2]
		elif method == 8:
			cstat[newc] = cstat[c1] + cstat[c2] - (aux[c1, c2] if present[c1, c2] else 0.0)
		elif method == 9:
			cstat[newc] = cstat[c1] + cstat[c2] - s12

		alive[c1] = False
		alive[c2] = False
		nAlive -= 2
		start = poolTop
		cnt = 0
		for x in range(M):
			if not alive[x]:
				continue
			p1 = present[c1, x]
			p2 = present[c2, x]
			if not (p1 or p2):
				continue
			sw = (esw[c1, x] if p1 else 0.0) + (esw[c2, x] if p2 else 0.0)
			nnew = nNode[newc]
			nx = nNode[x]
			if method == 0:
				w = sw / np.float64(nnew * nx)
			elif method == 1:
				w = 0.5 * ((aux[c1, x] if p1 else 0.0) + (aux[c2, x] if p2 else 0.0))
				aux[newc, x] = w
				aux[x, newc] = w
			elif method == 2:
				w = aux[c1, x] if p1 else 0.0
				b = aux[c2, x] if p2 else 0.0
				if b > w:
					w = b
				aux[newc, x] = w
				aux[x, newc] = w
			elif method == 3:
				w = aux[c1, x] if p1 else np.inf
				b = aux[c2, x] if p2 else np.inf
				if b < w:
					w = b
				aux[newc, x] = w
				aux[x, newc] = w
			elif method == 4:
				w = sw / np.float64(nnew * nx)
				w *= np.float64(nTotalNode - nnew - nx)
			elif method == 5:
				w = sw - cstat[newc] * cstat[x] / totalW
			elif method == 6:
				mn = nnew if nnew <= nx else nx
				d = mn * (nTotalNode - mn)
				if d == 0:
					return STATUS_ZERODIV, nMerge, merges, nn, st
				w = sw / np.float64(d)
			elif method == 7:
				mn = nnew if nnew <= nx else nx
				if mn == 0:
					return STATUS_ZERODIV, nMerge, merges, nn, st
				w = sw / np.float64(mn)
			elif method == 8:
				cw = (aux[c1, x] if p1 else 0.0) + (aux[c2, x] if p2 else 0.0)
				aux[newc, x] = cw
				aux[x, newc] = cw
				a_ = cstat[newc]
				b_ = cstat[x]
				d = b_ if b_ < a_ else a_
				if d == 0.0:
					return STATUS_ZERODIV, nMerge, merges, nn, st
				w = sw / d
			else:  # 9, 10
				a_ = cstat[newc]
				b_ = cstat[x]
				d = b_ if b_ < a_ else a_
				if d == 0.0:
					return STATUS_ZERODIV, nMerge, merges, nn, st
				w = sw / d
			if w != w:
				return STATUS_NAN, nMerge, merges, nn, st
			present[c1, x] = False
			present[x, c1] = False
			present[c2, x] = False
			present[x, c2] = False
			present[newc, x] = True
			present[x, newc] = True
			esw[newc, x] = sw
			esw[x, newc] = sw
			# pair key K(newc, x) as in Python
			if newc < x:
				pa = newc
				pb = x
			else:
				pa = x
				pb = newc
			# stable insertion into the (ascending) new list; ties -> let Python decide
			j = start + cnt
			while j > start and pool_w[j - 1] > w:
				pool_a[j] = pool_a[j - 1]
				pool_b[j] = pool_b[j - 1]
				pool_w[j] = pool_w[j - 1]
				j -= 1
			if j > start and pool_w[j - 1] == w:
				return STATUS_TIE, nMerge, merges, nn, st
			pool_a[j] = pa
			pool_b[j] = pb
			pool_w[j] = w
			cnt += 1
		if cnt > 0:
			poolTop = start + cnt
			lst_start[max_i] = start
			lst_end[max_i] = start + cnt
			hsize = _heappush(hw, hi, hsize, -pool_w[start + cnt - 1], max_i)
			max_i += 1
		alive[newc] = True
		nAlive += 1
		if newc > maxAlive:
			maxAlive = newc
		step += 1
		if not replay:
			newc += 1
			if nAlive <= 1:
				break
	return STATUS_OK, nMerge, merges, nn, st
