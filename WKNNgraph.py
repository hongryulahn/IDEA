#!/usr/bin/env python
import math
import numpy as np

def autoType(var):
	#first test bools
	if var == 'True':
		return True
	elif var == 'False':
		return False
	#int
	try:
		return int(var)
	except ValueError:
		pass
	#float
	try:
		return float(var)
	except ValueError:
		pass
	#string
	try:
		return str(var)
	except ValueError:
		raise NameError('Something Messed Up Autocasting var %s (%s)'%(var, type(var)))

_expCache = {}
def _exp(x):
	"""np.exp memoised: the argument is minus a sum of two neighbour ranks, so it
	only takes a handful of distinct values. Returns exactly np.exp(x)."""
	v = _expCache.get(x)
	if v is None:
		v = _expCache[x] = np.exp(x)
	return v

def _wnnWeights(n, rowptr, nbr, dst, pos, nNeighbor, l, descending=False):
	"""Vectorised WNN weighting shared by constructGraphByData / constructGraph.

	Rows are processed in order r = 0..n-1 exactly like the original loops
	    for n1 in nodes: for i, (n2, w) in enumerate(neighbours(n1)):
	        dic[n1][n2] = dic[n2][n1] = w; node_i[n1][n2] = i; set_edge.add(K(n1, n2))
	followed by the rank / neighbourness transform and the node-average scaling.
	The result (dict-of-dicts, including key insertion order and np.float64
	values) is identical to the original implementation:
	  * dict key order = order of first write, values = last write;
	  * edges are ranked by a stable sort of their weight; equal weights are
	    ordered by the iteration order of the original ``set_edge`` (rebuilt
	    in Python only when ties exist);
	  * (x)**2 uses Python's float pow, node averages use np.add.reduce per row
	    (== np.mean of the value list), exp is evaluated on Python scalars.
	rowptr/nbr/dst/pos: CSR-like arrays, row r = entries rowptr[r]:rowptr[r+1];
	pos holds the neighbour index i written to node_i (int or float).
	"""
	E = len(nbr)
	nbr = np.asarray(nbr, dtype=np.int64)
	dst = np.asarray(dst, dtype=np.float64)
	rowlen = np.diff(rowptr)
	row = np.repeat(np.arange(n, dtype=np.int64), rowlen)
	t = np.arange(E, dtype=np.int64)

	# outer dict order: first appearance in the sequence  r, nbr(r)...,  r+1, ...
	seq = np.empty(n + E, dtype=np.int64)
	head = rowptr[:-1] + np.arange(n)
	seq[head] = np.arange(n)
	seq[t + row + 1] = nbr
	_, first = np.unique(seq, return_index=True)
	uniqNodes = np.unique(seq)
	outerOrder = uniqNodes[np.argsort(first, kind='stable')]
	nNode = len(outerOrder)

	# directed writes dic[v][u] = d at time t (both directions)
	v = np.concatenate([row, nbr])
	u = np.concatenate([nbr, row])
	tt = np.concatenate([t, t])
	dd = np.concatenate([dst, dst])
	key = v * n + u
	o = np.lexsort((tt, key))
	ks = key[o]
	start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
	end = np.r_[start[1:], len(ks)] - 1
	dkey = ks[start]                 # unique directed keys (sorted)
	dfirst = tt[o][start]            # time of first write -> key order in dict v
	dlast = dd[o][end]               # value of last write
	dv, du = dkey // n, dkey % n

	# undirected edges
	lo, hi = np.minimum(row, nbr), np.maximum(row, nbr)
	ekey = np.unique(lo * n + hi)
	nEdge = len(ekey)
	elo, ehi = ekey // n, ekey % n
	ew = dlast[np.searchsorted(dkey, elo * n + ehi)]       # dic[lo][hi]

	# stable sort by weight; ties broken by original set iteration order
	sortw = -ew if descending else ew
	order = np.argsort(sortw, kind='stable')
	sw = sortw[order]
	if nEdge > 1 and np.any(sw[1:] == sw[:-1]):
		setEdge = set()
		r_ = row.tolist(); nb_ = nbr.tolist()
		for a, b in zip(r_, nb_):
			setEdge.add((a, b) if a < b else (b, a))
		setPos = np.empty(nEdge, dtype=np.int64)
		setPos[np.searchsorted(ekey, np.array([a * n + b for a, b in setEdge], dtype=np.int64))] = np.arange(nEdge)
		order = np.lexsort((setPos, sortw))
	rank = np.empty(nEdge, dtype=np.int64)
	rank[order] = np.arange(nEdge)

	# neighbourness: last index written for node_i[a][b]; default (nNeighbor+nNode)*0.5
	pkey = row * n + nbr
	po = np.lexsort((t, pkey))
	pks = pkey[po]
	plast = np.r_[pks[1:] != pks[:-1], True]
	pkeyU = pks[plast]
	posU = np.asarray(pos, dtype=object)[po][plast] if E else np.zeros(0, dtype=object)
	default = (nNeighbor + nNode) * 0.5
	def lookup(a, b):
		q = a * n + b
		idx = np.searchsorted(pkeyU, q)
		idx[idx >= len(pkeyU)] = 0
		found = (len(pkeyU) > 0) & (pkeyU[idx] == q) if len(pkeyU) else np.zeros(len(q), dtype=bool)
		out = np.empty(len(q), dtype=object)
		out[:] = default
		out[found] = posU[idx[found]]
		return out
	i1 = lookup(elo, ehi)
	i2 = lookup(ehi, elo)
	ex = {}
	expv = np.empty(nEdge, dtype=np.float64)
	for k, (a, b) in enumerate(zip(i1.tolist(), i2.tolist())):
		x = -a - b
		val = ex.get(x)
		if val is None:
			val = ex[x] = np.exp(x)
		expv[k] = val
	q = (nEdge - rank).astype(np.float64) / nEdge
	q2 = np.array([x**2 for x in q.tolist()], dtype=np.float64)
	newVal = np.sqrt(l*q2 + (1.0-l)*expv)

	# dict layout: rows in outer order, keys in first-write order
	lay = np.lexsort((dfirst, dv))
	lv, lu = dv[lay], du[lay]
	lval = newVal[np.searchsorted(ekey, np.minimum(lv, lu) * n + np.maximum(lv, lu))]
	cnt = np.bincount(lv, minlength=n)
	ptr = np.r_[0, np.cumsum(cnt)]
	avg = np.zeros(n, dtype=np.float64)
	add = np.add.reduce
	for r in range(n):
		if cnt[r]:
			avg[r] = add(lval[ptr[r]:ptr[r+1]]) / cnt[r]
	scaled = lval * np.minimum(avg[lv], avg[lu])
	final = np.where(lv == lu, lval, scaled)

	lu_list = lu.tolist()
	out = {}
	for r in outerOrder.tolist():
		a, b = ptr[r], ptr[r+1]
		out[r] = dict(zip(lu_list[a:b], final[a:b]))
	return out

class WKNNgraph:
	def __init__(self):
		self.dic_node2name=None
		self.dic_node2node_weight=None
		self.data=None

	def getName(self):
		lst_name = [self.dic_node2name[node] for node in sorted(self.dic_node2name.keys())]
		return lst_name

	def loadGraph(self, filename):
		set_name = set()
		IF=open(filename,'r')
		for line in IF:
			s=line.rstrip().split('\t')
			name1,name2=s[0],s[1]
			set_name.add(name1)
			set_name.add(name2)
		dic_name2node={}
		self.dic_node2name={}
		self.dic_node2node_weight={}
		for node, name in enumerate(sorted(set_name, key=autoType)):
			dic_name2node[name]=node
			self.dic_node2name[node]=name
			self.dic_node2node_weight[node]={}
		IF.seek(0)
		for line in IF:
			s=line.rstrip().split('\t')
			name1,name2=s[0],s[1]
			if len(s) == 3:
				weight=float(s[2])
			elif len(s) == 2:
				weight=1.0
			n1, n2 = dic_name2node[name1], dic_name2node[name2]
			if n1 not in self.dic_node2node_weight:
				self.dic_node2node_weight[n1]={}
			if n2 not in self.dic_node2node_weight:
				self.dic_node2node_weight[n2]={}
			self.dic_node2node_weight[n1][n2]=weight
			self.dic_node2node_weight[n2][n1]=weight
		IF.close()

	def constructGraphByData(self, filename, metric='euclidean', nNeighbor=10, l=0.5):
		IF=open(filename,'r')
		self.dic_node2name={}
		lst_point=[]
		for node, line in enumerate(IF):
			s=line.strip().split()
			name, point = s[0], list(map(float,s[1:]))
			self.dic_node2name[node]=name
			lst_point.append(point)
		IF.close()
		X=np.array(lst_point)
		self.data = X
		from sklearn.neighbors import NearestNeighbors
		tree = NearestNeighbors(n_neighbors=nNeighbor+1, algorithm='kd_tree', metric=metric)
		tree.fit(X)
		dist, ind = tree.kneighbors(X)
		n = len(self.dic_node2name)
		k = ind.shape[1]-1
		rowptr = np.arange(n+1, dtype=np.int64)*k
		pos = list(range(k))*n
		self.dic_node2node_weight = _wnnWeights(n, rowptr, ind[:,1:].ravel(), dist[:,1:].ravel(), pos, nNeighbor, l)

	def constructGraph(self, filename, weightType, nNeighbor=10, l=0.5):
		assert(weightType in ['similarity','dissimilarity']),(
			"weightType must be in ['similarity','dissimilarity']")
		IF=open(filename,'r')
		dic_name2neighbor={}
		if weightType=='dissimilarity':
			for line in IF:
				s=line.rstrip().split('\t')
				name1,name2,weight=s[0],s[1],float(s[2])
				if name1 not in dic_name2neighbor:
					dic_name2neighbor[name1]=[]
				if name2 not in dic_name2neighbor:
					dic_name2neighbor[name2]=[]
				if len(dic_name2neighbor[name1]) < nNeighbor or dic_name2neighbor[name1][-1][1] > weight:
					dic_name2neighbor[name1].append((name2,weight))
					dic_name2neighbor[name1].sort(key=lambda x:x[1])
					if len(dic_name2neighbor[name1]) > nNeighbor:
						del dic_name2neighbor[name1][-1]
				if len(dic_name2neighbor[name2]) < nNeighbor or dic_name2neighbor[name2][-1][1] > weight:
					dic_name2neighbor[name2].append((name1,weight))
					dic_name2neighbor[name2].sort(key=lambda x:x[1])
					if len(dic_name2neighbor[name2]) > nNeighbor:
						del dic_name2neighbor[name2][-1]
		if weightType=='similarity':
			for line in IF:
				s=line.rstrip().split('\t')
				name1,name2,weight=s[0],s[1],float(s[2])
				if name1 not in dic_name2neighbor:
					dic_name2neighbor[name1]=[]
				if name2 not in dic_name2neighbor:
					dic_name2neighbor[name2]=[]
				if len(dic_name2neighbor[name1]) <= nNeighbor or dic_name2neighbor[name1][-1][1] < weight:
					dic_name2neighbor[name1].append((name2,weight))
					dic_name2neighbor[name1].sort(key=lambda x:x[1],reverse=True)
					if len(dic_name2neighbor[name1]) > nNeighbor:
						del dic_name2neighbor[name1][-1]
				if len(dic_name2neighbor[name2]) <= nNeighbor or dic_name2neighbor[name2][-1][1] < weight:
					dic_name2neighbor[name2].append((name1,weight))
					dic_name2neighbor[name2].sort(key=lambda x:x[1],reverse=True)
					if len(dic_name2neighbor[name2]) > nNeighbor:
						del dic_name2neighbor[name2][-1]
		IF.close()
		dic_name2node={}
		self.dic_node2name={}
		for node, name in enumerate(sorted(dic_name2neighbor.keys(),key=autoType)):
			self.dic_node2name[node]=name
			dic_name2node[name]=node
		n = len(self.dic_node2name)
		rowptr=[0]; nbr=[]; dst=[]; pos=[]
		for n1 in range(n):
			for i, (name2, weight) in enumerate(dic_name2neighbor[self.dic_node2name[n1]]):
				nbr.append(dic_name2node[name2]); dst.append(weight); pos.append(float(i))
			rowptr.append(len(nbr))
		self.dic_node2node_weight = _wnnWeights(n, np.array(rowptr, dtype=np.int64), nbr, dst, pos, nNeighbor, l, descending=(weightType=='similarity'))

	def write(self, filename):
		OF=open(filename,'w')
		for n1 in self.dic_node2name.keys():
			for n2, weight in self.dic_node2node_weight[n1].items():
				name1, name2 = self.dic_node2name[n1], self.dic_node2name[n2]
				if name1 >= name2:
					continue
				OF.write('\t'.join(list(map(str,[name1,name2,weight])))+'\n')
		OF.close()

	def toMatrix(self,weightType='similarity'):
		nNode=len(self.dic_node2name.keys())
		lst_node = sorted(self.dic_node2name.keys())
		weightMatrix=np.zeros((nNode,nNode))
		for n1 in lst_node:
			if weightType == 'similarity':
				weightMatrix[n1,n1] = 1.0
			for n2, weight in self.dic_node2node_weight[n1].items():
				weightMatrix[n1,n2]=weight
		if min(lst_node) == 0 and max(lst_node) == nNode-1:
			return weightMatrix
		else:
			lst_name = [self.dic_node2name[node] for node in lst_node]
			return weightMatrix, lst_name

	def conversion(self, method):
		assert(method in ['Cauchy','generalizedGaussian','FermiDirac']),(
			"method must be in ['Cauchy','generalizedGaussian','FermiDirac']")
		for n1 in self.dic_node2node_weight:
			for n2, weight in self.dic_node2node_weight[n1].items():
				if method == 'Cauchy':
					weight = 1.0/(1.0+weight)
				elif method == 'generalizedGaussian':
					weight = np.exp(-weight)
				elif method == 'FermiDirac':
					weight = 1.0/(1.0+np.exp(weight))
	
	def subGraph(self, set_node):
		sG = WKNNgraph()
		sG.dic_node2name={}
		sG.dic_node2node_weight={}
		sG.data = self.data
		for n1 in set_node:
			sG.dic_node2name[n1]=self.dic_node2name[n1]
			sG.dic_node2node_weight[n1]={}
			for n2, weight in self.dic_node2node_weight[n1].items():
				if n2 not in set_node:
					continue
				sG.dic_node2node_weight[n1][n2]=weight
		return sG
	
if __name__ == "__main__":
	import argparse
	parser=argparse.ArgumentParser(
		usage='''\
	%(prog)s [options] graphfile k
	example: %(prog)s graphfile k -o out.txt
	''')
	
	parser.add_argument('graphfile', help='distance or weight graph file')
	parser.add_argument('-weightType', required=False, default='graph', choices=['graph', 'similarity', 'dissimilarity'], help='weightType')
	parser.add_argument('-K', required=False, type=int, default=10, help='K-nearest neighbor graph')
	parser.add_argument('-conversionMethod', required=False, default=None, choices=['Cauchy', 'generalizedGaussian', 'FermiDirac'], help='weightType')
	parser.add_argument('-l', required=False, type=float, default=0.5, help='rank vs. neighborness')
	parser.add_argument('-o', dest='outfile', required=False, metavar='str', default=None, help='outfile')
	args=parser.parse_args()
	
	wKNNgraph=WKNNgraph()
	if args.weightType == 'graph':
		wKNNgraph.loadGraph(args.graphfile)
	else:
		wKNNgraph.constructGraph(args.graphfile,args.weightType,args.K, args.l)
	if args.conversionMethod != None:
		wKNNgraph.conversion(args.conversionMethod)
	if args.outfile != None:
		wKNNgraph.write(args.outfile)
