#!/usr/bin/env python
import sys
import os
import argparse
import numpy as np
import itertools
import time
import random
import heapq
import operator
from WKNNgraph import *

# Optional compiled kernel for the chunk-level agents (falls back to pure Python).
# Set IDEA_NO_NUMBA=1 to force the pure Python implementation.
_kernel = None
if os.environ.get('IDEA_NO_NUMBA', '0') in ('', '0'):
	try:
		import IDEA_kernel as _kernel
	except ImportError:
		_kernel = None

def cutTree(linkageMatrix, t, outlierRatio=None, maxLeafID=None):
	dic_parent2childs={}
	dic_child2parent={}
	dic_node2weight={}
	dic_node2nNode={}
	dic_node2averageWeight={}
	set_leaf=set()
	if maxLeafID == None:
		maxLeafID = len(linkageMatrix)
	for c, x in enumerate(linkageMatrix,maxLeafID+1):
		c1, c2, w, nNode = int(x[0]), int(x[1]), x[2], x[3]
		if c1 not in dic_parent2childs:
			set_leaf.add(c1)
		if c2 not in dic_parent2childs:
			set_leaf.add(c2)
		dic_child2parent[c1]=c
		dic_child2parent[c2]=c
		dic_parent2childs[c]={c1,c2}
		dic_node2weight[c]=w
		dic_node2nNode[c]=nNode
		root=c
		
	set_root={root}
	set_outlierRoot=set()
	dic_root2name={root:'0'}
	while len(set_root) < t:
		target = sorted(set_root, key=lambda x:(dic_node2weight.get(x,-np.inf),dic_node2nNode.get(x,1.0)),reverse=True)[0]
		set_root.remove(target)
		c1, c2 = dic_parent2childs[target]
		if dic_node2nNode.get(c1,1.0) < dic_node2nNode.get(c2,1.0):
			c1, c2 = c2, c1
		dic_root2name[c1] = dic_root2name[target]+'0'
		dic_root2name[c2] = dic_root2name[target]+'1'
		set_root.add(c1)
		set_root.add(c2)
		if outlierRatio != None and len(set_root) >= t:
			lst_root = sorted(set_root, key=lambda x:dic_node2nNode.get(x,1.0),reverse=True)
			if dic_node2nNode.get(lst_root[-1],1.0) < dic_node2nNode.get(lst_root[0],1.0)*outlierRatio:
				set_outlierRoot.add(lst_root[-1])
				set_root.remove(lst_root[-1])
				continue

	set_root2=set()
	while True:
		if len(set_root) == 0:
			break
		root = set_root.pop()
		set_outlierRoot2=set()
		while True:
			if len(set_outlierRoot) == 0:
				break
			outlierRoot=set_outlierRoot.pop()
			if dic_child2parent[root]==dic_child2parent[outlierRoot]:
				root = dic_child2parent[root]
				set_outlierRoot |= set_outlierRoot2
				set_outlierRoot2 = set()
			else:
				set_outlierRoot2.add(outlierRoot)
		set_root2.add(root)
		set_outlierRoot = set_outlierRoot2
	set_root = set_root2

	dic_node2cluster={}
	for cluster, root in enumerate(sorted(set_root, key=lambda x:dic_root2name[x])):
		set_child = {root}
		while True:
			if len(set_child) == 0:
				break
			target = set_child.pop()
			if target in set_leaf:
				dic_node2cluster[target] = cluster
				continue
			c1, c2 = dic_parent2childs[target]
			if c1 in set_leaf:
				dic_node2cluster[c1]= cluster
			else:
				set_child.add(c1)
			if c2 in set_leaf:
				dic_node2cluster[c2]= cluster
			else:
				set_child.add(c2)

	for root in set_outlierRoot:
		cluster = -1
		set_child = {root}
		while True:
			if len(set_child) == 0:
				break
			target = set_child.pop()
			if target in set_leaf:
				dic_node2cluster[target] = cluster
				continue
			c1, c2 = dic_parent2childs[target]
			if c1 in set_leaf:
				dic_node2cluster[c1]= cluster
			else:
				set_child.add(c1)
			if c2 in set_leaf:
				dic_node2cluster[c2]= cluster
			else:
				set_child.add(c2)
	return dic_node2cluster

def computeSeparationGain(dic_key2value):
	w11,w22,w12 = dic_key2value['sumWeight1'],dic_key2value['sumWeight2'],dic_key2value['sumWeight12']
	n1, n2, = dic_key2value['nNode1'],dic_key2value['nNode2']
	if w12 == 0:
		return 1.0 - 0.1**20 * float(n1+n2)/(2*n1*n2) * 1.0/(-n1/float(n1+n2)*np.log2(n1/float(n1+n2))-n2/float(n1+n2)*np.log2(n2/float(n1+n2)))
	a= w12/(n1*n2)
	return 1.0 - a

def K(c1,c2):
	if c1 < c2:
		return (c1,c2)
	else:
		return (c2,c1)

_second = operator.itemgetter(1)
_EMPTY_I64 = np.zeros(0, dtype=np.int64)

def _hasTie(lst):
	"""True if a weight-sorted [pair, weight] list has equal (or NaN) neighbouring weights."""
	prev = lst[0][1]
	if prev != prev:
		return True
	for j in range(1, len(lst)):
		w = lst[j][1]
		if w == prev or w != w:
			return True
		prev = w
	return False

def _samePrefix(lm1, lm2, n):
	"""True if the first n merges (c1, c2, new_cluster) of two linkage lists agree."""
	for j in range(n):
		a, b = lm1[j], lm2[j]
		if a[0] != b[0] or a[1] != b[1] or a[2] != b[2]:
			return False
	return True

def delete(dic, key):
	if key in dic:
		del dic[key]


def connectedComponent(wKNNgraph,set_node=None):
	if set_node == None:
		set_node = set(wKNNgraph.dic_node2node_weight.keys())
	else:
		set_node = set(set_node)
	lst_connectedComponent=[]
	while True:
		if len(set_node) == 0:
			break
		pivot = set_node.pop()
		set_visited=set()
		set_unvisited=set()
		set_unvisited.add(pivot)
		while True:
			if len(set_unvisited)==0:
				break
			pivot = set_unvisited.pop()
			set_visited.add(pivot)
			for n2, weight in wKNNgraph.dic_node2node_weight[pivot].items():
				if n2 in set_node:
					if n2 in set_node and n2 not in set_visited:
						set_unvisited.add(n2)
		lst_connectedComponent.append(set_visited)
		set_node -= set_visited
	return lst_connectedComponent

def _reassignShatteredNodes(set_shatteredNode, dic_node2cluster, dic_cluster2nodes, dic_node2node_weight):
	"""Greedily give each shattered node (cluster -1) to its strongest neighbouring cluster.

	Equivalent to the original loop
		while lst: lst.sort(key=weightSum, reverse=True); node = lst.pop(0); ...
	(same pick order, including ties), but uses a heap instead of re-sorting the
	whole list after every pick.  The stable re-sort orders equal keys by their
	previous list position; this is reproduced with a per-node stamp: a node whose
	key increased moves behind every node that already has the new key, a node
	whose key decreased moves in front of them.
	"""
	dic_node2key = {}
	dic_node2c2w = {}
	for node in set_shatteredNode:
		dic_c2w = {}
		for n2, weight in dic_node2node_weight[node].items():
			c2 = dic_node2cluster[n2]
			if c2 == -1:
				continue
			if c2 not in dic_c2w:
				dic_c2w[c2] = 0.0
			dic_c2w[c2] += weight
		dic_node2key[node] = max(dic_c2w.values()) if len(dic_c2w) > 0 else 0.0
		dic_node2c2w[node] = dic_c2w
	lst_node = list(set_shatteredNode)
	dic_node2stamp = dict((node, i) for i, node in enumerate(lst_node))
	heap = [(-dic_node2key[node], i, node) for i, node in enumerate(lst_node)]
	heapq.heapify(heap)
	hiStamp, loStamp = len(lst_node), -1
	nRemaining = len(lst_node)
	heappop, heappush = heapq.heappop, heapq.heappush
	while nRemaining > 0:
		negkey, stamp, node = heappop(heap)
		if dic_node2stamp[node] != stamp:
			continue
		dic_node2stamp[node] = None
		nRemaining -= 1
		c = sorted(dic_node2c2w[node].items(), key=lambda x:x[1], reverse=True)[0][0]
		dic_cluster2nodes[c].add(node)
		dic_node2cluster[node] = c
		changed = []
		for n2, weight in dic_node2node_weight[node].items():
			if dic_node2cluster[n2] != -1:
				continue
			dic_c2w = dic_node2c2w[n2]
			if c not in dic_c2w:
				dic_c2w[c] = 0.0
			dic_c2w[c] += weight
			newKey = max(dic_c2w.values())
			oldKey = dic_node2key[n2]
			if newKey != oldKey:
				changed.append((-oldKey, dic_node2stamp[n2], n2, newKey))
				dic_node2key[n2] = newKey
		if changed:
			changed.sort()  # previous list order
			for negOld, stamp, n2, newKey in changed:
				if newKey > -negOld:
					dic_node2stamp[n2] = hiStamp
					hiStamp += 1
					heappush(heap, (-newKey, dic_node2stamp[n2], n2))
			for negOld, stamp, n2, newKey in reversed(changed):
				if not newKey > -negOld:
					dic_node2stamp[n2] = loStamp
					loStamp -= 1
					heappush(heap, (-newKey, dic_node2stamp[n2], n2))

def partitionByMetis(lst_node, nPartition, wKNNgraph, isExact=False, recursive=True):
	import pymetis
	adjncy=[]
	xadj=[]
	adjwgt=[]
	index=0
			
	if type(lst_node) == set:
		lst_node=list(lst_node)
	dic_node2i = dict([(node,i) for i, node in enumerate(lst_node)])
				
	for i1, n1 in enumerate(lst_node):
		xadj.append(index)
		for n2, weight in wKNNgraph.dic_node2node_weight[n1].items():
			if n2 not in dic_node2i:
				continue
			i2=dic_node2i[n2]
			intweight = int(weight*1000)
			if intweight <= 0:
				continue
			adjncy.append(i2)
			adjwgt.append(intweight)
			index+=1
	xadj.append(len(adjncy))
	if hasattr(pymetis, 'CSRAdjacency'):
		cutcount, tmplst_cluster = pymetis.part_graph(nPartition, pymetis.CSRAdjacency(xadj, adjncy), eweights=adjwgt, recursive=recursive)
	else:  # older pymetis
		cutcount, tmplst_cluster = pymetis.part_graph(nPartition, xadj=xadj, adjncy=adjncy, eweights=adjwgt, recursive=recursive)
	dic_cluster2nodes={}
	for i, node in enumerate(lst_node):
		cluster = tmplst_cluster[i]
		if cluster not in dic_cluster2nodes:
			dic_cluster2nodes[cluster]=set()
		dic_cluster2nodes[cluster].add(node)
	lst_clusters=[]
	for cluster, set_node in dic_cluster2nodes.items():
		if isExact:
			lst_clusters.append(set_node)
		else:
			lst_clusters += connectedComponent(wKNNgraph,set_node)
	return lst_clusters

def arrangeLinkageMatrix(linkageMatrix):
	nNode = len(linkageMatrix)+1
	dic_oldID2newID=dict([(i, i) for i in range(nNode)])
	newLinkageMatrix=[]
	sortedLinkageMatrix = sorted(linkageMatrix,key=lambda x:x[3].get('nNode',1.0))
	for c, x in enumerate(sortedLinkageMatrix,nNode):
		dic_oldID2newID[x[2]]=c
	for c, x in enumerate(sortedLinkageMatrix,nNode):
		c1 = dic_oldID2newID[x[0]]
		c2 = dic_oldID2newID[x[1]]
		newLinkageMatrix.append([c1,c2,x[3].get('separationGain',0.0),x[3].get('nNode',1.0)])
	return newLinkageMatrix
	
class hierarchicalClusteringAgent:
	def __init__(self, set_node, linkageMethod, dic_edge2maxWeight=None, dic_edge2minWeight=None, dic_edge2countWeight=None, dic_edge2sumWeight=None, dic_node2n={}, dic_node2countWeight={}, dic_node2sumWeight={}, nTotalNode=None):
		self.linkageMethod = linkageMethod
		self.set_cluster=set(set_node)

		self.dic_cluster2nNode = dict(dic_node2n)
		self.dic_cluster2countWeight = dict(dic_node2countWeight)
		self.dic_cluster2sumWeight = dict(dic_node2sumWeight)

		self.nNode = sum([self.dic_cluster2nNode.get(cluster,1.0) for cluster in self.set_cluster])
		self.costFunction = lambda weightSum, nNode: weightSum*nNode*nNode
		self.nTotalNode = nTotalNode
		self.cachePrefix = False
		self.useKernel = False
		self._kernelData = None
		self._lm = None
		self._lmRaw = None
		self.dic_cluster2cost = {}
		for cluster in self.set_cluster:
			self.dic_cluster2cost[cluster] = self.costFunction(self.dic_cluster2sumWeight.get(cluster,0.0),self.dic_cluster2nNode.get(cluster,1.0))
		self.dic_edge2sumWeight = dict(dic_edge2sumWeight)
		if linkageMethod == 'single':
			assert(dic_edge2maxWeight != None and dic_edge2sumWeight != None),(
				"single method has dic_edge2maxWeight, dic_edge2sumWeight")
			self.dic_edge2maxWeight = dict(dic_edge2maxWeight)
			self.sorted_edge=sorted(self.dic_edge2maxWeight.items(),key=lambda x:x[1])
		elif linkageMethod == 'complete':
			assert(dic_edge2minWeight != None and dic_edge2sumWeight != None),(
				"complete method has dic_edge2minWeight, dic_edge2sumWeight")
			self.dic_edge2minWeight = dict(dic_edge2minWeight)
			self.sorted_edge=sorted(self.dic_edge2minWeight.items(),key=lambda x:x[1])
		elif linkageMethod == 'weighted':
			assert(dic_edge2sumWeight != None),(
				"weighted method has dic_edge2sumWeight")
			self.sorted_edge=[]
			self.dic_edge2weightedWeight = {}
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= float(self.dic_cluster2nNode.get(c1,1.0)*self.dic_cluster2nNode.get(c2,1.0))
				self.dic_edge2weightedWeight[(c1,c2)] = weight
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'average':
			assert(dic_edge2sumWeight != None),(
				"average method has dic_edge2sumWeight")
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= float(self.dic_cluster2nNode.get(c1,1.0)*self.dic_cluster2nNode.get(c2,1.0))
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'deltaCost':
			assert(dic_edge2sumWeight != None),(
				"deltaCost method has dic_edge2sumWeight")
			self.dic_cluster2edgeSumWeight = {}
			for cluster in self.set_cluster:
				self.dic_cluster2edgeSumWeight[cluster]=0.0
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				self.dic_cluster2edgeSumWeight[c1] += weight
				self.dic_cluster2edgeSumWeight[c2] += weight
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight = 1.0 / (self.dic_cluster2sumWeight.get(c1,0.0) * self.dic_cluster2nNode.get(c2,1.0) + self.dic_cluster2sumWeight.get(c2,0.0) * self.dic_cluster2nNode.get(c1,1.0) - weight * float(self.dic_cluster2nNode.get(c1,1.0)*self.dic_cluster2nNode.get(c2,1.0)))
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'deltaRev':
			assert(dic_edge2sumWeight != None),(
				"deltaRev method has dic_edge2sumWeight")
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= float(self.dic_cluster2nNode.get(c1,1.0)*self.dic_cluster2nNode.get(c2,1.0))
				weight *= self.nTotalNode-self.dic_cluster2nNode.get(c1,1.0)-self.dic_cluster2nNode.get(c2,1.0)
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'deltaModularity':
			assert(dic_edge2sumWeight != None),(
				"deltaModularity method has dic_edge2sumWeight")
			self.dic_cluster2modularitySumWeight = {}
			self.totalEdgeSumWeight=0.0
			for cluster in self.set_cluster:
				weight = self.dic_cluster2sumWeight.get(cluster,0.0)
				self.dic_cluster2modularitySumWeight[cluster] = 2.0*weight
				self.totalEdgeSumWeight+=2.0*weight
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				self.dic_cluster2modularitySumWeight[c1] += weight
				self.dic_cluster2modularitySumWeight[c2] += weight
				self.totalEdgeSumWeight+=weight
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight -= self.dic_cluster2modularitySumWeight.get(c1,0.0)*self.dic_cluster2modularitySumWeight.get(c2,0.0)/self.totalEdgeSumWeight
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'edgeCountNormalized':
			assert(dic_edge2countWeight != None and dic_edge2sumWeight != None),(
				"edgeCountNormalized method has dic_edge2countWeight, dic_edge2sumWeight")
			self.dic_edge2countWeight = dict(dic_edge2countWeight)
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= self.dic_edge2countWeight.get(K(c1,c2),0.0)
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'minEdgeCountNormalized':
			assert(dic_edge2countWeight != None and dic_edge2sumWeight != None),(
				"minEdgeCountNormalized method has dic_edge2countWeight, dic_edge2sumWeight")
			self.dic_edge2countWeight = dict(dic_edge2countWeight)
			self.dic_cluster2edgeSumCount = {}
			for cluster in self.set_cluster:
				self.dic_cluster2edgeSumCount[cluster]=0.0
			for (c1,c2),weight in self.dic_edge2countWeight.items():
				self.dic_cluster2edgeSumCount[c1] += weight
				self.dic_cluster2edgeSumCount[c2] += weight
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= min(self.dic_cluster2edgeSumCount.get(c1,0.0), self.dic_cluster2edgeSumCount.get(c2,0.0))
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'maxEdgeCountNormalized':
			assert(dic_edge2countWeight != None and dic_edge2sumWeight != None),(
				"maxEdgeCountNormalized method has dic_edge2countWeight, dic_edge2sumWeight")
			self.dic_edge2countWeight = dict(dic_edge2countWeight)
			self.dic_cluster2edgeSumCount = {}
			for cluster in self.set_cluster:
				self.dic_cluster2edgeSumCount[cluster]=0.0
			for (c1,c2),weight in self.dic_edge2countWeight.items():
				self.dic_cluster2edgeSumCount[c1] += weight
				self.dic_cluster2edgeSumCount[c2] += weight
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= max(self.dic_cluster2edgeSumCount.get(c1,0.0), self.dic_cluster2edgeSumCount.get(c2,0.0))
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'unionEdgeCountNormalized':
			assert(dic_edge2countWeight != None and dic_edge2sumWeight != None),(
				"unionEdgeCountNormalized method has dic_edge2countWeight, dic_edge2sumWeight")
			self.dic_edge2countWeight = dict(dic_edge2countWeight)
			self.dic_cluster2edgeSumCount = {}
			for cluster in self.set_cluster:
				self.dic_cluster2edgeSumCount[cluster]=0.0
			for (c1,c2),weight in self.dic_edge2countWeight.items():
				self.dic_cluster2edgeSumCount[c1] += weight
				self.dic_cluster2edgeSumCount[c2] += weight
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= self.dic_cluster2edgeSumCount.get(c1,0.0) + self.dic_cluster2edgeSumCount.get(c2,0.0) - self.dic_edge2countWeight.get(K(c1,c2),0.0)
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'minEdgeWeightNormalized':
			assert(dic_edge2sumWeight != None),(
				"minEdgeWeightNormalized method has dic_edge2sumWeight")
			self.dic_cluster2edgeSumWeight = {}
			for cluster in self.set_cluster:
				self.dic_cluster2edgeSumWeight[cluster]=0.0
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				self.dic_cluster2edgeSumWeight[c1] += weight
				self.dic_cluster2edgeSumWeight[c2] += weight
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= max(self.dic_cluster2edgeSumWeight.get(c1,0.0), self.dic_cluster2edgeSumWeight.get(c2,0.0))
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'maxEdgeWeightNormalized':
			assert(dic_edge2sumWeight != None),(
				"maxEdgeWeightNormalized method has dic_edge2sumWeight")
			self.dic_cluster2edgeSumWeight = {}
			for cluster in self.set_cluster:
				self.dic_cluster2edgeSumWeight[cluster]=0.0
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				self.dic_cluster2edgeSumWeight[c1] += weight
				self.dic_cluster2edgeSumWeight[c2] += weight
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= max(self.dic_cluster2edgeSumWeight.get(c1,0.0), self.dic_cluster2edgeSumWeight.get(c2,0.0))
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'unionEdgeWeightNormalized':
			assert(dic_edge2sumWeight != None),(
				"unionEdgeWeightNormalized method has dic_edge2sumWeight")
			self.dic_cluster2edgeSumWeight = {}
			for cluster in self.set_cluster:
				self.dic_cluster2edgeSumWeight[cluster]=0.0
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				self.dic_cluster2edgeSumWeight[c1] += weight
				self.dic_cluster2edgeSumWeight[c2] += weight
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= self.dic_cluster2edgeSumWeight.get(c1,0.0) + self.dic_cluster2edgeSumWeight.get(c2,0.0) - weight
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'minModularityWeightNormalized':
			assert(dic_edge2sumWeight != None),(
				"minModularityWeightNormalized method has dic_edge2sumWeight")
			self.dic_cluster2modularitySumWeight = {}
			for cluster in self.set_cluster:
				self.dic_cluster2modularitySumWeight[cluster] = 2.0*self.dic_cluster2sumWeight.get(cluster,0.0)
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				self.dic_cluster2modularitySumWeight[c1] += weight
				self.dic_cluster2modularitySumWeight[c2] += weight
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= min(self.dic_cluster2modularitySumWeight.get(c1,0.0),self.dic_cluster2modularitySumWeight.get(c2,0.0))
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'maxModularityWeightNormalized':
			assert(dic_edge2sumWeight != None),(
				"maxModularityWeightNormalized method has dic_edge2sumWeight")
			self.dic_cluster2modularitySumWeight = {}
			for cluster in self.set_cluster:
				self.dic_cluster2modularitySumWeight[cluster] = 2.0*self.dic_cluster2sumWeight.get(cluster,0.0)
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				self.dic_cluster2modularitySumWeight[c1] += weight
				self.dic_cluster2modularitySumWeight[c2] += weight
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= max(self.dic_cluster2modularitySumWeight.get(c1,0.0),self.dic_cluster2modularitySumWeight.get(c2,0.0))
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'avgModularityWeightNormalized':
			assert(dic_edge2sumWeight != None),(
				"avgModularityWeightNormalized method has dic_edge2sumWeight")
			self.dic_cluster2modularitySumWeight = {}
			for cluster in self.set_cluster:
				self.dic_cluster2modularitySumWeight[cluster] = 2.0*self.dic_cluster2sumWeight.get(cluster,0.0)
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				self.dic_cluster2modularitySumWeight[c1] += weight
				self.dic_cluster2modularitySumWeight[c2] += weight
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				weight /= 0.5*(self.dic_cluster2modularitySumWeight.get(c1,0.0)+self.dic_cluster2modularitySumWeight.get(c2,0.0))
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'minPossibleEdgeCountNormalized':
			assert(dic_edge2sumWeight != None),(
				"minPossibleEdgeCountNormalized has dic_edge2sumWeight")
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				minNNode = min(self.dic_cluster2nNode.get(c1,1.0),self.dic_cluster2nNode.get(c2,1.0))
				weight /= minNNode*(nTotalNode-minNNode)
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])
		elif linkageMethod == 'minNodeNormalized':
			assert(dic_edge2sumWeight != None),(
				"minNodeNormalized method has dic_edge2sumWeight")
			self.sorted_edge=[]
			for (c1,c2),weight in self.dic_edge2sumWeight.items():
				minNNode = min(self.dic_cluster2nNode.get(c1,1.0),self.dic_cluster2nNode.get(c2,1.0))
				weight /= minNNode
				self.sorted_edge.append([(c1,c2),weight])
			self.sorted_edge.sort(key=lambda x:x[1])

	def _initState(self):
		# Adjacency (cluster -> set of clusters sharing an edge) derived from the
		# edge dictionary. Lets each merge visit only the neighbours of the merged
		# pair instead of every live cluster.
		nbrs = {}
		for (a, b) in self.dic_edge2sumWeight:
			if a in nbrs:
				nbrs[a].add(b)
			else:
				nbrs[a] = {b}
			if b in nbrs:
				nbrs[b].add(a)
			else:
				nbrs[b] = {a}
		self.dic_cluster2nbrs = nbrs

	# linkageMatrix is materialised lazily when it comes from the compiled kernel
	@property
	def linkageMatrix(self):
		if self._lm is None and self._lmRaw is not None:
			merges, nn, st, k = self._lmRaw
			merges = self._kernelIDs(merges[:k])
			nn = nn[:k].astype(np.float64) if self._kernelData[-1] else nn[:k]
			lm = []
			for (c1, c2, c), (n1, n2), (s1, s2, s12, sw, cost) in zip(merges.tolist(), nn.tolist(), st[:k].tolist()):
				dic_key2value = {'nNode1':n1, 'nNode2':n2, 'nNode':n1+n2, 'sumWeight1':s1, 'sumWeight2':s2, 'sumWeight12':s12, 'sumWeight':sw, 'cost':cost}
				dic_key2value['separationGain'] = computeSeparationGain(dic_key2value)
				lm.append([c1, c2, c, dic_key2value])
			self._lm = lm
			self._lmRaw = None
		return self._lm

	@linkageMatrix.setter
	def linkageMatrix(self, value):
		self._lm = value
		self._lmRaw = None

	def finalCost(self):
		"""cost of the root of the last fitted tree (== linkageMatrix[-1][3]['cost'])"""
		if self._lm is None and self._lmRaw is not None:
			merges, nn, st, k = self._lmRaw
			return float(st[k-1, 4])
		return self._lm[-1][3]['cost']

	def mergePairs(self, start=0):
		"""[(c1, c2), ...] of the last fitted tree from merge `start` on"""
		if self._lm is None and self._lmRaw is not None:
			merges, nn, st, k = self._lmRaw
			return [(c1, c2) for c1, c2 in self._kernelIDs(merges[start:k, :2]).tolist()]
		return [(x[0], x[1]) for x in self._lm[start:]]

	def _prepareKernel(self):
		"""Pack the initial state into arrays for IDEA_kernel; None if not applicable."""
		if _kernel is None or self.linkageMethod not in _kernel.METHOD_CODES:
			return None
		method = _kernel.METHOD_CODES[self.linkageMethod]
		n0 = len(self.set_cluster)
		if n0 < 2 or n0 > _kernel.MAX_CLUSTERS or not isinstance(self.nTotalNode, (int, np.integer)):
			return None
		if any(not isinstance(c, (int, np.integer)) or isinstance(c, bool) for c in self.set_cluster):
			return None
		# kernel ids are 0..n0-1 (ascending original id); merged clusters keep the
		# original numbering max(id)+1, max(id)+2, ... via _kernelIDs
		lst_id = sorted(self.set_cluster)
		relabel = lst_id != list(range(n0))
		dic_id2k = dict((c, i) for i, c in enumerate(lst_id))
		# cluster sizes: all ints, or all integral floats (the 1.0 default of the node-level agents)
		raw = [self.dic_cluster2nNode.get(c, 1.0) for c in lst_id]
		if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in raw):
			nNodeFloat = False
		elif all(isinstance(v, float) and v.is_integer() and 0 < v < 2**52 for v in raw):
			nNodeFloat = True
		else:
			return None
		nNode0 = [int(v) for v in raw]
		if any(not isinstance(v, (float, int)) for v in self.dic_cluster2sumWeight.values()):
			return None
		sumW0 = [self.dic_cluster2sumWeight.get(c, 0.0) for c in lst_id]
		cost0 = [self.dic_cluster2cost[c] for c in lst_id]
		totalW = 1.0
		if method in (5, 10):
			cstat0 = [self.dic_cluster2modularitySumWeight.get(c, 0.0) for c in lst_id]
			if method == 5:
				totalW = self.totalEdgeSumWeight
		elif method == 8:
			cstat0 = [self.dic_cluster2edgeSumCount.get(c, 0.0) for c in lst_id]
		elif method == 9:
			cstat0 = [self.dic_cluster2edgeSumWeight.get(c, 0.0) for c in lst_id]
		else:
			cstat0 = [0.0]*n0
		auxDict = {1: getattr(self, 'dic_edge2weightedWeight', None), 2: getattr(self, 'dic_edge2maxWeight', None),
			3: getattr(self, 'dic_edge2minWeight', None), 8: getattr(self, 'dic_edge2countWeight', None)}.get(method)
		ea, eb, es, ex = [], [], [], []
		for (a, b), w in self.dic_edge2sumWeight.items():
			if not (a < b and a in dic_id2k and b in dic_id2k):
				return None
			ea.append(dic_id2k[a]); eb.append(dic_id2k[b]); es.append(w)
			if auxDict is not None:
				if (a, b) not in auxDict:
					return None
				ex.append(auxDict[(a, b)])
			else:
				ex.append(0.0)
		if auxDict is not None and len(auxDict) != len(self.dic_edge2sumWeight):
			return None
		ia, ib, iw = [], [], []
		for (a, b), w in self.sorted_edge:
			ia.append(dic_id2k[a]); ib.append(dic_id2k[b]); iw.append(w)
		i64, f64 = np.int64, np.float64
		if relabel:
			# kernel id -> original id (initial clusters), merged ids continue after max(original id)
			idMap = np.concatenate([np.array(lst_id, i64), np.arange(lst_id[-1]+1, lst_id[-1]+n0, dtype=i64)])
		else:
			idMap = None
		return (method, n0, int(self.nTotalNode), np.array(nNode0, i64), np.array(sumW0, f64), np.array(cost0, f64),
			np.array(cstat0, f64), float(totalW), np.array(ea, i64), np.array(eb, i64), np.array(es, f64), np.array(ex, f64),
			np.array(ia, i64), np.array(ib, i64), np.array(iw, f64), idMap, nNodeFloat)

	def _kernelIDs(self, arr):
		idMap = self._kernelData[15]
		return arr if idMap is None else idMap[arr]

	def fit(self, linkageMatrix=(), set_forbiddenPair=frozenset(), returnCandidate=False, prefixArray=None):
		"""Fit the agent; uses the compiled kernel when enabled and applicable."""
		if self.useKernel and not returnCandidate:
			if self._kernelData is None:
				self._kernelData = self._prepareKernel() or False
			if self._kernelData and (self._kernelData[15] is None or (len(linkageMatrix) == 0 and not set_forbiddenPair)):
				if prefixArray is None:
					prefixArray = np.array([[x[0], x[1], x[2]] for x in linkageMatrix], dtype=np.int64).reshape(-1, 3)
				if set_forbiddenPair:
					fp = np.array(list(set_forbiddenPair), dtype=np.int64).reshape(-1, 2)
					fa, fb = fp[:, 0].copy(), fp[:, 1].copy()
				else:
					fa = fb = _EMPTY_I64
				status, k, merges, nn, st = _kernel.agglomerate(*self._kernelData[:15], prefixArray, fa, fb)
				if status == 0:
					self._lm = None
					self._lmRaw = (merges, nn, st, k)
					return
				# order-sensitive corner case: let the exact Python implementation decide
		return self._fitPython(linkageMatrix, set_forbiddenPair, returnCandidate)

	def _fitPython(self, linkageMatrix=(), set_forbiddenPair=frozenset(), returnCandidate=False):
		"""Greedy agglomeration, optionally replaying a fixed merge prefix first.

		Semantics (merge order, tie breaking, floating point arithmetic) are
		identical to the original implementation; only the bookkeeping is faster:
		  * neighbours of the merged pair are found via an adjacency map instead
		    of scanning all live clusters, but are visited in the same order the
		    original scan would visit them (the order of ``set_cluster``);
		  * per-method weight updates are bound once per call (no string chain);
		  * the replayed prefix state is cached and reused when the next call
		    extends the same prefix (see ``chunkEnsembleClustering``).
		"""
		if not hasattr(self, 'dic_cluster2nbrs'):
			self._initState()
		nTotalNode = self.nTotalNode
		linkageMethod = self.linkageMethod
		costFunction = self.costFunction
		inf = np.inf

		# ---- restore state: from cache (longest matching replayed prefix) or from scratch
		nPrefix = len(linkageMatrix)
		cache = getattr(self, '_prefixCache', None)
		if cache is not None and cache['n'] <= nPrefix and _samePrefix(cache['lm'], linkageMatrix, cache['n']):
			st = cache
			start = cache['n']
			set_cluster = set(st['set_cluster'])
			lm = list(st['lm'])
		else:
			st = None
			start = 0
			set_cluster = set(self.set_cluster)
			lm = []
		src = st if st is not None else self.__dict__
		dic_cluster2nNode = dict(src['dic_cluster2nNode'])
		dic_cluster2cost = dict(src['dic_cluster2cost'])
		dic_cluster2sumWeight = dict(src['dic_cluster2sumWeight'])
		dic_edge2sumWeight = dict(src['dic_edge2sumWeight'])
		nbrs = dict((c, set(s)) for c, s in src['dic_cluster2nbrs'].items())
		ext = {}
		for name in ('dic_edge2maxWeight', 'dic_edge2minWeight', 'dic_edge2weightedWeight', 'dic_edge2countWeight', 'dic_cluster2edgeSumWeight', 'dic_cluster2edgeSumCount', 'dic_cluster2modularitySumWeight'):
			if name in src:
				ext[name] = dict(src[name])
		totalEdgeSumWeight = getattr(self, 'totalEdgeSumWeight', None)
		dic_edge2maxWeight = ext.get('dic_edge2maxWeight')
		dic_edge2minWeight = ext.get('dic_edge2minWeight')
		dic_edge2weightedWeight = ext.get('dic_edge2weightedWeight')
		dic_edge2countWeight = ext.get('dic_edge2countWeight')
		dic_cluster2edgeSumWeight = ext.get('dic_cluster2edgeSumWeight')
		dic_cluster2edgeSumCount = ext.get('dic_cluster2edgeSumCount')
		dic_cluster2modularitySumWeight = ext.get('dic_cluster2modularitySumWeight')

		if st is not None:
			dic_i2list = dict((i, list(l)) for i, l in st['dic_i2list'].items())
			lst_target = list(st['lst_target'])
			max_i = st['max_i']
		else:
			max_i = 0
			lst_target = []
			dic_i2list = {}
			if len(self.sorted_edge) > 0:
				dic_i2list[max_i] = list(self.sorted_edge)
				heapq.heappush(lst_target, (-dic_i2list[max_i][-1][1], max_i))
				max_i = max(dic_i2list.keys())+1

		nN = dic_cluster2nNode.get

		# ---- per-method pieces
		# clusterUpdate(c1, c2, new_cluster, e12, replay): cluster-level statistics of the merged cluster
		# edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay): linkage value of a new edge
		lm_ = linkageMethod
		if lm_ in ('deltaModularity', 'minModularityWeightNormalized', 'maxModularityWeightNormalized', 'avgModularityWeightNormalized'):
			msw = dic_cluster2modularitySumWeight
			def clusterUpdate(c1, c2, new_cluster, e12, replay):
				msw[new_cluster] = msw.get(c1, 0.0)+msw.get(c2, 0.0)
		elif lm_ in ('minEdgeCountNormalized', 'maxEdgeCountNormalized', 'unionEdgeCountNormalized'):
			ecs, ecw = dic_cluster2edgeSumCount, dic_edge2countWeight
			def clusterUpdate(c1, c2, new_cluster, e12, replay):
				ecs[new_cluster] = ecs.get(c1, 0.0) + ecs.get(c2, 0.0) - ecw.get(e12, 0.0)
		elif lm_ in ('minEdgeWeightNormalized', 'maxEdgeWeightNormalized', 'unionEdgeWeightNormalized'):
			cesw = dic_cluster2edgeSumWeight
			def clusterUpdate(c1, c2, new_cluster, e12, replay):
				cesw[new_cluster] = cesw.get(c1, 0.0) + cesw.get(c2, 0.0) - dic_edge2sumWeight.get(e12, 0.0)
		elif lm_ == 'deltaCost':
			cesw = dic_cluster2edgeSumWeight
			def clusterUpdate(c1, c2, new_cluster, e12, replay):
				# the original only updates this during prefix replay
				if replay:
					cesw[new_cluster] = cesw.get(c1, 0.0)+cesw.get(c2, 0.0) - dic_edge2sumWeight.get(e12, 0.0)
		else:
			clusterUpdate = None

		if lm_ == 'single':
			d = dic_edge2maxWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				weight = max(d.pop(e1, 0.0), d.pop(e2, 0.0))
				d[new_e] = weight
				return weight
		elif lm_ == 'complete':
			d = dic_edge2minWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				weight = min(d.pop(e1, inf), d.pop(e2, inf))
				d[new_e] = weight
				return weight
		elif lm_ == 'weighted':
			d = dic_edge2weightedWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				weight = 0.5*(d.pop(e1, 0.0)+d.pop(e2, 0.0))
				d[new_e] = weight
				return weight
		elif lm_ == 'average':
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				return sumWeight / float(nN(new_cluster, 1.0)*nN(cluster, 1.0))
		elif lm_ == 'deltaCost':
			cesw = dic_cluster2edgeSumWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				# the original uses different per-cluster sums in replay vs greedy phase
				dd = cesw if replay else dic_cluster2sumWeight
				return 1.0 / (dd.get(new_cluster, 0.0) * nN(cluster, 1.0) + dd.get(cluster, 0.0) * nN(new_cluster, 1.0) - sumWeight * float(nN(new_cluster, 1.0)*nN(cluster, 1.0)))
		elif lm_ == 'deltaRev':
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				weight = sumWeight / float(nN(new_cluster, 1.0)*nN(cluster, 1.0))
				weight *= nTotalNode-nN(new_cluster, 1.0)-nN(cluster, 1.0)
				return weight
		elif lm_ == 'deltaModularity':
			msw = dic_cluster2modularitySumWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				return sumWeight - msw.get(new_cluster, 0.0)*msw.get(cluster, 0.0)/totalEdgeSumWeight
		elif lm_ in ('edgeCountNormalized', 'minEdgeCountNormalized', 'maxEdgeCountNormalized', 'unionEdgeCountNormalized'):
			ecw = dic_edge2countWeight
			ecs = dic_cluster2edgeSumCount
			if lm_ == 'edgeCountNormalized':
				def norm(new_cluster, cluster, countWeight):
					return countWeight
			elif lm_ == 'minEdgeCountNormalized':
				def norm(new_cluster, cluster, countWeight):
					return min(ecs.get(new_cluster, 0.0), ecs.get(cluster, 0.0))
			elif lm_ == 'maxEdgeCountNormalized':
				def norm(new_cluster, cluster, countWeight):
					return max(ecs.get(new_cluster, 0.0), ecs.get(cluster, 0.0))
			else:
				def norm(new_cluster, cluster, countWeight):
					return ecs.get(new_cluster, 0.0)+ecs.get(cluster, 0.0)-countWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				countWeight = ecw.pop(e1, 0.0)+ecw.pop(e2, 0.0)
				ecw[new_e] = countWeight
				return sumWeight / norm(new_cluster, cluster, countWeight)
		elif lm_ == 'minEdgeWeightNormalized':
			cesw = dic_cluster2edgeSumWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				return sumWeight / min(cesw.get(new_cluster, 0.0), cesw.get(cluster, 0.0))
		elif lm_ == 'maxEdgeWeightNormalized':
			cesw = dic_cluster2edgeSumWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				return sumWeight / max(cesw.get(new_cluster, 0.0), cesw.get(cluster, 0.0))
		elif lm_ == 'unionEdgeWeightNormalized':
			cesw = dic_cluster2edgeSumWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				return sumWeight / (cesw.get(new_cluster, 0.0)+cesw.get(cluster, 0.0)-sumWeight)
		elif lm_ == 'minModularityWeightNormalized':
			msw = dic_cluster2modularitySumWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				return sumWeight / min(msw.get(new_cluster, 0.0), msw.get(cluster, 0.0))
		elif lm_ == 'maxModularityWeightNormalized':
			msw = dic_cluster2modularitySumWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				return sumWeight / max(msw.get(new_cluster, 0.0), msw.get(cluster, 0.0))
		elif lm_ == 'avgModularityWeightNormalized':
			msw = dic_cluster2modularitySumWeight
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				return sumWeight / (0.5*(msw.get(new_cluster, 0.0)+msw.get(cluster, 0.0)))
		elif lm_ == 'minPossibleEdgeCountNormalized':
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				minNNode = min(nN(new_cluster, 1.0), nN(cluster, 1.0))
				return sumWeight / (minNNode*(nTotalNode-minNNode))
		elif lm_ == 'minNodeNormalized':
			def edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay):
				return sumWeight / min(nN(new_cluster, 1.0), nN(cluster, 1.0))
		else:
			raise ValueError('unknown linkageMethod %s' % linkageMethod)

		esw = dic_edge2sumWeight
		esw_pop = esw.pop
		empty = frozenset()

		def merge(c1, c2, new_cluster, replay):
			nonlocal max_i
			nNode1, nNode2 = nN(c1, 1.0), nN(c2, 1.0)
			nNode = nNode1+nNode2
			sumWeight1, sumWeight2 = dic_cluster2sumWeight.get(c1, 0.0), dic_cluster2sumWeight.get(c2, 0.0)
			e12 = (c1, c2) if c1 < c2 else (c2, c1)
			sumWeight12 = esw.get(e12, 0.0)
			sumWeight = sumWeight1+sumWeight2+sumWeight12
			cost1, cost2 = dic_cluster2cost.get(c1, 0.0), dic_cluster2cost.get(c2, 0.0)
			cost = cost1+cost2+costFunction(sumWeight12, nNode)
			dic_cluster2nNode[new_cluster] = nNode
			dic_cluster2sumWeight[new_cluster] = sumWeight
			dic_cluster2cost[new_cluster] = cost
			dic_key2value = {'nNode1':nNode1, 'nNode2':nNode2, 'nNode':nNode, 'sumWeight1':sumWeight1, 'sumWeight2':sumWeight2, 'sumWeight12':sumWeight12, 'sumWeight':sumWeight, 'cost':cost}
			dic_key2value['separationGain'] = computeSeparationGain(dic_key2value)
			lm.append([c1, c2, new_cluster, dic_key2value])
			if clusterUpdate is not None:
				clusterUpdate(c1, c2, new_cluster, e12, replay)

			set_cluster.remove(c1)
			set_cluster.remove(c2)
			cand = set(nbrs.pop(c1, empty))
			cand |= nbrs.pop(c2, empty)
			cand.intersection_update(set_cluster)
			nbrs[new_cluster] = cand
			if cand:
				lst_clusterPair2weight2 = []
				append = lst_clusterPair2weight2.append
				for cluster in cand:
					if cluster < c1:
						e1 = (cluster, c1)
					else:
						e1 = (c1, cluster)
					if cluster < c2:
						e2 = (cluster, c2)
					else:
						e2 = (c2, cluster)
					new_e = (new_cluster, cluster) if new_cluster < cluster else (cluster, new_cluster)
					sumWeight = esw_pop(e1, 0.0) + esw_pop(e2, 0.0)
					esw[new_e] = sumWeight
					nb = nbrs[cluster]
					nb.discard(c1)
					nb.discard(c2)
					nb.add(new_cluster)
					append([new_e, edgeWeight(new_cluster, cluster, sumWeight, e1, e2, new_e, replay)])
				lst_clusterPair2weight2.sort(key=_second)
				if len(lst_clusterPair2weight2) > 1 and _hasTie(lst_clusterPair2weight2):
					# Equal weights: the original's stable sort ordered them by the
					# iteration order of set_cluster, so reproduce that exactly.
					rank = dict((c, j) for j, c in enumerate([c for c in set_cluster if c in cand]))
					other = lambda e: e[0] if e[1] == new_cluster else e[1]
					lst_clusterPair2weight2.sort(key=lambda x: rank[other(x[0])])
					lst_clusterPair2weight2.sort(key=_second)
				dic_i2list[max_i] = lst_clusterPair2weight2
				heapq.heappush(lst_target, (-lst_clusterPair2weight2[-1][1], max_i))
				max_i += 1
			set_cluster.add(new_cluster)

		# ---- replay the fixed prefix
		for j in range(start, nPrefix):
			x = linkageMatrix[j]
			merge(x[0], x[1], x[2], True)

		if nPrefix > 0 and self.cachePrefix:
			self._prefixCache = {
				'n': nPrefix, 'lm': list(lm), 'set_cluster': set(set_cluster),
				'dic_cluster2nNode': dict(dic_cluster2nNode), 'dic_cluster2cost': dict(dic_cluster2cost),
				'dic_cluster2sumWeight': dict(dic_cluster2sumWeight), 'dic_edge2sumWeight': dict(dic_edge2sumWeight),
				'dic_cluster2nbrs': dict((c, set(s)) for c, s in nbrs.items()),
				'dic_i2list': dict((i, list(l)) for i, l in dic_i2list.items()),
				'lst_target': list(lst_target), 'max_i': max_i,
			}
			for name, dd in ext.items():
				self._prefixCache[name] = dict(dd)

		self.linkageMatrix = lm

		if returnCandidate:
			lst_candidate=[]
			for candidates in dic_i2list.values():
				lst_candidate+=[[(c1,c2),weight] for (c1,c2),weight in candidates if c1 in set_cluster and c2 in set_cluster]
			lst_candidate.sort(key=lambda x:x[1])
			return lst_candidate

		if len(set_cluster) <= 1:
			return
		new_cluster = max(set_cluster) + 1
		heappop, heappush = heapq.heappop, heapq.heappush
		while True:
			if len(lst_target) != 0:
				negweight, i = heappop(lst_target)
				lst_i = dic_i2list[i]
				(c1, c2), weight = lst_i.pop()
				if c1 not in set_cluster or c2 not in set_cluster:
					while True:
						if len(lst_i) == 0:
							break
						(c1, c2), weight = lst_i.pop()
						if c1 in set_cluster and c2 in set_cluster:
							lst_i.append([(c1,c2), weight])
							break
					if len(lst_i) == 0:
						del dic_i2list[i]
					else:
						heappush(lst_target,(-lst_i[-1][1],i))
					continue
				if len(lst_i) == 0:
					del dic_i2list[i]
				else:
					heappush(lst_target,(-lst_i[-1][1],i))
				if set_forbiddenPair and K(c1, c2) in set_forbiddenPair:
					continue
			else:
				(c1, c2), weight = sorted(set_cluster, key=lambda x:dic_cluster2nNode.get(x,1.0))[0:2], 0.0

			merge(c1, c2, new_cluster, False)
			new_cluster+=1
			if len(set_cluster) <= 1:
				break
		self.linkageMatrix.sort(key=lambda x:x[2])

class ChunkTree:
	def __init__(self, ID, wKNNgraph=None, nTotalNode=None):
		self.ID = ID
		self.wKNNgraph = wKNNgraph
		if wKNNgraph != None:
			self.set_node = set(self.wKNNgraph.dic_node2name.keys())
		else:
			self.set_node = None
		self.dic_chunk2nodes= None
		self.dic_node2chunk= None
		self.bestLinkageMatrix=None
		self.bestMethod=None
		self.score=None
		if nTotalNode == None:
			self.nTotalNode = len(wKNNgraph.dic_node2node_weight)
		else:
			self.nTotalNode = nTotalNode
		self.costFunction=None

	def chunking(self, nChunk, nTrial=5, chunkingMethod='metis'):
		if nChunk >= len(self.set_node):
			self.dic_chunk2nodes={}
			self.dic_node2chunk={}
			for chunk, node in enumerate(self.set_node):
				self.dic_chunk2nodes[chunk]={node}
				self.dic_node2chunk[node]=chunk
			return

		if nChunk <= 1:
			self.dic_node2chunk={}
			self.dic_chunk2nodes={0:self.set_node}
			for node in self.set_node:
				self.dic_node2chunk[node]=0
			return

		if chunkingMethod == 'average':
			import BaselineHC_fast
			linkageMatrix = BaselineHC_fast.fit(self.wKNNgraph.dic_node2node_weight, linkageMethod=chunkingMethod)
			dic_node2cluster = cutTree(linkageMatrix, t=nChunk, maxLeafID=max(self.set_node))
			self.dic_chunk2nodes={}
			self.dic_node2chunk={}
			dic_cluster2chunk={}
			for node, cluster in dic_node2cluster.items():
				if cluster not in dic_cluster2chunk:
					dic_cluster2chunk[cluster]=len(dic_cluster2chunk)
				chunk = dic_cluster2chunk[cluster]
				if chunk not in self.dic_chunk2nodes:
					self.dic_chunk2nodes[chunk]=set()
				self.dic_chunk2nodes[chunk].add(node)
				self.dic_node2chunk[node]=chunk
			return

		if chunkingMethod == 'minModularityWeightNormalized':
			import BaselineHC_fast
			linkageMatrix = BaselineHC_fast.fit(self.wKNNgraph.dic_node2node_weight, linkageMethod=chunkingMethod)
			dic_node2cluster = cutTree(linkageMatrix, t=nChunk)
			self.dic_chunk2nodes={}
			self.dic_node2chunk={}
			dic_cluster2chunk={}
			for node, cluster in dic_node2cluster.items():
				if cluster not in dic_cluster2chunk:
					dic_cluster2chunk[cluster]=len(dic_cluster2chunk)
				chunk = dic_cluster2chunk[cluster]
				if chunk not in self.dic_chunk2nodes:
					self.dic_chunk2nodes[chunk]=set()
				self.dic_chunk2nodes[chunk].add(node)
				self.dic_node2chunk[node]=chunk
			return

		if chunkingMethod == 'kmeans':
			import sklearn.cluster
			lst_node = sorted(self.set_node)
			X=self.wKNNgraph.data[lst_node,:]
			kmeans = sklearn.cluster.KMeans(n_clusters=nChunk, random_state=0).fit(X)
			kmeans.labels_
			self.dic_chunk2nodes={}
			self.dic_chunk2nNode={}
			self.dic_node2chunk={}
			dic_cluster2chunk={}
			for node, cluster in zip(lst_node,kmeans.labels_):
				if cluster not in dic_cluster2chunk:
					dic_cluster2chunk[cluster]=len(dic_cluster2chunk)
				chunk = dic_cluster2chunk[cluster]
				if chunk not in self.dic_chunk2nodes:
					self.dic_chunk2nodes[chunk]=set()
					self.dic_chunk2nNode[chunk]=0
				self.dic_chunk2nodes[chunk].add(node)
				self.dic_chunk2nNode[chunk]+=1
				self.dic_node2chunk[node]=chunk
			self.nChunk=len(self.dic_chunk2nodes)
			self.set_chunk=set(self.dic_chunk2nodes.keys())
			return

		if chunkingMethod == 'metis_kway':
			lst_clusters=partitionByMetis(self.set_node, nChunk, self.wKNNgraph, recursive=False)
			dic_node2cluster={}
			for cluster, set_node in enumerate(lst_clusters):
				for node in set_node:
					dic_node2cluster[node]=cluster
			self.dic_chunk2nodes={}
			self.dic_chunk2nNode={}
			self.dic_node2chunk={}
			dic_cluster2chunk={}
			for node, cluster in dic_node2cluster.items():
				if cluster not in dic_cluster2chunk:
					dic_cluster2chunk[cluster]=len(dic_cluster2chunk)
				chunk = dic_cluster2chunk[cluster]
				if chunk not in self.dic_chunk2nodes:
					self.dic_chunk2nodes[chunk]=set()
					self.dic_chunk2nNode[chunk]=0
				self.dic_chunk2nodes[chunk].add(node)
				self.dic_chunk2nNode[chunk]+=1
				self.dic_node2chunk[node]=chunk
			self.nChunk=len(self.dic_chunk2nodes)
			self.set_chunk=set(self.dic_chunk2nodes.keys())
			return

		lst_nodes = connectedComponent(self.wKNNgraph)
		if len(lst_nodes) == 1:
			pass
		else:
			self.dic_chunk2nodes={}
			self.dic_node2chunk={}
			for chunk, set_node in enumerate(lst_nodes):
				self.dic_chunk2nodes[chunk]=set_node
				for node in set_node:
					self.dic_node2chunk[node]=chunk
			return

		if self.dic_chunk2nodes == None:
			dic_cluster2nodes={}
			dic_node2cluster={}
			lst_clusters=partitionByMetis(self.set_node, nChunk, self.wKNNgraph)
			for cluster, set_node in enumerate(lst_clusters):
				dic_cluster2nodes[cluster]=set_node
				for node in set_node:
					dic_node2cluster[node]=cluster
			new_cluster = max(dic_cluster2nodes.keys())+1
		else:
			dic_cluster2nodes = self.dic_chunk2nodes
			dic_node2cluster = self.dic_node2chunk
			lst_cluster = list(dic_cluster2nodes.keys())
			new_cluster = max(lst_cluster)+1
			while True:
				if len(lst_cluster) >= nChunk:
					break
				lst_cluster.sort(key=lambda x: len(dic_cluster2nodes[x]), reverse=True)
				cluster = lst_cluster.pop(0)
				lst_clusters=partitionByMetis(dic_cluster2nodes[cluster], 2, self.wKNNgraph)
				for set_node in lst_clusters:
					dic_cluster2nodes[new_cluster]=set_node
					for node in set_node:
						dic_node2cluster[node]=new_cluster
					lst_cluster.append(new_cluster)
					new_cluster+=1
				del dic_cluster2nodes[cluster]
				
		for trial in range(nTrial):
			set_unconfirmedCluster=set(dic_cluster2nodes.keys())
			while True:
				while True:
					if len(dic_cluster2nodes) >= nChunk:
						break
					cluster = sorted(dic_cluster2nodes.keys(), key=lambda x:len(dic_cluster2nodes[x]), reverse=True)[0]
					if len(dic_cluster2nodes[cluster]) == 1:
						break
					lst_clusters=partitionByMetis(dic_cluster2nodes[cluster], 2, self.wKNNgraph)
					for set_node in lst_clusters:
						dic_cluster2nodes[new_cluster]=set_node
						for node in set_node:
							dic_node2cluster[node]=new_cluster
						#if cluster in set_unconfirmedCluster:
						#	set_unconfirmedCluster.add(new_cluster)
						new_cluster+=1
					del dic_cluster2nodes[cluster]
					set_unconfirmedCluster.discard(cluster)
				if len(set_unconfirmedCluster) == 0:
					break
				cluster = sorted(set_unconfirmedCluster, key=lambda x:len(dic_cluster2nodes[x]), reverse=True)[0]
				set_unconfirmedCluster.remove(cluster)
				if len(dic_cluster2nodes[cluster]) == 1:
					continue
				lst_clusters = partitionByMetis(dic_cluster2nodes[cluster],2,self.wKNNgraph,isExact=True)
				tmpC1, tmpC2 = new_cluster, new_cluster+1
				if len(lst_clusters[0]) >= len(lst_clusters[1]):
					dic_cluster2nodes[tmpC1]=lst_clusters[0]
					dic_cluster2nodes[tmpC2]=lst_clusters[1]
				else:
					dic_cluster2nodes[tmpC1]=lst_clusters[1]
					dic_cluster2nodes[tmpC2]=lst_clusters[0]
				for node in dic_cluster2nodes[tmpC1]:
					dic_node2cluster[node]=tmpC1
				for node in dic_cluster2nodes[tmpC2]:
					dic_node2cluster[node]=tmpC2
					
				set_shatteredNode=set()
				lst_silh1 = []
				for node in dic_cluster2nodes[tmpC1]:
					weight_aa, weight_ab = 0.0, 0.0
					for n2, weight in self.wKNNgraph.dic_node2node_weight[node].items():
						c = dic_node2cluster[n2]
						if c == tmpC1:
							continue
						elif c == tmpC2:
							weight_aa += weight
						else:
							weight_ab += weight
					if weight_aa == 0.0 and weight_ab == 0.0:
						silh = 0.0
					else:
						max_weight = max(weight_aa,weight_ab)
						silh = (weight_aa-weight_ab)/max_weight
					if silh < -0.5:
						set_shatteredNode.add(node)
					lst_silh1.append(silh)
				lst_silh2 = []
				for node in dic_cluster2nodes[tmpC2]:
					weight_aa, weight_ab = 0.0, 0.0
					dic_c2weights={}
					for n2, weight in self.wKNNgraph.dic_node2node_weight[node].items():
						c = dic_node2cluster[n2]
						if c == tmpC2:
							continue
						elif c == tmpC1:
							weight_aa += weight
						else:
							weight_ab += weight
					if weight_aa == 0.0 and weight_ab == 0.0:
						silh = 0.0
					else:
						max_weight = max(weight_aa,weight_ab)
						silh = (weight_aa-weight_ab)/max_weight
					if silh < -0.5:
						set_shatteredNode.add(node)
					lst_silh2.append(silh)

				nMember = len(dic_cluster2nodes[cluster])

				tmpSilh = sum(lst_silh1+lst_silh2) / float(nMember)
				if tmpSilh < -0.0:
					del dic_cluster2nodes[cluster]
					#set_unconfirmedCluster.add(tmpC1)
					#set_unconfirmedCluster.add(tmpC2)
					new_cluster+=2
				else:
					del dic_cluster2nodes[tmpC1]
					del dic_cluster2nodes[tmpC2]
					for node in dic_cluster2nodes[cluster]:
						dic_node2cluster[node]=cluster
	
				for node in set_shatteredNode:
					c = dic_node2cluster[node]
					dic_cluster2nodes[c].remove(node)
					dic_node2cluster[node]=-1

				_reassignShatteredNodes(set_shatteredNode, dic_node2cluster, dic_cluster2nodes, self.wKNNgraph.dic_node2node_weight)

				if tmpSilh < -0.0:
					if len(dic_cluster2nodes[tmpC1]) == 0:
						del dic_cluster2nodes[tmpC1]
						#set_unconfirmedCluster.remove(tmpC1)
					if len(dic_cluster2nodes[tmpC2]) == 0:
						del dic_cluster2nodes[tmpC2]
						#set_unconfirmedCluster.remove(tmpC2)
				else:
					if len(dic_cluster2nodes[cluster]) == 0:
						del dic_cluster2nodes[cluster]

			set_shatteredNode=set()
			lst_cluster=sorted(dic_cluster2nodes.keys(), key=lambda x:len(dic_cluster2nodes[x]), reverse=True)
			for cluster in lst_cluster[nChunk:]:
				for node in dic_cluster2nodes[cluster]:
					set_shatteredNode.add(node)
					dic_node2cluster[node]=-1
				del dic_cluster2nodes[cluster]

			_reassignShatteredNodes(set_shatteredNode, dic_node2cluster, dic_cluster2nodes, self.wKNNgraph.dic_node2node_weight)

		self.dic_chunk2nodes={}
		self.dic_node2chunk={}
		for new_chunk, (chunk, set_node) in enumerate(dic_cluster2nodes.items()):
			self.dic_chunk2nodes[new_chunk]=set_node
			for node in set_node:
				self.dic_node2chunk[node]=new_chunk
		return

	def chunkEnsembleClustering(self, linkageMethod='auto'):
		if linkageMethod == 'auto':
			#lst_linkageMethod=['single','complete','weighted','average','normalizedAverage','edgeAverage']
			#lst_linkageMethod=['single','complete','average','edgeAverage','weighted','normalizedAverage','normalizedEdgeAverage','deltaCost','deltaRev','deltaModularity']
			#lst_linkageMethod=['single','complete','average','weighted','deltaCost','deltaRev','deltaModularity','edgeCountNormalized','minEdgeCountNormalized','maxEdgeCountNormalized','unionEdgeCountNormalized','minEdgeWeightNormalized','maxEdgeWeightNormalized','unionEdgeWeightNormalized','minModularityWeightNormalized','maxModularityWeightNormalized','avgModularityWeightNormalized']
			#lst_linkageMethod=['single','complete','average','weighted','deltaCost','deltaRev','deltaModularity','edgeCountNormalized','minEdgeCountNormalized','unionEdgeCountNormalized','minEdgeWeightNormalized','unionEdgeWeightNormalized','minModularityWeightNormalized','avgModularityWeightNormalized']
			lst_linkageMethod=['average','weighted','single','complete','deltaRev','deltaModularity','minPossibleEdgeCountNormalized','minNodeNormalized','minEdgeCountNormalized','minEdgeWeightNormalized','minModularityWeightNormalized']
		else:
			lst_linkageMethod=[linkageMethod]

		wKNNgraph = self.wKNNgraph

		# initialize chunk network
		dic_chunk2countWeight=dict([(chunk,0.0) for chunk in self.dic_chunk2nodes.keys()])
		dic_chunk2sumWeight=dict([(chunk,0.0) for chunk in self.dic_chunk2nodes.keys()])
		dic_chunkEdge2maxWeight = {}
		dic_chunkEdge2minWeight = {}
		dic_chunkEdge2countWeight = {}
		dic_chunkEdge2sumWeight = {}
		for n1 in self.set_node:
			c1 = self.dic_node2chunk[n1]
			for n2, weight in wKNNgraph.dic_node2node_weight[n1].items():
				if n1 > n2:
					continue
				c2 = self.dic_node2chunk[n2]
				if c1 == c2:	
					dic_chunk2countWeight[c1]+=1.0
					dic_chunk2sumWeight[c1]+=weight
					continue
				edge = K(c1, c2)
				if edge not in dic_chunkEdge2maxWeight:
					dic_chunkEdge2maxWeight[edge]=.0
					dic_chunkEdge2minWeight[edge]=np.inf
					dic_chunkEdge2countWeight[edge]=.0
					dic_chunkEdge2sumWeight[edge]=.0
				if weight > dic_chunkEdge2maxWeight[edge]:
					dic_chunkEdge2maxWeight[edge]=weight
				if weight < dic_chunkEdge2minWeight[edge]:
					dic_chunkEdge2minWeight[edge]=weight
				dic_chunkEdge2countWeight[edge]+=1.0
				dic_chunkEdge2sumWeight[edge]+=weight

		lst_agent = []
		dic_chunk2nNode=dict([(chunk, len(set_node)) for chunk, set_node in self.dic_chunk2nodes.items()])
		for linkageMethod in lst_linkageMethod:
			agent = hierarchicalClusteringAgent(self.dic_chunk2nodes.keys(), linkageMethod, dic_edge2maxWeight=dic_chunkEdge2maxWeight, dic_edge2minWeight=dic_chunkEdge2minWeight, dic_edge2countWeight=dic_chunkEdge2countWeight, dic_edge2sumWeight=dic_chunkEdge2sumWeight, dic_node2n=dic_chunk2nNode, dic_node2countWeight=dic_chunk2countWeight, dic_node2sumWeight=dic_chunk2sumWeight, nTotalNode=self.nTotalNode)
			agent.cachePrefix = True  # rounds below re-fit with an ever-growing merge prefix
			agent.useKernel = True    # compiled kernel if available (exact, falls back to Python)
			lst_agent.append(agent)

		'''
		# IDEA best select
		best_i=None
		bestMethod=''
		bestCost=np.inf
		for i, agent in enumerate(lst_agent):
			agent.fit()
			cost = agent.linkageMatrix[-1][3]['cost']
			if cost < bestCost:
				best_i=i
				bestCost=cost
				bestMethod=agent.linkageMethod
			elif cost == bestCost:
				bestMethod+=';'+agent.linkageMethod

		self.bestLinkageMatrix=list(lst_agent[best_i].linkageMatrix)
		self.bestMethod=[bestMethod for i in range(len(self.bestLinkageMatrix))]

		dic_method2score={}
		for method in bestMethod.split(';'):	
			if method not in dic_method2score:
				dic_method2score[method]=0.0
			dic_method2score[method]+=1.0/len(bestMethod.split(';'))
		print 'asdf',sorted(dic_method2score.items(),key=lambda x:x[1],reverse=True)
		return
		'''

		# IDEA integrate best
		set_cluster = set(self.dic_chunk2nodes.keys())
		new_cluster = max(set_cluster)+1
		currentRound = 0
		bestCost=np.inf
		self.bestLinkageMatrix=None
		self.bestMethod=[]
		dic_method2score={}
		dic_i2score={}

		while True:
			if len(set_cluster) <= 1:
				break
			if self.bestLinkageMatrix is None:
				linkageMatrix=[]
			else:
				linkageMatrix=list(self.bestLinkageMatrix[:currentRound])
			prefixArray = np.array([[x[0], x[1], x[2]] for x in linkageMatrix], dtype=np.int64).reshape(-1, 3)
			for i, agent in enumerate(lst_agent):
				agent.fit(linkageMatrix, prefixArray=prefixArray)
				agentCost = agent.finalCost()
				if agentCost < bestCost:
					lst_best_i=[i]
					bestCost=agentCost
					self.bestLinkageMatrix=list(agent.linkageMatrix)
					bestMethod=agent.linkageMethod
				elif agentCost == bestCost:
					lst_best_i.append(i)
					bestMethod+=';'+agent.linkageMethod
			for i in lst_best_i:
				if i not in dic_i2score:
					dic_i2score[i]=0.0
				dic_i2score[i]+=1.0/len(lst_best_i)
			best_i=sorted(lst_best_i,key=lambda x:dic_i2score[x],reverse=True)[0]
				
			for method in bestMethod.split(';'):	
				if method not in dic_method2score:
					dic_method2score[method]=0.0
				dic_method2score[method]+=1.0/len(bestMethod.split(';'))

			lst_agent[best_i].linkageMatrix = list(self.bestLinkageMatrix)

			set_forbiddenNode=set([lst_agent[best_i].linkageMatrix[currentRound][0],lst_agent[best_i].linkageMatrix[currentRound][1]])
			set_forbiddenPair=set()
			set_forbiddenPair.add(K(lst_agent[best_i].linkageMatrix[currentRound][0],lst_agent[best_i].linkageMatrix[currentRound][1]))
			nIter=10
			for n in range(nIter):
				lst_agent[best_i].fit(linkageMatrix,set_forbiddenPair=set_forbiddenPair,prefixArray=prefixArray)
				agentCost = lst_agent[best_i].finalCost()
				if agentCost < bestCost:
					bestCost=agentCost
					self.bestLinkageMatrix=list(lst_agent[best_i].linkageMatrix)
				set_forbiddenPair2=set()
				for c1, c2 in lst_agent[best_i].mergePairs(currentRound):
					pair = K(c1,c2)
					if pair[0] in set_forbiddenNode or pair[1] in set_forbiddenNode:
						if pair not in set_forbiddenPair:
							set_forbiddenPair2.add(pair)
				if len(set_forbiddenPair2) == 0:
					break
				set_forbiddenPair |= set_forbiddenPair2
			lst_agent[best_i].linkageMatrix = list(self.bestLinkageMatrix)
			self.bestMethod.append(bestMethod)
			set_cluster.remove(self.bestLinkageMatrix[currentRound][0])
			set_cluster.remove(self.bestLinkageMatrix[currentRound][1])
			set_cluster.add(self.bestLinkageMatrix[currentRound][2])
			new_cluster+=1
			currentRound+=1
		return

	def membershipRefinement(self):
		return

	def fit(self, nChunk='auto', chunkingMethod='metis'): 
		assert(nChunk == 'auto' or type(nChunk)==int),("nChunk must be 'auto' or int")
		assert(self.wKNNgraph != None),("wKNNgraph must not be set before fitting")
		nNode = len(self.set_node)
		if nChunk == 'auto':
			nChunk=80+int(np.log2(nNode))
		else:
			nChunk=int(nChunk)
		self.chunking(nChunk, chunkingMethod=chunkingMethod)
		self.chunkEnsembleClustering()

	def halfSplit(self):
		if len(self.set_node) <= 1:
			return None, None

		if self.bestLinkageMatrix == None or len(self.bestLinkageMatrix) == 0:
			self.fit()

		b1=ChunkTree(ID=self.ID+'0', nTotalNode=self.nTotalNode)
		b2=ChunkTree(ID=self.ID+'1', nTotalNode=self.nTotalNode)

		root1,root2,cluster,dic_key2value = self.bestLinkageMatrix[-1]
		dic_cluster2root={root1:0, root2:1}
		b1.dic_chunk2nodes, b2.dic_chunk2nodes= {}, {}
		b1.bestMethod, b2.bestMethod = self.bestMethod, self.bestMethod
		b1.bestLinkageMatrix, b2.bestLinkageMatrix=[], []
		if root1 in self.dic_chunk2nodes:
			b1.dic_chunk2nodes[root1]=self.dic_chunk2nodes[root1]
		if root2 in self.dic_chunk2nodes:
			b2.dic_chunk2nodes[root2]=self.dic_chunk2nodes[root2]

		for x in reversed(self.bestLinkageMatrix[:-1]):
			c1, c2, cluster, dic_key2value = x
			dic_cluster2root[c1]=dic_cluster2root[cluster]
			dic_cluster2root[c2]=dic_cluster2root[cluster]
			if dic_cluster2root[c1] == 0:
				if c1 in self.dic_chunk2nodes:
					b1.dic_chunk2nodes[c1]=self.dic_chunk2nodes[c1]
				if c2 in self.dic_chunk2nodes:
					b1.dic_chunk2nodes[c2]=self.dic_chunk2nodes[c2]
				new_x = list(x)
				b1.bestLinkageMatrix.insert(0, new_x)
			else:
				if c1 in self.dic_chunk2nodes:
					b2.dic_chunk2nodes[c1]=self.dic_chunk2nodes[c1]
				if c2 in self.dic_chunk2nodes:
					b2.dic_chunk2nodes[c2]=self.dic_chunk2nodes[c2]
				new_x = list(x)
				b2.bestLinkageMatrix.insert(0, new_x)

		b1.dic_node2chunk={}
		for chunk, set_node in b1.dic_chunk2nodes.items():
			for node in set_node:
				b1.dic_node2chunk[node]=chunk
		b1.set_node=set(b1.dic_node2chunk.keys())

		b2.dic_node2chunk={}
		for chunk, set_node in b2.dic_chunk2nodes.items():
			for node in set_node:
				b2.dic_node2chunk[node]=chunk
		b2.set_node=set(b2.dic_node2chunk.keys())

		b1.wKNNgraph = self.wKNNgraph.subGraph(b1.set_node)
		b2.wKNNgraph = self.wKNNgraph.subGraph(b2.set_node)

		if len(b1.set_node) < len(b2.set_node):
			b1, b2 = b2, b1
			b1.ID = self.ID+'0'
			b2.ID = self.ID+'1'

		#return b1, b2
		self.subtree1, self.subtree2 = b1, b2

	def fullLinkageMatrix(self, linkageMethod='average', internalNodePrefix=''):
		subtreePrefix=internalNodePrefix
		linkageMatrix=[]
		for chunk, set_node in self.dic_chunk2nodes.items():
			if len(set_node) == 1:
				continue
			dic_edge2weight={}
			dic_edge2count={}
			for node in set_node:
				for n2, weight in self.wKNNgraph.dic_node2node_weight[node].items():
					if n2 not in set_node:
						continue
					if node < n2:
						dic_edge2weight[K(node,n2)]=weight
						dic_edge2count[K(node,n2)]=1.0
			agent = hierarchicalClusteringAgent(set_node, linkageMethod, dic_edge2maxWeight=dic_edge2weight, dic_edge2minWeight=dic_edge2weight, dic_edge2countWeight=dic_edge2count, dic_edge2sumWeight=dic_edge2weight, nTotalNode=self.nTotalNode)
			agent.useKernel = True
			agent.fit()
			tmplinkageMatrix=agent.linkageMatrix
			for x in tmplinkageMatrix:
				if x[0] not in set_node:
					x[0]='_'.join(map(str,[subtreePrefix,chunk,x[0]]))
				if x[1] not in set_node:
					x[1]='_'.join(map(str,[subtreePrefix,chunk,x[1]]))
				if x[2] not in set_node:
					x[2]='_'.join(map(str,[subtreePrefix,chunk,x[2]]))
			tmplinkageMatrix[-1][2] = '_'.join(map(str,[subtreePrefix,chunk]))
			linkageMatrix+=tmplinkageMatrix
		for xx in self.bestLinkageMatrix:
			x = list(xx)
			if x[0] in self.dic_chunk2nodes and len(self.dic_chunk2nodes[x[0]]) == 1:
				x[0]=list(self.dic_chunk2nodes[x[0]])[0]
			else:
				x[0]='_'.join(map(str,[subtreePrefix,x[0]]))
			if x[1] in self.dic_chunk2nodes and len(self.dic_chunk2nodes[x[1]]) == 1:
				x[1]=list(self.dic_chunk2nodes[x[1]])[0]
			else:
				x[1]='_'.join(map(str,[subtreePrefix,x[1]]))
			x[2]='_'.join(map(str,[subtreePrefix,x[2]]))
			linkageMatrix.append(x)
		return linkageMatrix

class IDEA:
	def __init__(self):
		self.linkageMatrix=None
		self.membership=None
		self.lst_name=None
		pass

	def fit(self, X, k, Xtype='graph', nNeighbor=10, nChunk='auto', nIteration='auto', outlierRatio=0.09, GraphConversionMethod='WKNN', chunkingMethod='metis', costFunction=lambda weightSum, nNode: weightSum*nNode*nNode):
		if nIteration == 'auto':
			nIteration=2+int(np.sqrt(k))
		assert(Xtype in ['datapoints', 'similarity', 'dissimilarity']),("Xtype must be one of the 'graph', 'similarity', 'dissimilarity'")
		wKNNgraph = WKNNgraph()
		if Xtype == 'datapoints':
			wKNNgraph.constructGraphByData(filename=X, metric='euclidean', nNeighbor=nNeighbor)
		elif Xtype == 'similarity':
			wKNNgraph.constructGraph(filename=X, weightType='similarity', nNeighbor=nNeighbor)
		elif Xtype == 'dissimilarity':
			wKNNgraph.constructGraph(filename=X, weightType='dissimilarity', nNeighbor=nNeighbor)
		'''
		if Xtype == 'graph':
			wKNNgraph.loadGraph(filename=X)
		'''
		self.lst_name=[name for node, name in sorted(wKNNgraph.dic_node2name.items())]
		assert(len(self.lst_name)>1),(
			"# of node must be > 1")
		nNode = len(wKNNgraph.dic_node2name)
		assert(k<=nNode),(
			"k must be less than or equal to nNode")
		subtree = ChunkTree(ID='0', wKNNgraph=wKNNgraph, nTotalNode=nNode)
		lst_subtree = []
		linkageMatrix=[]
		nConnectedComponent = len(connectedComponent(wKNNgraph))
		if nConnectedComponent > 1:
			print('warning: # of connected components is %d (>1)'%(nConnectedComponent)) 
		iteration=0
		lst_subtree.append(subtree)
		while True:
			if iteration >= nIteration or len(lst_subtree)==nNode:
				break
			iteration+=1
			subtree=lst_subtree.pop(0)
			subtree.fit(nChunk=nChunk, chunkingMethod=chunkingMethod)
			subtree.halfSplit()
			subtree1, subtree2= subtree.subtree1, subtree.subtree2
			if subtree1.bestLinkageMatrix == None or len(subtree1.bestLinkageMatrix) == 0:
				subtree1.score = -np.inf
			else:
				subtree1.score = subtree1.bestLinkageMatrix[-1][3]['separationGain']
			if subtree2.bestLinkageMatrix == None or len(subtree2.bestLinkageMatrix) == 0:
				subtree2.score = -np.inf
			else:
				subtree2.score = subtree2.bestLinkageMatrix[-1][3]['separationGain']
			linkageMatrix.append([subtree1.ID,subtree2.ID,subtree.ID,subtree.bestLinkageMatrix[-1][3]])
			del subtree
			if len(subtree1.set_node) == 1: #leaf node
				linkageMatrix[-1][0] = list(subtree1.set_node)[0]
			lst_subtree.append(subtree1)
			if len(subtree2.set_node) == 1: #leaf node
				linkageMatrix[-1][1] = list(subtree2.set_node)[0]
			lst_subtree.append(subtree2)
			lst_subtree.sort(key=lambda subtree:len(subtree.set_node), reverse=True)

		lst_subtree.sort(key=lambda subtree:subtree.ID)
		for clusterID, subtree in enumerate(lst_subtree):
			tmplinkageMatrix=subtree.fullLinkageMatrix(internalNodePrefix=clusterID)
			if len(tmplinkageMatrix) >= 1:
				tmplinkageMatrix[-1][2]=subtree.ID
				linkageMatrix+=tmplinkageMatrix
		self.linkageMatrix = arrangeLinkageMatrix(linkageMatrix)
		dic_node2cluster = cutTree(self.linkageMatrix,k,outlierRatio=outlierRatio)###
		assert(max(dic_node2cluster.keys())==len(dic_node2cluster)-1),("error")
		self.membership = [cluster for node, cluster in sorted(dic_node2cluster.items())]
		self.wKNNgraph = wKNNgraph
		return
		
if __name__ == "__main__":
	parser=argparse.ArgumentParser(
		usage='''\
	%(prog)s [options] datafile k -o cluster_result
	example: %(prog)s datapoints.txt 7 -o cluster_result.txt
	''')
	
	parser.add_argument('datafile', help='datapoints, similarity, or dissimilarity file')
	parser.add_argument('k', type=int, help='the number of cluster')
	parser.add_argument('-dataType', required=False, default='datapoints', choices=['datapoints', 'similarity', 'dissimilarity'], help='dataType')
	parser.add_argument('-o', required=False, default='stdout', help='outfile of cluster result')
	parser.add_argument('-o2', required=False, default=None, help='outfile of linkage matrix (default = None)')
	parser.add_argument('-o3', required=False, default=None, help='outfile of WKNN graph (default = None)')
	parser.add_argument('-nNeighbor', required=False, type=int, default=10, help='Alpha-nearest neighbor graph (default = 10)')
	parser.add_argument('-nChunk', required=False, default='auto', help='the number of chunks (default = auto = 80+int(log_2 N))')
	parser.add_argument('-nIteration', required=False, default='auto', help='the iteration of recursive optimization (default = auto = 2+int(sqrt(k))')
	parser.add_argument('-outlierRatio', required=False, type=float, default=0.09, help='the outlier ratio (default = 0.09)')
	parser.add_argument('-GraphConversionMethod', required=False, default='WKNN', choices=['None','WKNN'], help='How to convert distance into graph (default = WKNN)')
	parser.add_argument('-chunkingMethod', required=False, default='metis', choices=['metis', 'average'], help='chunking method')
	args=parser.parse_args()
	

	global_starttime=time.time()
	idea = IDEA()
	idea.fit(X=args.datafile, k=args.k, Xtype=args.dataType, nNeighbor=args.nNeighbor, nChunk=args.nChunk, nIteration=args.nIteration, outlierRatio=args.outlierRatio, GraphConversionMethod=args.GraphConversionMethod, chunkingMethod=args.chunkingMethod)

	# output clustering results
	if args.o == 'stdout':
		OF=sys.stdout
	else:
		OF=open(args.o,'w')

	for nodename, membership in zip(idea.lst_name, idea.membership):
		OF.write('\t'.join(map(str,[nodename,membership]))+'\n')
	OF.close()

	# output clustering tree
	if args.o2 != None:
		OF=open(args.o2,'w')
		for x in idea.linkageMatrix:
			OF.write('\t'.join(map(str,x))+'\n')
		OF.close()

	# output clustering tree
	if args.o3 != None:
		idea.wKNNgraph.write(args.o3)
