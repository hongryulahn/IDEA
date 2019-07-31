#!/usr/bin/env python
import sys
import argparse
import numpy as np
import itertools
import time
import random
import pymetis
import heapq
from WKNNgraph import *

def cutTree2(linkageMatrix, t, outlierRatio=None):
	dic_parent2childs={}
	dic_child2parent={}
	dic_node2weight={}
	dic_node2nNode={}
	dic_node2averageWeight={}
	set_leaf=set()
	for c, x in enumerate(linkageMatrix,len(linkageMatrix)+1):
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
	round = 0
	while len(set_root) < t:
		target = sorted(set_root, key=lambda x:dic_node2weight.get(x,-np.inf),reverse=True)[0]
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

	lst_cluster = [None for i in range(len(set_leaf))]
	for cluster, root in enumerate(sorted(set_root, key=lambda x:dic_root2name[x])):
		set_child = {root}
		while True:
			if len(set_child) == 0:
				break
			target = set_child.pop()
			if target in set_leaf:
				lst_cluster[target] = cluster
				continue
			c1, c2 = dic_parent2childs[target]
			if c1 in set_leaf:
				lst_cluster[c1]= cluster
			else:
				set_child.add(c1)
			if c2 in set_leaf:
				lst_cluster[c2]= cluster
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
				lst_cluster[target] = cluster
				continue
			c1, c2 = dic_parent2childs[target]
			if c1 in set_leaf:
				lst_cluster[c1]= cluster
			else:
				set_child.add(c1)
			if c2 in set_leaf:
				lst_cluster[c2]= cluster
			else:
				set_child.add(c2)
	return lst_cluster

def computeSeparationGain(dic_key2value):
	#c11,c22,c12 = dic_key2value['countWeight1'],dic_key2value['countWeight2'],dic_key2value['countWeight12']
	w11,w22,w12 = dic_key2value['sumWeight1'],dic_key2value['sumWeight2'],dic_key2value['sumWeight12']
	n1, n2, = dic_key2value['nNode1'],dic_key2value['nNode2']
	if w12 == 0:
		return 1.0 - 0.1**20 * float(n1+n2)/(2*n1*n2) * 1.0/(-n1/float(n1+n2)*np.log2(n1/float(n1+n2))-n2/float(n1+n2)*np.log2(n2/float(n1+n2)))
	a= w12/(n1*n2)
	#b2= w12/c12/((w11+w22+n1+n2)/(c11+c22+n1+n2))
	#c2= w12/(n1*n2)/((w11+w22+n1+n2)/(n1*(n1+1)*0.5+n2*(n2+1)*0.5))
	return 1.0 - a

def K(c1,c2):
	if c1 < c2:
		return (c1,c2)
	else:
		return (c2,c1)

def cutTree(linkageMatrix, maxLeafID, t):
	dic_parent2childs={}
	dic_child2parent={}
	dic_node2weight={}
	dic_node2nNode={}
	set_leaf=set()
	for c, x in enumerate(linkageMatrix,maxLeafID+1):
		c1, c2, w, nNode = int(x[0]), int(x[1]), x[2], x[3]
		if c1 not in dic_parent2childs:
			set_leaf.add(c1)
		if c2 not in dic_parent2childs:
			set_leaf.add(c2)
		dic_child2parent[c1]=c
		dic_child2parent[c2]=c
		dic_parent2childs[c]={c1,c2}
		dic_node2weight[c]=(nNode,c)
		dic_node2nNode[c]=nNode
		root=c
	set_root={root}
	dic_root2name={root:'0'}
	while len(set_root) < t:
		target = sorted(set_root, key=lambda x:dic_node2weight.get(x,(1.0,0.0)),reverse=True)[0]
		set_root.remove(target)
		c1, c2 = dic_parent2childs[target]
		if dic_node2nNode.get(c1,1.0) < dic_node2nNode.get(c2,1.0):
			c1, c2 = c2, c1
		dic_root2name[c1] = dic_root2name[target]+'0'
		dic_root2name[c2] = dic_root2name[target]+'1'
		set_root.add(c1)
		set_root.add(c2)
	dic_leaf2cluster={}
	for cluster, root in enumerate(sorted(set_root, key=lambda x:dic_root2name[x])):
		set_child = {root}
		while True:
			if len(set_child) == 0:
				break
			target = set_child.pop()
			if target in set_leaf:
				dic_leaf2cluster[target] = cluster
				continue
			c1, c2 = dic_parent2childs[target]
			if c1 in set_leaf:
				dic_leaf2cluster[c1] = cluster
			else:
				set_child.add(c1)
			if c2 in set_leaf:
				dic_leaf2cluster[c2] = cluster
			else:
				set_child.add(c2)
	return dic_leaf2cluster

def delete(dic, key):
	if key in dic:
		del dic[key]

global_starttime=time.time()

def kMedoids(D, k, tmax=100, seed=None):
	if seed != None:
		#print seed
		np.random.seed(seed)
	# determine dimensions of distance matrix D
	m, n = D.shape

	if k >= n:
		M, C = [i for i in range(len(lst_gene))], dict([(i,i) for i in range(len(lst_gene))])
		return M, C
	# randomly initialize an array of k medoid indices
	M = np.arange(n)
	np.random.shuffle(M)
	M = np.sort(M[:k])

	# create a copy of the array of medoid indices
	Mnew = np.copy(M)

	# initialize a dictionary to represent clusters
	C = {}
	for t in xrange(tmax):
		# determine clusters, i. e. arrays of data indices
		J = np.argmin(D[:,M], axis=1)
		for kappa, mm in enumerate(M):
			J[mm]=kappa
		for kappa in range(k):
			C[kappa] = np.where(J==kappa)[0]
		# update cluster medoids
		for kappa in range(k):
			J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
			j = np.argmin(J)
			Mnew[kappa] = C[kappa][j]
		np.sort(Mnew)
		# check for convergence
		if np.array_equal(M, Mnew):
			break
		M = np.copy(Mnew)
	else:
		# final update of cluster memberships
		J = np.argmin(D[:,M], axis=1)
		for kappa in range(k):
			C[kappa] = np.where(J==kappa)[0]

	# return results
	return M, C

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

def partitionByMetis(lst_node, nPartition, wKNNgraph, isExact=False, recursive=True):
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
		#self.costFunction = lambda weightSum, nNode: weightSum*nNode
		self.costFunction = lambda weightSum, nNode: weightSum*nNode*nNode
		#self.costFunction = lambda weightSum, nNode: weightSum*np.sqrt(nNode)
		#self.costFunction = lambda weightSum, nNode: weightSum*np.log(nNode+1)
		#self.costFunction = lambda weightSum, nNode: np.log(weightSum+1)*nNode
		self.nTotalNode = nTotalNode
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

	def fit(self, linkageMatrix=[], set_forbiddenPair=set(), returnCandidate=False):
		nTotalNode=self.nTotalNode
		set_cluster=set(self.set_cluster)
		linkageMethod=self.linkageMethod
		dic_cluster2nNode = dict(self.dic_cluster2nNode)
		dic_cluster2cost = dict(self.dic_cluster2cost)
		dic_cluster2sumWeight = dict(self.dic_cluster2sumWeight)
		dic_edge2sumWeight = dict(self.dic_edge2sumWeight)
		if linkageMethod == 'single':
			dic_edge2maxWeight = dict(self.dic_edge2maxWeight)
		elif linkageMethod == 'complete':
			dic_edge2minWeight = dict(self.dic_edge2minWeight)
		elif linkageMethod == 'weighted':
			dic_edge2weightedWeight = dict(self.dic_edge2weightedWeight)
		elif linkageMethod == 'average':
			pass
		elif linkageMethod == 'deltaCost':
			dic_cluster2edgeSumWeight = dict(self.dic_cluster2edgeSumWeight)
		elif linkageMethod == 'deltaRev':
			pass
		elif linkageMethod == 'deltaModularity':
			dic_cluster2modularitySumWeight = dict(self.dic_cluster2modularitySumWeight)
			totalEdgeSumWeight = self.totalEdgeSumWeight
		elif linkageMethod == 'edgeCountNormalized':
			dic_edge2countWeight = dict(self.dic_edge2countWeight)
		elif linkageMethod == 'minEdgeCountNormalized':
			dic_edge2countWeight = dict(self.dic_edge2countWeight)
			dic_cluster2edgeSumCount = dict(self.dic_cluster2edgeSumCount)
		elif linkageMethod == 'maxEdgeCountNormalized':
			dic_edge2countWeight = dict(self.dic_edge2countWeight)
			dic_cluster2edgeSumCount = dict(self.dic_cluster2edgeSumCount)
		elif linkageMethod == 'unionEdgeCountNormalized':
			dic_edge2countWeight = dict(self.dic_edge2countWeight)
			dic_cluster2edgeSumCount = dict(self.dic_cluster2edgeSumCount)
		elif linkageMethod == 'minEdgeWeightNormalized':
			dic_cluster2edgeSumWeight = dict(self.dic_cluster2edgeSumWeight)
		elif linkageMethod == 'maxEdgeWeightNormalized':
			dic_cluster2edgeSumWeight = dict(self.dic_cluster2edgeSumWeight)
		elif linkageMethod == 'unionEdgeWeightNormalized':
			dic_cluster2edgeSumWeight = dict(self.dic_cluster2edgeSumWeight)
		elif linkageMethod == 'minModularityWeightNormalized':
			dic_cluster2modularitySumWeight = dict(self.dic_cluster2modularitySumWeight)
		elif linkageMethod == 'maxModularityWeightNormalized':
			dic_cluster2modularitySumWeight = dict(self.dic_cluster2modularitySumWeight)
		elif linkageMethod == 'avgModularityWeightNormalized':
			dic_cluster2modularitySumWeight = dict(self.dic_cluster2modularitySumWeight)

		max_i=0
		lst_target=[]
		dic_i2list={}
		if len(self.sorted_edge) > 0:
			dic_i2list[max_i]=list(self.sorted_edge)
			heapq.heappush(lst_target,(-dic_i2list[max_i][-1][1],max_i))
			max_i=max(dic_i2list.keys())+1

		self.linkageMatrix=[]

		for i, x in enumerate(linkageMatrix):
			c1,c2,new_cluster = x[0],x[1],x[2]
			nNode1, nNode2 = dic_cluster2nNode.get(c1,1.0), dic_cluster2nNode.get(c2,1.0)
			nNode = nNode1+nNode2
			sumWeight1, sumWeight2 = dic_cluster2sumWeight.get(c1,0.0), dic_cluster2sumWeight.get(c2,0.0)
			sumWeight12 = dic_edge2sumWeight.get(K(c1,c2),0.0)
			sumWeight = sumWeight1+sumWeight2+sumWeight12
			cost1, cost2 = dic_cluster2cost.get(c1,0.0), dic_cluster2cost.get(c2,0.0)
			cost12 = self.costFunction(dic_edge2sumWeight.get(K(c1,c2),0.0),nNode)
			cost = cost1+cost2+cost12
			dic_cluster2nNode[new_cluster] = nNode
			dic_cluster2sumWeight[new_cluster] = sumWeight
			dic_cluster2cost[new_cluster] = cost
			dic_key2value={'nNode1':nNode1,'nNode2':nNode2,'nNode':nNode,'sumWeight1':sumWeight1,'sumWeight2':sumWeight2,'sumWeight12':sumWeight12,'sumWeight':sumWeight,'cost':cost}
			separationGain=computeSeparationGain(dic_key2value)
			dic_key2value['separationGain']=separationGain
			self.linkageMatrix.append([c1,c2,new_cluster,dic_key2value])
			if linkageMethod == 'deltaCost':
				dic_cluster2edgeSumWeight[new_cluster]=dic_cluster2edgeSumWeight.get(c1,0.0)+dic_cluster2edgeSumWeight.get(c2,0.0) - dic_edge2sumWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'deltaModularity':
				dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)
			elif linkageMethod == 'minEdgeCountNormalized':
				dic_cluster2edgeSumCount[new_cluster] = dic_cluster2edgeSumCount.get(c1,0.0) + dic_cluster2edgeSumCount.get(c2,0.0) - dic_edge2countWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'maxEdgeCountNormalized':
				dic_cluster2edgeSumCount[new_cluster] = dic_cluster2edgeSumCount.get(c1,0.0) + dic_cluster2edgeSumCount.get(c2,0.0) - dic_edge2countWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'unionEdgeCountNormalized':
				dic_cluster2edgeSumCount[new_cluster] = dic_cluster2edgeSumCount.get(c1,0.0) + dic_cluster2edgeSumCount.get(c2,0.0) - dic_edge2countWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'minEdgeWeightNormalized':
				dic_cluster2edgeSumWeight[new_cluster] = dic_cluster2edgeSumWeight.get(c1,0.0) + dic_cluster2edgeSumWeight.get(c2,0.0) - dic_edge2sumWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'maxEdgeWeightNormalized':
				dic_cluster2edgeSumWeight[new_cluster] = dic_cluster2edgeSumWeight.get(c1,0.0) + dic_cluster2edgeSumWeight.get(c2,0.0) - dic_edge2sumWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'unionEdgeWeightNormalized':
				dic_cluster2edgeSumWeight[new_cluster] = dic_cluster2edgeSumWeight.get(c1,0.0) + dic_cluster2edgeSumWeight.get(c2,0.0) - dic_edge2sumWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'minModularityWeightNormalized':
				dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)
			elif linkageMethod == 'maxModularityWeightNormalized':
				dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)
			elif linkageMethod == 'avgModularityWeightNormalized':
				dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)

			set_cluster.remove(c1)
			set_cluster.remove(c2)
			lst_clusterPair2weight2=[]
			for cluster in set_cluster:
				e1, e2 =  K(c1,cluster), K(c2,cluster)
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight
				if linkageMethod == 'single':
					weight = max(dic_edge2maxWeight.get(e1,0.0),dic_edge2maxWeight.get(e2,0.0))
					if e1 in dic_edge2maxWeight:
						del dic_edge2maxWeight[e1]
					if e2 in dic_edge2maxWeight:
						del dic_edge2maxWeight[e2]
					dic_edge2maxWeight[new_e]=weight
				elif linkageMethod == 'complete':
					weight = min(dic_edge2minWeight.get(e1,np.inf),dic_edge2minWeight.get(e2,np.inf))
					if e1 in dic_edge2minWeight:
						del dic_edge2minWeight[e1]
					if e2 in dic_edge2minWeight:
						del dic_edge2minWeight[e2]
					dic_edge2minWeight[new_e]=weight
				elif linkageMethod == 'weighted':
					weight = 0.5*(dic_edge2weightedWeight.get(e1,0.0)+dic_edge2weightedWeight.get(e2,0.0))
					if e1 in dic_edge2weightedWeight:
						del dic_edge2weightedWeight[e1]
					if e2 in dic_edge2weightedWeight:
						del dic_edge2weightedWeight[e2]
					dic_edge2weightedWeight[new_e]=weight
				elif linkageMethod == 'average':
					weight = sumWeight
					weight /= float(dic_cluster2nNode.get(new_cluster,1.0)*dic_cluster2nNode.get(cluster,1.0))
				elif linkageMethod == 'deltaCost':
					weight = 1.0 / (dic_cluster2edgeSumWeight.get(new_cluster,0.0) * dic_cluster2nNode.get(cluster,1.0) + dic_cluster2edgeSumWeight.get(cluster,0.0) * dic_cluster2nNode.get(new_cluster,1.0) - sumWeight * float(dic_cluster2nNode.get(new_cluster,1.0)*dic_cluster2nNode.get(cluster,1.0)))
				elif linkageMethod == 'deltaRev':
					weight = sumWeight
					weight /= float(dic_cluster2nNode.get(new_cluster,1.0)*dic_cluster2nNode.get(cluster,1.0))
					weight *= nTotalNode-dic_cluster2nNode.get(new_cluster,1.0)-dic_cluster2nNode.get(cluster,1.0)
				elif linkageMethod == 'deltaModularity':
					weight = sumWeight
					weight -= dic_cluster2modularitySumWeight.get(new_cluster,0.0)*dic_cluster2modularitySumWeight.get(cluster,0.0)/totalEdgeSumWeight
				elif linkageMethod == 'edgeCountNormalized':
					countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
					if e1 in dic_edge2countWeight:
						del dic_edge2countWeight[e1]
					if e2 in dic_edge2countWeight:
						del dic_edge2countWeight[e2]
					dic_edge2countWeight[new_e]=countWeight
					weight = sumWeight/countWeight
				elif linkageMethod == 'minEdgeCountNormalized':
					countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
					if e1 in dic_edge2countWeight:
						del dic_edge2countWeight[e1]
					if e2 in dic_edge2countWeight:
						del dic_edge2countWeight[e2]
					dic_edge2countWeight[new_e]=countWeight
					weight = sumWeight
					weight /= min(dic_cluster2edgeSumCount.get(new_cluster,0.0),dic_cluster2edgeSumCount.get(cluster,0.0))
				elif linkageMethod == 'maxEdgeCountNormalized':
					countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
					if e1 in dic_edge2countWeight:
						del dic_edge2countWeight[e1]
					if e2 in dic_edge2countWeight:
						del dic_edge2countWeight[e2]
					dic_edge2countWeight[new_e]=countWeight
					weight = sumWeight
					weight /= max(dic_cluster2edgeSumCount.get(new_cluster,0.0),dic_cluster2edgeSumCount.get(cluster,0.0))
				elif linkageMethod == 'unionEdgeCountNormalized':
					countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
					if e1 in dic_edge2countWeight:
						del dic_edge2countWeight[e1]
					if e2 in dic_edge2countWeight:
						del dic_edge2countWeight[e2]
					dic_edge2countWeight[new_e]=countWeight
					weight = sumWeight
					weight /= dic_cluster2edgeSumCount.get(new_cluster,0.0)+dic_cluster2edgeSumCount.get(cluster,0.0)-countWeight
				elif linkageMethod == 'minEdgeWeightNormalized':
					weight = sumWeight
					weight /= min(dic_cluster2edgeSumWeight.get(new_cluster,0.0),dic_cluster2edgeSumWeight.get(cluster,0.0))
				elif linkageMethod == 'maxEdgeWeightNormalized':
					weight = sumWeight
					weight /= max(dic_cluster2edgeSumWeight.get(new_cluster,0.0),dic_cluster2edgeSumWeight.get(cluster,0.0))
				elif linkageMethod == 'unionEdgeWeightNormalized':
					weight = sumWeight
					weight /= dic_cluster2edgeSumWeight.get(new_cluster,0.0)+dic_cluster2edgeSumWeight.get(cluster,0.0)-sumWeight
				elif linkageMethod == 'minModularityWeightNormalized':
					weight = sumWeight
					weight /= min(dic_cluster2modularitySumWeight.get(new_cluster,0.0),dic_cluster2modularitySumWeight.get(cluster,0.0))
				elif linkageMethod == 'maxModularityWeightNormalized':
					weight = sumWeight
					weight /= max(dic_cluster2modularitySumWeight.get(new_cluster,0.0),dic_cluster2modularitySumWeight.get(cluster,0.0))
				elif linkageMethod == 'avgModularityWeightNormalized':
					weight = sumWeight
					weight /= 0.5*(dic_cluster2modularitySumWeight.get(new_cluster,0.0)+dic_cluster2modularitySumWeight.get(cluster,0.0))
				elif linkageMethod == 'minPossibleEdgeCountNormalized':
					minNNode=min(dic_cluster2nNode.get(new_cluster,1.0),dic_cluster2nNode.get(cluster,1.0))
					weight = sumWeight
					weight /= minNNode*(nTotalNode-minNNode)
				elif linkageMethod == 'minNodeNormalized':
					minNNode=min(dic_cluster2nNode.get(new_cluster,1.0),dic_cluster2nNode.get(cluster,1.0))
					weight = sumWeight
					weight /= minNNode
				lst_clusterPair2weight2.append([K(new_cluster, cluster), weight])
			if len(lst_clusterPair2weight2) > 0:
				lst_clusterPair2weight2.sort(key=lambda x: x[1])
				dic_i2list[max_i]=lst_clusterPair2weight2
				heapq.heappush(lst_target,(-dic_i2list[max_i][-1][1],max_i))
				max_i+=1
			set_cluster.add(new_cluster)

		if returnCandidate:
			lst_candidate=[]
			for candidates in dic_i2list.values():
				lst_candidate+=[[(c1,c2),weight] for (c1,c2),weight in candidates if c1 in set_cluster and c2 in set_cluster]
			lst_candidate.sort(key=lambda x:x[1])
			return lst_candidate

		if len(set_cluster) <= 1:
			return
		new_cluster = max(set_cluster) + 1
		while True:
			if len(lst_target) != 0:
				negweight, i = heapq.heappop(lst_target)
				(c1, c2), weight = dic_i2list[i].pop()
				if c1 not in set_cluster or c2 not in set_cluster:
					while True:
						if len(dic_i2list[i]) == 0:
							break
						(c1, c2), weight = dic_i2list[i].pop()
						if c1 in set_cluster and c2 in set_cluster:
							dic_i2list[i].append([(c1,c2), weight])
							break
					if len(dic_i2list[i]) == 0:
						del dic_i2list[i]
					else:
						heapq.heappush(lst_target,(-dic_i2list[i][-1][1],i))
					continue
				if len(dic_i2list[i]) == 0:
					del dic_i2list[i]
				else:
					heapq.heappush(lst_target,(-dic_i2list[i][-1][1],i))
				if K(c1, c2) in set_forbiddenPair:
					continue
			else:
				(c1, c2), weight = sorted(set_cluster, key=lambda x:dic_cluster2nNode.get(x,1.0))[0:2], 0.0

			nNode1, nNode2 = dic_cluster2nNode.get(c1,1.0), dic_cluster2nNode.get(c2,1.0)
			nNode = nNode1+nNode2
			sumWeight1, sumWeight2 = dic_cluster2sumWeight.get(c1,0.0), dic_cluster2sumWeight.get(c2,0.0)
			sumWeight12 = dic_edge2sumWeight.get(K(c1,c2),0.0)
			sumWeight = sumWeight1+sumWeight2+sumWeight12
			cost1, cost2 = dic_cluster2cost.get(c1,0.0), dic_cluster2cost.get(c2,0.0)
			cost12 = self.costFunction(dic_edge2sumWeight.get(K(c1,c2),0.0),nNode)
			cost = cost1+cost2+cost12
			dic_cluster2nNode[new_cluster] = nNode
			dic_cluster2sumWeight[new_cluster] = sumWeight
			dic_cluster2cost[new_cluster] = cost
			dic_key2value={'nNode1':nNode1,'nNode2':nNode2,'nNode':nNode,'sumWeight1':sumWeight1,'sumWeight2':sumWeight2,'sumWeight12':sumWeight12,'sumWeight':sumWeight,'cost':cost}
			separationGain=computeSeparationGain(dic_key2value)
			dic_key2value['separationGain']=separationGain
			self.linkageMatrix.append([c1,c2,new_cluster,dic_key2value])
			if linkageMethod == 'deltaModularity':
				dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)
			elif linkageMethod == 'minEdgeCountNormalized':
				dic_cluster2edgeSumCount[new_cluster] = dic_cluster2edgeSumCount.get(c1,0.0) + dic_cluster2edgeSumCount.get(c2,0.0) - dic_edge2countWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'maxEdgeCountNormalized':
				dic_cluster2edgeSumCount[new_cluster] = dic_cluster2edgeSumCount.get(c1,0.0) + dic_cluster2edgeSumCount.get(c2,0.0) - dic_edge2countWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'unionEdgeCountNormalized':
				dic_cluster2edgeSumCount[new_cluster] = dic_cluster2edgeSumCount.get(c1,0.0) + dic_cluster2edgeSumCount.get(c2,0.0) - dic_edge2countWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'minEdgeWeightNormalized':
				dic_cluster2edgeSumWeight[new_cluster] = dic_cluster2edgeSumWeight.get(c1,0.0) + dic_cluster2edgeSumWeight.get(c2,0.0) - dic_edge2sumWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'maxEdgeWeightNormalized':
				dic_cluster2edgeSumWeight[new_cluster] = dic_cluster2edgeSumWeight.get(c1,0.0) + dic_cluster2edgeSumWeight.get(c2,0.0) - dic_edge2sumWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'unionEdgeWeightNormalized':
				dic_cluster2edgeSumWeight[new_cluster] = dic_cluster2edgeSumWeight.get(c1,0.0) + dic_cluster2edgeSumWeight.get(c2,0.0) - dic_edge2sumWeight.get(K(c1,c2),0.0)
			elif linkageMethod == 'minModularityWeightNormalized':
				dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)
			elif linkageMethod == 'maxModularityWeightNormalized':
				dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)
			elif linkageMethod == 'avgModularityWeightNormalized':
				dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)

			set_cluster.remove(c1)
			set_cluster.remove(c2)
			lst_clusterPair2weight2=[]
			for cluster in set_cluster:
				e1, e2 =  K(c1,cluster), K(c2,cluster)
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight
				if linkageMethod == 'single':
					weight = max(dic_edge2maxWeight.get(e1,0.0),dic_edge2maxWeight.get(e2,0.0))
					if e1 in dic_edge2maxWeight:
						del dic_edge2maxWeight[e1]
					if e2 in dic_edge2maxWeight:
						del dic_edge2maxWeight[e2]
					dic_edge2maxWeight[new_e]=weight
				elif linkageMethod == 'complete':
					weight = min(dic_edge2minWeight.get(e1,np.inf),dic_edge2minWeight.get(e2,np.inf))
					if e1 in dic_edge2minWeight:
						del dic_edge2minWeight[e1]
					if e2 in dic_edge2minWeight:
						del dic_edge2minWeight[e2]
					dic_edge2minWeight[new_e]=weight
				elif linkageMethod == 'weighted':
					weight = 0.5*(dic_edge2weightedWeight.get(e1,0.0)+dic_edge2weightedWeight.get(e2,0.0))
					if e1 in dic_edge2weightedWeight:
						del dic_edge2weightedWeight[e1]
					if e2 in dic_edge2weightedWeight:
						del dic_edge2weightedWeight[e2]
					dic_edge2weightedWeight[new_e]=weight
				elif linkageMethod == 'average':
					weight = sumWeight
					weight /= float(dic_cluster2nNode.get(new_cluster,1.0)*dic_cluster2nNode.get(cluster,1.0))
				elif linkageMethod == 'deltaCost':
					weight = 1.0 / (dic_cluster2sumWeight.get(new_cluster,0.0) * dic_cluster2nNode.get(cluster,1.0) + dic_cluster2sumWeight.get(cluster,0.0) * dic_cluster2nNode.get(new_cluster,1.0) - sumWeight * float(dic_cluster2nNode.get(new_cluster,1.0)*dic_cluster2nNode.get(cluster,1.0)))
				elif linkageMethod == 'deltaRev':
					weight = sumWeight
					weight /= float(dic_cluster2nNode.get(new_cluster,1.0)*dic_cluster2nNode.get(cluster,1.0))
					weight *= nTotalNode-dic_cluster2nNode.get(new_cluster,1.0)-dic_cluster2nNode.get(cluster,1.0)
				elif linkageMethod == 'deltaModularity':
					weight = sumWeight
					weight -= dic_cluster2modularitySumWeight.get(new_cluster,0.0)*dic_cluster2modularitySumWeight.get(cluster,0.0)/totalEdgeSumWeight
				elif linkageMethod == 'edgeCountNormalized':
					countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
					if e1 in dic_edge2countWeight:
						del dic_edge2countWeight[e1]
					if e2 in dic_edge2countWeight:
						del dic_edge2countWeight[e2]
					dic_edge2countWeight[new_e]=countWeight
					weight = sumWeight/countWeight
				elif linkageMethod == 'minEdgeCountNormalized':
					countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
					if e1 in dic_edge2countWeight:
						del dic_edge2countWeight[e1]
					if e2 in dic_edge2countWeight:
						del dic_edge2countWeight[e2]
					dic_edge2countWeight[new_e]=countWeight
					weight = sumWeight
					weight /= min(dic_cluster2edgeSumCount.get(new_cluster,0.0),dic_cluster2edgeSumCount.get(cluster,0.0))
				elif linkageMethod == 'maxEdgeCountNormalized':
					countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
					if e1 in dic_edge2countWeight:
						del dic_edge2countWeight[e1]
					if e2 in dic_edge2countWeight:
						del dic_edge2countWeight[e2]
					dic_edge2countWeight[new_e]=countWeight
					weight = sumWeight
					weight /= max(dic_cluster2edgeSumCount.get(new_cluster,0.0),dic_cluster2edgeSumCount.get(cluster,0.0))
				elif linkageMethod == 'unionEdgeCountNormalized':
					countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
					if e1 in dic_edge2countWeight:
						del dic_edge2countWeight[e1]
					if e2 in dic_edge2countWeight:
						del dic_edge2countWeight[e2]
					dic_edge2countWeight[new_e]=countWeight
					weight = sumWeight
					weight /= dic_cluster2edgeSumCount.get(new_cluster,0.0)+dic_cluster2edgeSumCount.get(cluster,0.0)-countWeight
				elif linkageMethod == 'minEdgeWeightNormalized':
					weight = sumWeight
					weight /= min(dic_cluster2edgeSumWeight.get(new_cluster,0.0),dic_cluster2edgeSumWeight.get(cluster,0.0))
				elif linkageMethod == 'maxEdgeWeightNormalized':
					weight = sumWeight
					weight /= max(dic_cluster2edgeSumWeight.get(new_cluster,0.0),dic_cluster2edgeSumWeight.get(cluster,0.0))
				elif linkageMethod == 'unionEdgeWeightNormalized':
					weight = sumWeight
					weight /= dic_cluster2edgeSumWeight.get(new_cluster,0.0)+dic_cluster2edgeSumWeight.get(cluster,0.0)-sumWeight
				elif linkageMethod == 'minModularityWeightNormalized':
					weight = sumWeight
					weight /= min(dic_cluster2modularitySumWeight.get(new_cluster,0.0),dic_cluster2modularitySumWeight.get(cluster,0.0))
				elif linkageMethod == 'maxModularityWeightNormalized':
					weight = sumWeight
					weight /= max(dic_cluster2modularitySumWeight.get(new_cluster,0.0),dic_cluster2modularitySumWeight.get(cluster,0.0))
				elif linkageMethod == 'avgModularityWeightNormalized':
					weight = sumWeight
					weight /= 0.5*(dic_cluster2modularitySumWeight.get(new_cluster,0.0)+dic_cluster2modularitySumWeight.get(cluster,0.0))
				elif linkageMethod == 'minPossibleEdgeCountNormalized':
					minNNode=min(dic_cluster2nNode.get(new_cluster,1.0),dic_cluster2nNode.get(cluster,1.0))
					weight = sumWeight
					weight /= minNNode*(nTotalNode-minNNode)
				elif linkageMethod == 'minNodeNormalized':
					minNNode=min(dic_cluster2nNode.get(new_cluster,1.0),dic_cluster2nNode.get(cluster,1.0))
					weight = sumWeight
					weight /= minNNode
				lst_clusterPair2weight2.append([K(new_cluster, cluster), weight])
			if len(lst_clusterPair2weight2) > 0:
				lst_clusterPair2weight2.sort(key=lambda x: x[1])
				dic_i2list[max_i]=lst_clusterPair2weight2
				heapq.heappush(lst_target,(-dic_i2list[max_i][-1][1],max_i))
				max_i+=1
			set_cluster.add(new_cluster)
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
			for node in enumerate(self.set_node):
				self.dic_node2chunk[node]=0
			return

		if chunkingMethod == 'average':
			import BaselineHC_fast
			linkageMatrix = BaselineHC_fast.fit(self.wKNNgraph.dic_node2node_weight, linkageMethod=chunkingMethod)
			dic_node2cluster = cutTree(linkageMatrix, maxLeafID=max(self.set_node), t=nChunk)
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
			dic_node2cluster = cutTree(linkageMatrix, maxLeafID=max(self.set_node), t=nChunk)
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

				dic_shatteredNode2clusterWeightSum={}
				for node in set_shatteredNode:
					dic_c2w={}
					for n2, weight in self.wKNNgraph.dic_node2node_weight[node].items():
						c2 = dic_node2cluster[n2]
						if c2 == -1:
							continue
						if c2 not in dic_c2w:
							dic_c2w[c2]=0.0
						dic_c2w[c2]+=weight
						#dic_c2w[c2]=max(dic_c2w[c2],weight)
					if len(dic_c2w) == 0:
						weightSum = 0.0
					else:
						weightSum = max(dic_c2w.values())
					dic_shatteredNode2clusterWeightSum[node]=[weightSum, dic_c2w]
				lst_shatteredNode=list(set_shatteredNode)
				while True:
					if len(lst_shatteredNode) == 0:
						break
					lst_shatteredNode.sort(key=lambda x:dic_shatteredNode2clusterWeightSum[x][0],reverse=True)
					node = lst_shatteredNode.pop(0)
					weightSum, dic_c2w = dic_shatteredNode2clusterWeightSum[node]
					c = sorted(dic_c2w.items(),key=lambda x:x[1], reverse=True)[0][0]
					dic_cluster2nodes[c].add(node)
					dic_node2cluster[node]=c
					for n2, weight in self.wKNNgraph.dic_node2node_weight[node].items():
						c2 = dic_node2cluster[n2]
						if c2 != -1:
							continue
						weightSum, dic_c2w = dic_shatteredNode2clusterWeightSum[n2]
						if c not in dic_c2w:
							dic_c2w[c]=0.0
						dic_c2w[c]+=weight
						#dic_c2w[c]=max(dic_c2w[c],weight)
						weightSum = max(dic_c2w.values())
						dic_shatteredNode2clusterWeightSum[n2] = weightSum, dic_c2w

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

			dic_shatteredNode2clusterWeightSum={}
			for node in set_shatteredNode:
				dic_c2w={}
				for n2, weight in self.wKNNgraph.dic_node2node_weight[node].items():
					c2 = dic_node2cluster[n2]
					if c2 == -1:
						continue
					if c2 not in dic_c2w:
						dic_c2w[c2]=0.0
					dic_c2w[c2]+=weight
					#dic_c2w[c2]=max(dic_c2w[c2],weight)
				if len(dic_c2w) == 0:
					weightSum = 0.0
				else:
					weightSum = max(dic_c2w.values())
				dic_shatteredNode2clusterWeightSum[node]=[weightSum, dic_c2w]
			lst_shatteredNode=list(set_shatteredNode)
			while True:
				if len(lst_shatteredNode) == 0:
					break
				lst_shatteredNode.sort(key=lambda x:dic_shatteredNode2clusterWeightSum[x][0],reverse=True)
				node = lst_shatteredNode.pop(0)
				weightSum, dic_c2w = dic_shatteredNode2clusterWeightSum[node]
				cluster = sorted(dic_c2w.items(),key=lambda x:x[1], reverse=True)[0][0]
				dic_cluster2nodes[cluster].add(node)
				dic_node2cluster[node]=cluster
				for n2, weight in self.wKNNgraph.dic_node2node_weight[node].items():
					c2 = dic_node2cluster[n2]
					if c2 != -1:
						continue
					weightSum, dic_c2w = dic_shatteredNode2clusterWeightSum[n2]
					if cluster not in dic_c2w:
						dic_c2w[cluster]=0.0
					dic_c2w[cluster]+=weight
					#dic_c2w[cluster]=max(dic_c2w[cluster],weight)
					weightSum = max(dic_c2w.values())
					dic_shatteredNode2clusterWeightSum[n2] = weightSum, dic_c2w

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
			if self.bestLinkageMatrix == None:
				linkageMatrix=[]
				set_forbiddenPair=set()
			else:
				linkageMatrix=list(self.bestLinkageMatrix[:currentRound])
				set_forbiddenPair=set()
				set_forbiddenPair.add(K(self.bestLinkageMatrix[currentRound][0],self.bestLinkageMatrix[currentRound][1]))
			for i, agent in enumerate(lst_agent):
				#if len(linkageMatrix) == 0 or K(agent.linkageMatrix[currentRound][0],agent.linkageMatrix[currentRound][1]) != K(self.bestLinkageMatrix[currentRound][0],self.bestLinkageMatrix[currentRound][1]):
				#agent.fit(linkageMatrix,set_forbiddenPair=set_forbiddenPair)
				agent.fit(linkageMatrix)
				if agent.linkageMatrix[-1][3]['cost'] < bestCost:
					#print 'here1'
					lst_best_i=[i]
					bestCost=agent.linkageMatrix[-1][3]['cost']
					self.bestLinkageMatrix=list(agent.linkageMatrix)
					bestMethod=agent.linkageMethod
				elif agent.linkageMatrix[-1][3]['cost'] == bestCost:
					lst_best_i.append(i)
					bestMethod+=';'+agent.linkageMethod
			#print 'herex', lst_best_i, bestMethod, bestCost
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
				lst_agent[best_i].fit(linkageMatrix,set_forbiddenPair=set_forbiddenPair)
				if lst_agent[best_i].linkageMatrix[-1][3]['cost'] < bestCost:
					#print 'here2'
					bestCost=lst_agent[best_i].linkageMatrix[-1][3]['cost']
					self.bestLinkageMatrix=list(lst_agent[best_i].linkageMatrix)
				set_forbiddenPair2=set()
				for x in lst_agent[best_i].linkageMatrix[currentRound:]:
					pair = K(x[0],x[1])
					if pair[0] in set_forbiddenNode or pair[1] in set_forbiddenNode:
						if pair not in set_forbiddenPair:
							set_forbiddenPair2.add(pair)
				if len(set_forbiddenPair2) == 0:
					#print 'ooooo'
					break
				set_forbiddenPair |= set_forbiddenPair2
			lst_agent[best_i].linkageMatrix = list(self.bestLinkageMatrix)
			#print 'herey', lst_best_i, bestMethod, bestCost
			self.bestMethod.append(bestMethod)
			set_cluster.remove(self.bestLinkageMatrix[currentRound][0])
			set_cluster.remove(self.bestLinkageMatrix[currentRound][1])
			set_cluster.add(self.bestLinkageMatrix[currentRound][2])
			new_cluster+=1
			currentRound+=1
		print 'asdf',sorted(dic_method2score.items(),key=lambda x:x[1],reverse=True)
		return

	def membershipRefinement(self):
		return

	def fit(self, nChunk='auto', chunkingMethod='metis', skipEnsembleRatio=None): 
		#assert(nChunk == 'auto' or type(nChunk)==int),(
		#	"nChunk must be 'auto' or int")
		assert(self.wKNNgraph != None),(
			"wKNNgraph must not be set before fitting")
		nNode = len(self.set_node)
		if nChunk == 'auto':
			nChunk=80+int(np.log2(nNode))
		else:
			nChunk=int(nChunk)+int(np.log2(nNode))
		if skipEnsembleRatio != None and self.dic_chunk2nodes != None and len(self.dic_chunk2nodes) > nChunk*skipEnsembleRatio and self.bestLinkageMatrix != None and len(self.bestLinkageMatrix) > 1:
			print 'kkk chunkEnsemble skipped'
			ensemble = False
			pass
		else:
			print self.ID, 'chunking'
			self.chunking(nChunk, chunkingMethod=chunkingMethod)
			#for node in self.set_node:
			#	print '\t'.join(map(str,[self.wKNNgraph.dic_node2name[node],'_'.join(map(str,[self.ID,self.dic_node2chunk[node]]))]))
			print self.ID, 'ensembleClustering'
			self.chunkEnsembleClustering()
			ensemble = True
		#print self.ID, 'membershipRefinement'
		#self.membershipRefinement()
		return ensemble

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

	def fit(self, X, k, K=10, nChunk='auto', outlierRatio=0.09, chunkingMethod='metis', Xtype='graph'):
		#chunkingMethod='average'
		#chunkingMethod='minModularityWeightNormalized'
		nIteration=2+int(np.sqrt(k))
		wKNNgraph = X
		assert(Xtype in ['graph', 'similarity', 'dissimilarity']),(
			"Xtype must be one of the 'graph', 'similarity', 'dissimilarity'")
		'''
		wKNNgraph = WKNNgraph()
		if Xtype == 'graph':
			wKNNgraph.loadGraph(filename=X)
		elif Xtype == 'similarity':
			wKNNgraph.constructGraph(filename=X, weightType='similarity', nNeighbor=K)
		elif Xtype == 'dissimilarity':
			wKNNgraph.constructGraph(filename=X, weightType='dissimilarity', nNeighbor=K)
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
		'''
		subtree.fit(nChunk=nChunk, chunkingMethod=chunkingMethod)
		tmplinkageMatrix=subtree.fullLinkageMatrix(internalNodePrefix='') ###
		self.linkageMatrix = arrangeLinkageMatrix(tmplinkageMatrix)###
		self.membership = cutTree2(self.linkageMatrix,k,outlierRatio=0.09)###
		return###
		'''
		iteration=0
		lst_subtree.append(subtree)
		while True:
			if iteration >= nIteration or len(lst_subtree)==nNode:
				break
			subtree=lst_subtree.pop(0)
			#iteration += int(subtree.fit(nChunk=nChunk, chunkingMethod=chunkingMethod, skipEnsembleRatio=1.0-outlierRatio))
			iteration += int(subtree.fit(nChunk=nChunk, chunkingMethod=chunkingMethod))
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
			print subtree1.ID,subtree2.ID,subtree.ID, len(subtree.set_node), len(subtree1.set_node), len(subtree2.set_node)
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
		self.membership = cutTree2(self.linkageMatrix,k,outlierRatio=outlierRatio)###
		return
		
if __name__ == "__main__":
	parser=argparse.ArgumentParser(
		usage='''\
	%(prog)s [options] weightfile k
	example: %(prog)s weightfile k -o out.txt
	''')
	
	parser.add_argument('weightfile', help='distance or weight file')
	parser.add_argument('k', type=int, help='the number of cluster')
	parser.add_argument('-weightType', required=False, default='graph', choices=['graph', 'similarity', 'dissimilarity'], help='weightType')
	parser.add_argument('-K', required=False, type=int, default=10, help='K-nearest neighbor graph')
	parser.add_argument('-M', required=False, default='auto', help='the number of chunks')
	parser.add_argument('-alpha', type=float, required=False, default=0.9, help='the number of initcluster')
	parser.add_argument('-chunkingMethod', required=False, default='metis', choices=['metis', 'average'], help='chunking method')
	parser.add_argument('-o', dest='outfile', required=False, metavar='str', default='stdout', help='outfile')
	args=parser.parse_args()
	
	idea = IDEA()
	idea.fit(X=args.weightfile, k=args.k, K=args.K, M=args.M, chunkingMethod=chunkingMethod, Xtype=args.weightType)

	if args.outfile == 'stdout':
		OF=sys.stdout
	else:
		OF=open(args.outfile,'w')

	print '\n'.join(map(str,idea.linkageMatrix))
	for nodename, membership in zip(idea.lst_name, idea.membership):
		OF.write('\t'.join(map(str,[nodename,membership]))+'\n')
