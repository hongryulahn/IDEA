#!/usr/bin/env python
import numpy as np
import itertools
from heapq import *

def fit(dic_node2node_weight, linkageMethod):
	assert(linkageMethod in ['single', 'complete','weighted','average','deltaCost','deltaRev','deltaModularity','edgeCountNormalized','minEdgeCountNormalized','maxEdgeCountNormalized','unionEdgeCountNormalized','minEdgeWeightNormalized','maxEdgeWeightNormalized','unionEdgeWeightNormalized','minModularityWeightNormalized','maxModularityWeightNormalized','avgModularityWeightNormalized','minPossibleEdgeCountNormalized','minNodeNormalized']),(
		"linkageMethod must be in ['single','complete','weighted','average','deltaCost','deltaRev','deltaModularity','edgeCountNormalized','minEdgeCountNormalized','maxEdgeCountNormalized','unionEdgeCountNormalized','minEdgeWeightNormalized','maxEdgeWeightNormalized','unionEdgeWeightNormalized','minModularityWeightNormalized','maxModularityWeightNormalized','avgModularityWeightNormalized','minPossibleEdgeCountNormalized','minNodeNormalized']")
	def K(c1,c2):
		if c1 < c2:
			return (c1,c2)
		else:
			return (c2,c1)
	set_cluster=set(dic_node2node_weight.keys())
	nTotalNode=len(set_cluster)
	dic_cluster2nNode={}
	if linkageMethod == 'single':
		dic_edge2maxWeight = {}
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2maxWeight[c1,c2]=weight
		sorted_edge=sorted(dic_edge2maxWeight.items(),key=lambda x:x[1])
	elif linkageMethod == 'complete':
		dic_edge2minWeight = {}
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2minWeight[c1,c2]=weight
		sorted_edge=sorted(dic_edge2minWeight.items(),key=lambda x:x[1])
	elif linkageMethod == 'weighted':
		dic_edge2weightedWeight = {}
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2weightedWeight[c1,c2]=weight
		sorted_edge=sorted(dic_edge2weightedWeight.items(),key=lambda x:x[1])
	elif linkageMethod == 'average':
		dic_edge2sumWeight = {}
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
		sorted_edge=sorted(dic_edge2sumWeight.items(),key=lambda x:x[1])
	elif linkageMethod == 'deltaCost':
		dic_cluster2edgeSumWeight = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2edgeSumWeight[c1]=sum(dic_node2node_weight[c1].values())
		dic_edge2sumWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					weight = 1.0 / (dic_cluster2edgeSumWeight.get(c1,0.0) * dic_cluster2nNode.get(c2,1.0) + dic_cluster2edgeSumWeight.get(c2,0.0) * dic_cluster2nNode.get(c1,1.0) - weight * float(dic_cluster2nNode.get(c1,1.0)*dic_cluster2nNode.get(c2,1.0)))
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'deltaRev':
		dic_cluster2edgeSumWeight = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2edgeSumWeight[c1]=sum(dic_node2node_weight[c1].values())
		dic_edge2sumWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					weight /= float(dic_cluster2nNode.get(c1,1.0)*dic_cluster2nNode.get(c2,1.0))
					weight *= nTotalNode-dic_cluster2nNode.get(c1,1.0)-dic_cluster2nNode.get(c2,1.0)
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'deltaModularity':
		totalEdgeSumWeight=0.0
		dic_cluster2modularitySumWeight = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2modularitySumWeight[c1]=sum(dic_node2node_weight[c1].values())
			totalEdgeSumWeight+=dic_cluster2modularitySumWeight[c1]
		dic_edge2sumWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					weight /= float(dic_cluster2nNode.get(c1,1.0)*dic_cluster2nNode.get(c2,1.0))
					weight *= nTotalNode-dic_cluster2nNode.get(c1,1.0)-dic_cluster2nNode.get(c2,1.0)
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'edgeCountNormalized':
		dic_edge2sumWeight = {}
		dic_edge2countWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					dic_edge2countWeight[(c1,c2)]=1.0
					weight /= dic_edge2countWeight.get((c1,c2),0.0)
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'minEdgeCountNormalized':
		dic_cluster2edgeSumCount = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2edgeSumCount[c1]=0.0
			for c2 in dic_node2node_weight[c1].keys():
				dic_cluster2edgeSumCount[c1]+=1.0
		dic_edge2sumWeight = {}
		dic_edge2countWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					dic_edge2countWeight[(c1,c2)]=1.0
					weight /= min(dic_cluster2edgeSumCount.get(c1,0.0),dic_cluster2edgeSumCount.get(c2,0.0))
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'maxEdgeCountNormalized':
		dic_cluster2edgeSumCount = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2edgeSumCount[c1]=0.0
			for c2 in dic_node2node_weight[c1].keys():
				dic_cluster2edgeSumCount[c1]+=1.0
		dic_edge2sumWeight = {}
		dic_edge2countWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					dic_edge2countWeight[(c1,c2)]=1.0
					weight /= max(dic_cluster2edgeSumCount.get(c1,0.0),dic_cluster2edgeSumCount.get(c2,0.0))
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'unionEdgeCountNormalized':
		dic_cluster2edgeSumCount = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2edgeSumCount[c1]=0.0
			for c2 in dic_node2node_weight[c1].keys():
				dic_cluster2edgeSumCount[c1]+=1.0
		dic_edge2sumWeight = {}
		dic_edge2countWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					dic_edge2countWeight[(c1,c2)]=1.0
					weight /= dic_cluster2edgeSumCount.get(c1,0.0)+dic_cluster2edgeSumCount.get(c2,0.0)-dic_edge2countWeight.get((c1,c2),0.0)
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'minEdgeWeightNormalized':
		dic_cluster2edgeSumWeight = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2edgeSumWeight[c1]=sum(dic_node2node_weight[c1].values())
		dic_edge2sumWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					weight /= min(dic_cluster2edgeSumWeight.get(c1,0.0),dic_cluster2edgeSumWeight.get(c2,0.0))
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'maxEdgeWeightNormalized':
		dic_cluster2edgeSumWeight = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2edgeSumWeight[c1]=sum(dic_node2node_weight[c1].values())
		dic_edge2sumWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					weight /= max(dic_cluster2edgeSumWeight.get(c1,0.0),dic_cluster2edgeSumWeight.get(c2,0.0))
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'unionEdgeWeightNormalized':
		dic_cluster2edgeSumWeight = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2edgeSumWeight[c1]=sum(dic_node2node_weight[c1].values())
		dic_edge2sumWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					weight /= dic_cluster2edgeSumWeight.get(c1,0.0)+dic_cluster2edgeSumWeight.get(c2,0.0)-weight
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'minModularityWeightNormalized':
		dic_cluster2modularitySumWeight = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2modularitySumWeight[c1]=sum(dic_node2node_weight[c1].values())
		dic_edge2sumWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					weight /= min(dic_cluster2modularitySumWeight.get(c1,0.0),dic_cluster2modularitySumWeight.get(c2,0.0))
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'maxModularityWeightNormalized':
		dic_cluster2modularitySumWeight = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2modularitySumWeight[c1]=sum(dic_node2node_weight[c1].values())
		dic_edge2sumWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					weight /= max(dic_cluster2modularitySumWeight.get(c1,0.0),dic_cluster2modularitySumWeight.get(c2,0.0))
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'avgModularityWeightNormalized':
		dic_cluster2modularitySumWeight = {}
		for c1 in dic_node2node_weight.keys():
			dic_cluster2modularitySumWeight[c1]=sum(dic_node2node_weight[c1].values())
		dic_edge2sumWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					weight /= 0.5*(dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0))
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'minPossibleEdgeCountNormalized':
		dic_edge2sumWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					minNNode=min(dic_cluster2nNode.get(c1,1.0),dic_cluster2nNode.get(c2,1.0))
					weight /= minNNode*(nTotalNode-minNNode)
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])
	elif linkageMethod == 'minNodeNormalized':
		dic_edge2sumWeight = {}
		sorted_edge=[]
		for c1 in dic_node2node_weight.keys():
			for c2, weight in dic_node2node_weight[c1].items():
				if c1 < c2:
					dic_edge2sumWeight[(c1,c2)]=weight
					minNNode=min(dic_cluster2nNode.get(c1,1.0),dic_cluster2nNode.get(c2,1.0))
					weight /= minNNode
					sorted_edge.append([(c1,c2),weight])
		sorted_edge.sort(key=lambda x:x[1])

	lst_target=[]
	dic_i2list={}
	max_i=0
	dic_i2list[max_i]=sorted_edge
	heappush(lst_target,(-dic_i2list[max_i][-1][1],max_i))
	max_i+=1
	linkageMatrix=[]
	new_cluster=max(set_cluster)+1
	while True:
		#print len(lst_target), lst_target
		if len(lst_target) != 0:
			negweight, i = heappop(lst_target)
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
					heappush(lst_target,(-dic_i2list[i][-1][1],i))
				continue
			if len(dic_i2list[i]) == 0:
				del dic_i2list[i]
			else:
				heappush(lst_target,(-dic_i2list[i][-1][1],i))
		else:
			(c1, c2), weight = (sorted(set_cluster)[0], sorted(set_cluster)[1]), 0.0
		nNode1, nNode2 = dic_cluster2nNode.get(c1,1.0), dic_cluster2nNode.get(c2,1.0)
		nNode = nNode1+nNode2
		dic_cluster2nNode[new_cluster] = nNode
		e=K(c1,c2)
		if linkageMethod == 'single':
			if e in dic_edge2maxWeight:
				del dic_edge2maxWeight[e]
		elif linkageMethod == 'complete':
			if e in dic_edge2minWeight:
				del dic_edge2minWeight[e]
		elif linkageMethod == 'weighted':
			if e in dic_edge2weightedWeight:
				del dic_edge2weightedWeight[e]
		elif linkageMethod == 'average':
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		elif linkageMethod == 'deltaRev':
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		elif linkageMethod == 'deltaCost':
			dic_cluster2edgeSumWeight[new_cluster]=dic_cluster2edgeSumWeight.get(c1,0.0)+dic_cluster2edgeSumWeight.get(c2,0.0) - dic_edge2sumWeight.get(K(c1,c2),0.0)
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		elif linkageMethod == 'deltaModularity':
			dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		elif linkageMethod == 'minEdgeCountNormalized':
			dic_cluster2edgeSumCount[new_cluster] = dic_cluster2edgeSumCount.get(c1,0.0) + dic_cluster2edgeSumCount.get(c2,0.0) - dic_edge2countWeight.get(K(c1,c2),0.0)
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
				del dic_edge2countWeight[e]
		elif linkageMethod == 'maxEdgeCountNormalized':
			dic_cluster2edgeSumCount[new_cluster] = dic_cluster2edgeSumCount.get(c1,0.0) + dic_cluster2edgeSumCount.get(c2,0.0) - dic_edge2countWeight.get(K(c1,c2),0.0)
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
				del dic_edge2countWeight[e]
		elif linkageMethod == 'unionEdgeCountNormalized':
			dic_cluster2edgeSumCount[new_cluster] = dic_cluster2edgeSumCount.get(c1,0.0) + dic_cluster2edgeSumCount.get(c2,0.0) - dic_edge2countWeight.get(K(c1,c2),0.0)
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
				del dic_edge2countWeight[e]
		elif linkageMethod == 'minEdgeWeightNormalized':
			dic_cluster2edgeSumWeight[new_cluster] = dic_cluster2edgeSumWeight.get(c1,0.0) + dic_cluster2edgeSumWeight.get(c2,0.0) - dic_edge2sumWeight.get(K(c1,c2),0.0)
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		elif linkageMethod == 'maxEdgeWeightNormalized':
			dic_cluster2edgeSumWeight[new_cluster] = dic_cluster2edgeSumWeight.get(c1,0.0) + dic_cluster2edgeSumWeight.get(c2,0.0) - dic_edge2sumWeight.get(K(c1,c2),0.0)
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		elif linkageMethod == 'unionEdgeWeightNormalized':
			dic_cluster2edgeSumWeight[new_cluster] = dic_cluster2edgeSumWeight.get(c1,0.0) + dic_cluster2edgeSumWeight.get(c2,0.0) - dic_edge2sumWeight.get(K(c1,c2),0.0)
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		elif linkageMethod == 'minModularityWeightNormalized':
			dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		elif linkageMethod == 'maxModularityWeightNormalized':
			dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		elif linkageMethod == 'avgModularityWeightNormalized':
			dic_cluster2modularitySumWeight[new_cluster]=dic_cluster2modularitySumWeight.get(c1,0.0)+dic_cluster2modularitySumWeight.get(c2,0.0)
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		elif linkageMethod == 'minPossibleEdgeCountNormalized':
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		elif linkageMethod == 'minNodeNormalized':
			if e in dic_edge2sumWeight:
				del dic_edge2sumWeight[e]
		linkageMatrix.append([c1,c2,1.0-weight,nNode])
		set_cluster.remove(c1)
		set_cluster.remove(c2)
		if len(set_cluster) <= 0:
			break
		lst_clusterPair2weight2=[]
		for cluster in set_cluster:
			e1, e2 =  K(c1,cluster), K(c2,cluster)
			if linkageMethod == 'single':
				if e1 not in dic_edge2maxWeight and e2 not in dic_edge2maxWeight:
					continue
				new_e = K(new_cluster,cluster)
				weight = max(dic_edge2maxWeight.get(e1,0.0),dic_edge2maxWeight.get(e2,0.0))
				if e1 in dic_edge2maxWeight:
					del dic_edge2maxWeight[e1]
				if e2 in dic_edge2maxWeight:
					del dic_edge2maxWeight[e2]
				dic_edge2maxWeight[new_e]=weight
			elif linkageMethod == 'complete':
				if e1 not in dic_edge2minWeight and e2 not in dic_edge2minWeight:
					continue
				new_e = K(new_cluster,cluster)
				weight = min(dic_edge2minWeight.get(e1,np.inf),dic_edge2minWeight.get(e2,np.inf))
				if e1 in dic_edge2minWeight:
					del dic_edge2minWeight[e1]
				if e2 in dic_edge2minWeight:
					del dic_edge2minWeight[e2]
				dic_edge2minWeight[new_e]=weight
			elif linkageMethod == 'weighted':
				if e1 not in dic_edge2weightedWeight and e2 not in dic_edge2weightedWeight:
					continue
				new_e = K(new_cluster,cluster)
				weight = 0.5*(dic_edge2weightedWeight.get(e1,0.0)+dic_edge2weightedWeight.get(e2,0.0))
				if e1 in dic_edge2weightedWeight:
					del dic_edge2weightedWeight[e1]
				if e2 in dic_edge2weightedWeight:
					del dic_edge2weightedWeight[e2]
				dic_edge2weightedWeight[new_e]=weight
			elif linkageMethod == 'average':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight
				weight = sumWeight
				weight /= float(dic_cluster2nNode.get(new_cluster,1.0)*dic_cluster2nNode.get(cluster,1.0))
			elif linkageMethod == 'deltaCost':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				weight = 1.0 / (dic_cluster2edgeSumWeight.get(new_cluster,0.0) * dic_cluster2nNode.get(cluster,1.0) + dic_cluster2edgeSumWeight.get(cluster,0.0) * dic_cluster2nNode.get(new_cluster,1.0) - sumWeight * float(dic_cluster2nNode.get(new_cluster,1.0)*dic_cluster2nNode.get(cluster,1.0))) * dic_cluster2nNode.get(new_cluster,1.0) * dic_cluster2nNode.get(cluster,1.0)
			elif linkageMethod == 'deltaRev':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				weight = sumWeight
				weight /= float(dic_cluster2nNode.get(new_cluster,1.0)*dic_cluster2nNode.get(cluster,1.0))
				weight *= nTotalNode-dic_cluster2nNode.get(new_cluster,1.0)-dic_cluster2nNode.get(cluster,1.0)
			elif linkageMethod == 'deltaModularity':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

			
				weight = sumWeight
				weight -= dic_cluster2modularitySumWeight.get(new_cluster,0.0)*dic_cluster2modularitySumWeight.get(cluster,0.0)/totalEdgeSumWeight
			elif linkageMethod == 'edgeCountNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
				if e1 in dic_edge2countWeight:
					del dic_edge2countWeight[e1]
				if e2 in dic_edge2countWeight:
					del dic_edge2countWeight[e2]
				dic_edge2countWeight[new_e]=countWeight
				weight = sumWeight/countWeight
			elif linkageMethod == 'minEdgeCountNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
				if e1 in dic_edge2countWeight:
					del dic_edge2countWeight[e1]
				if e2 in dic_edge2countWeight:
					del dic_edge2countWeight[e2]
				dic_edge2countWeight[new_e]=countWeight
				weight = sumWeight
				weight /= min(dic_cluster2edgeSumCount.get(new_cluster,0.0),dic_cluster2edgeSumCount.get(cluster,0.0))
			elif linkageMethod == 'maxEdgeCountNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
				if e1 in dic_edge2countWeight:
					del dic_edge2countWeight[e1]
				if e2 in dic_edge2countWeight:
					del dic_edge2countWeight[e2]
				dic_edge2countWeight[new_e]=countWeight
				weight = sumWeight
				weight /= max(dic_cluster2edgeSumCount.get(new_cluster,0.0),dic_cluster2edgeSumCount.get(cluster,0.0))
			elif linkageMethod == 'unionEdgeCountNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				countWeight = dic_edge2countWeight.get(e1,0.0)+dic_edge2countWeight.get(e2,0.0)
				if e1 in dic_edge2countWeight:
					del dic_edge2countWeight[e1]
				if e2 in dic_edge2countWeight:
					del dic_edge2countWeight[e2]
				dic_edge2countWeight[new_e]=countWeight
				weight = sumWeight
				weight /= dic_cluster2edgeSumCount.get(new_cluster,0.0)+dic_cluster2edgeSumCount.get(cluster,0.0)-countWeight
			elif linkageMethod == 'minEdgeWeightNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				weight = sumWeight
				weight /= min(dic_cluster2edgeSumWeight.get(new_cluster,0.0),dic_cluster2edgeSumWeight.get(cluster,0.0))
			elif linkageMethod == 'maxEdgeWeightNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				weight = sumWeight
				weight /= max(dic_cluster2edgeSumWeight.get(new_cluster,0.0),dic_cluster2edgeSumWeight.get(cluster,0.0))
			elif linkageMethod == 'unionEdgeWeightNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				weight = sumWeight
				weight /= dic_cluster2edgeSumWeight.get(new_cluster,0.0)+dic_cluster2edgeSumWeight.get(cluster,0.0)-sumWeight
			elif linkageMethod == 'minModularityWeightNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				weight = sumWeight
				weight /= min(dic_cluster2modularitySumWeight.get(new_cluster,0.0),dic_cluster2modularitySumWeight.get(cluster,0.0))
			elif linkageMethod == 'maxModularityWeightNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				weight = sumWeight
				weight /= max(dic_cluster2modularitySumWeight.get(new_cluster,0.0),dic_cluster2modularitySumWeight.get(cluster,0.0))
			elif linkageMethod == 'avgModularityWeightNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight

				weight = sumWeight
				weight /= 0.5*(dic_cluster2modularitySumWeight.get(new_cluster,0.0)+dic_cluster2modularitySumWeight.get(cluster,0.0))
			elif linkageMethod == 'minPossibleEdgeCountNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight
				minNNode = min(dic_cluster2nNode.get(new_cluster,1.0),dic_cluster2nNode.get(cluster,1.0))
				weight = sumWeight
				weight /= minNNode*(nTotalNode-minNNode)

			elif linkageMethod == 'minNodeNormalized':
				if e1 not in dic_edge2sumWeight and e2 not in dic_edge2sumWeight:
					continue
				new_e = K(new_cluster,cluster)
				sumWeight = dic_edge2sumWeight.get(e1,0.0) + dic_edge2sumWeight.get(e2,0.0)
				if e1 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e1]
				if e2 in dic_edge2sumWeight:
					del dic_edge2sumWeight[e2]
				dic_edge2sumWeight[new_e]=sumWeight
				minNNode = min(dic_cluster2nNode.get(new_cluster,1.0),dic_cluster2nNode.get(cluster,1.0))
				weight = sumWeight
				weight /= minNNode

			lst_clusterPair2weight2.append([K(new_cluster, cluster), weight])
		if len(lst_clusterPair2weight2) > 0:
			lst_clusterPair2weight2.sort(key=lambda x: x[1])
			dic_i2list[max_i]=lst_clusterPair2weight2
			heappush(lst_target,(-dic_i2list[max_i][-1][1],max_i))
			max_i+=1
		set_cluster.add(new_cluster)
		new_cluster+=1

	return linkageMatrix
		
if __name__ == "__main__":
	import argparse
	import scipy.cluster.hierarchy
	from WKNNgraph import *
	parser=argparse.ArgumentParser(
		usage='''\
	%(prog)s [options] graphfile k
	example: %(prog)s graphfile k -o out.txt
	''')
	
	parser.add_argument('graphfile', help='distance or weight graph file')
	parser.add_argument('-k', required=False, type=int, default=None, help='k flat clusters')
	parser.add_argument('-weightType', required=False, default='dissimilarity', choices=['graph', 'similarity', 'dissimilarity'], help='weightType')
	parser.add_argument('-K', required=False, type=int, default=10, help='K-nearest neighbor graph')
	parser.add_argument('-o', dest='outfile', required=False, metavar='str', default=None, help='outfile')
	args=parser.parse_args()
	
	wKNNgraph=WKNNgraph()
	if weightType == 'graph':
		wKNNgraph.loadGraph(args.graphfile)
	else:
		wKNNgraph.constructGraph(args.graphfile,weightType,K)

	linkageMatrix = fit(wKNNgraph.dic_node2node_weight)
	if args.outfile != None:
		OF.open(args.outfile,'w')
		if args.k == None:
			for x in linkageMatrix:
				OF.write('\t'.join(map(str,x))+'\n')
		else:
			lst_cluster = scipy.cluster.hierarchy.fcluster(linkageMatrix, t=k, criterion='maxclust')
			for node, cluster in enumerate(lst_cluster):
				name = wKNNgraph.dic_node2name[node]
				OF.write('\t'.join(map(str,[name,cluster]))+'\n')
		OF.close()
