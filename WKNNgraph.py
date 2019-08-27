#!/usr/bin/env python
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
		dic_name2node={}
		lst_point=[]
		for node, line in enumerate(IF):
			s=line.strip().split()
			name, point = s[0], list(map(float,s[1:]))
			self.dic_node2name[node]=name
			dic_name2node[name]=node
			lst_point.append(point)
		IF.close()
		#self.data = np.array(lst_point)
		X=np.array(lst_point)
		#import sklearn.neighbors
		#tree = sklearn.neighbors.KDTree(X, metric=metric)
		from sklearn.neighbors import NearestNeighbors
		tree = NearestNeighbors(n_neighbors=nNeighbor+1, algorithm='kd_tree', metric=metric)
		tree.fit(X)
		dist, ind = tree.kneighbors(X)

		def K(c1,c2):
			if c1 < c2:
				return (c1,c2)
			else:
				return (c2,c1)

		set_edge=set()
		self.dic_node2node_weight={}
		dic_node2node_i={}
		for n1 in self.dic_node2name.keys():
			if n1 not in self.dic_node2node_weight:
				self.dic_node2node_weight[n1]={}
				dic_node2node_i[n1]={}
			for i, (n2, weight) in enumerate(zip(ind[n1,1:],dist[n1,1:])):
				if n2 not in self.dic_node2node_weight:
					self.dic_node2node_weight[n2]={}
					dic_node2node_i[n2]={}
				self.dic_node2node_weight[n1][n2]=weight
				self.dic_node2node_weight[n2][n1]=weight
				dic_node2node_i[n1][n2]=i
				set_edge.add(K(n1,n2))
		nNode=len(self.dic_node2node_weight)
		nEdge=len(set_edge)
		for rank, (n1, n2) in enumerate(sorted(set_edge,key=lambda x: self.dic_node2node_weight[x[0]][x[1]])):
			i1 = dic_node2node_i[n1].get(n2, (nNeighbor+nNode)*0.5)
			i2 = dic_node2node_i[n2].get(n1, (nNeighbor+nNode)*0.5)
			#new_val = l*float(nEdge-rank)/nEdge + (1.0-l)*0.5*(np.exp(-i1)+np.exp(-i2))
			new_val = np.sqrt(l*(float(nEdge-rank)/nEdge)**2 + (1.0-l)*(np.exp(-i1-i2)))
			self.dic_node2node_weight[n1][n2]=new_val
			self.dic_node2node_weight[n2][n1]=new_val

		dic_node2average={}
		for n1 in self.dic_node2node_weight.keys():
			dic_node2average[n1]=np.mean(list(self.dic_node2node_weight[n1].values()))
				
		for n1 in self.dic_node2node_weight.keys():
			for n2, weight in self.dic_node2node_weight[n1].items():
				if n1 < n2:
					self.dic_node2node_weight[n1][n2]=weight*min(dic_node2average[n1],dic_node2average[n2])
					self.dic_node2node_weight[n2][n1]=weight*min(dic_node2average[n1],dic_node2average[n2])
					#self.dic_node2node_weight[n1][n2]=weight*(dic_node2average[n1]+dic_node2average[n2])*0.5
					#self.dic_node2node_weight[n2][n1]=weight*(dic_node2average[n1]+dic_node2average[n2])*0.5


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

		def K(c1,c2):
			if c1 < c2:
				return (c1,c2)
			else:
				return (c2,c1)

		set_edge=set()
		self.dic_node2node_weight={}
		dic_node2node_i={}
		for n1 in self.dic_node2name.keys():
			if n1 not in self.dic_node2node_weight:
				self.dic_node2node_weight[n1]={}
				dic_node2node_i[n1]={}
			for i, (name2, weight) in enumerate(dic_name2neighbor[self.dic_node2name[n1]]):
				n2=dic_name2node[name2]
				if n2 not in self.dic_node2node_weight:
					self.dic_node2node_weight[n2]={}
					dic_node2node_i[n2]={}
				self.dic_node2node_weight[n1][n2]=weight
				self.dic_node2node_weight[n2][n1]=weight
				dic_node2node_i[n1][n2]=float(i)
				#dic_node2node_i[n1][n2]=dic_node2node_i[n1].get(n2, 0.0) + 0.5*i
				#dic_node2node_i[n2][n1]=dic_node2node_i[n2].get(n1, 0.0) + 0.5*i
				set_edge.add(K(n1,n2))
		nNode=len(self.dic_node2node_weight)
		nEdge=len(set_edge)
		if weightType=='dissimilarity':
			for rank, (n1, n2) in enumerate(sorted(set_edge,key=lambda x: self.dic_node2node_weight[x[0]][x[1]])):
				i1 = dic_node2node_i[n1].get(n2, (nNeighbor+nNode)*0.5)
				i2 = dic_node2node_i[n2].get(n1, (nNeighbor+nNode)*0.5)
				#new_val = l*float(nEdge-rank)/nEdge + (1.0-l)*0.5*(np.exp(-i1)+np.exp(-i2))
				new_val = np.sqrt(l*(float(nEdge-rank)/nEdge)**2 + (1.0-l)*(np.exp(-i1-i2)))
				self.dic_node2node_weight[n1][n2]=new_val
				self.dic_node2node_weight[n2][n1]=new_val
		if weightType=='similarity':
			for rank, (n1, n2) in enumerate(sorted(set_edge,key=lambda x: self.dic_node2node_weight[x[0]][x[1]],reverse=True)):
				i1 = dic_node2node_i[n1].get(n2, (nNeighbor+nNode)*0.5)
				i2 = dic_node2node_i[n2].get(n1, (nNeighbor+nNode)*0.5)
				#new_val = l*float(nEdge-rank)/nEdge + (1.0-l)*0.5*(np.exp(-i1)+np.exp(-i2))
				new_val = np.sqrt(l*(float(nEdge-rank)/nEdge)**2 + (1.0-l)*(np.exp(-i1-i2)))
				self.dic_node2node_weight[n1][n2]=new_val
				self.dic_node2node_weight[n2][n1]=new_val

		dic_node2average={}
		for n1 in self.dic_node2node_weight.keys():
			dic_node2average[n1]=np.mean(list(self.dic_node2node_weight[n1].values()))
				
		for n1 in self.dic_node2node_weight.keys():
			for n2, weight in self.dic_node2node_weight[n1].items():
				if n1 < n2:
					self.dic_node2node_weight[n1][n2]=weight*min(dic_node2average[n1],dic_node2average[n2])
					self.dic_node2node_weight[n2][n1]=weight*min(dic_node2average[n1],dic_node2average[n2])
					#self.dic_node2node_weight[n1][n2]=weight*(dic_node2average[n1]+dic_node2average[n2])*0.5
					#self.dic_node2node_weight[n2][n1]=weight*(dic_node2average[n1]+dic_node2average[n2])*0.5



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
