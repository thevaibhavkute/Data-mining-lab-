from scipy.sparse import csr_matrix, find
import numpy as np
from math import inf, sqrt
from itertools import chain, combinations
from collections import defaultdict
import tkinter as tk
from tkinter import *
import pandas as pd
from tkinter import filedialog
from tkintertable.Tables import TableCanvas
import matplotlib as plt
import matplotlib
from scipy.sparse.linalg import svds
plt.rcParams.update({'font.size': 13})  # set the font-size in the figure



matplotlib.use("TkAgg")


Target = "Input"  # input values to train or test
Y = "Expected Output"  # present in csv
predicted = "by classfier"  # model will predict
splitting_method = "gini"
(X_train, X_test, y_train, y_test) = [], [], [], []
df = ""
min_confidence,min_support,max_length=0.5,0.5,4

K_clustures = 3
col1 = ''
col2 = ''


def powerset(s):
	return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


def getAboveMinSup(itemSet, itemSetList, minSup, globalItemSetWithSup):
	freqItemSet = set()
	localItemSetWithSup = defaultdict(int)

	for item in itemSet:
		for z in itemSetList:
			if item.issubset(z):
				globalItemSetWithSup[item] += 1
				localItemSetWithSup[item] += 1

	for item, supCount in localItemSetWithSup.items():
		support = float(supCount / len(itemSetList))
		if(support >= minSup):
			freqItemSet.add(item)

	return freqItemSet


def getUnion(itemSet, length):
	return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def pruning(candidateSet, prevFreqSet, length):
	tempCandidateSet = candidateSet.copy()
	for item in candidateSet:
		subsets = combinations(item, length)
		for subset in subsets:
            # if the subset is not in previous K-frequent get, then remove the set
			if(frozenset(subset) not in prevFreqSet):
				tempCandidateSet.remove(item)
				break
	return tempCandidateSet


def associationRule(freqItemSet, itemSetWithSup, minConf):
	rules = []
	for k, itemSet in freqItemSet.items():
		for item in itemSet:
			subsets = powerset(item)
			for s in subsets:
				if itemSetWithSup[frozenset(s)] !=0:
					confidence = float(itemSetWithSup[item] / itemSetWithSup[frozenset(s)])

					if(confidence > minConf):
						aub=itemSetWithSup[item]
						a = itemSetWithSup[frozenset(s)]
						b = itemSetWithSup[ item.difference(s)]
						if b==0:
							lift=inf
							max_confidence=inf
							Kulczynski=inf
							cosine=inf
						else:
							lift=confidence/b
							max_confidence = max(confidence, aub/b)
							Kulczynski = (confidence+aub/b)/2
							cosine = aub/sqrt(a*b)
						all_confidence = aub / max(a, b)
						rules.append([set(s), set(item.difference(s)), confidence, lift, all_confidence,
									max_confidence, Kulczynski,cosine])
	return rules


def getItemSetFromList(itemSetList):
	tempItemSet = set()

	for itemSet in itemSetList:
		for item in itemSet:
			tempItemSet.add(frozenset([item]))

	return tempItemSet


def apriori():
	# set value for itemList
	itemSetList=df.values.tolist()
	# itemSetList = [['eggs', 'bacon', 'soup'],
                                #  ['eggs', 'bacon', 'apple'],
                                #  ['soup', 'bacon', 'banana']]
    # print(itemSetList)
	minSup,minConf,max_k = min_support,min_confidence,max_length
	# unique items
	C1ItemSet = getItemSetFromList(itemSetList)
	
	# Final result global frequent itemset
	globalFreqItemSet = dict()
	# Storing global itemset with support count
	globalItemSetWithSup = defaultdict(int)

	L1ItemSet = getAboveMinSup(C1ItemSet, itemSetList, minSup, globalItemSetWithSup)
	currentLSet = L1ItemSet
	print(L1ItemSet)
	k = 2

    # Calculating frequent item set
	while(currentLSet and k <= max_k):
		print("entered")
        # Storing frequent itemset
		globalFreqItemSet[k-1] = currentLSet
        # Self-joining Lk
        
		candidateSet = getUnion(currentLSet, k)
        # Perform subset testing and remove pruned supersets
		candidateSet = pruning(candidateSet, currentLSet, k-1)
        # Scanning itemSet for counting support
		currentLSet = getAboveMinSup(candidateSet, itemSetList, minSup, globalItemSetWithSup)
		k += 1

	rules = associationRule(globalFreqItemSet, globalItemSetWithSup, minConf)
	rules.sort(key=lambda x: x[2])
	print(globalFreqItemSet)
	new_df = pd.DataFrame(rules, columns=["A",'B','confidence','lift','all confidence','max confidence','Kulczynski','cosine'])
	print(new_df)
	refresh(new_df)


class SparseHITS:
	def load_graph_dataset(self, is_undirected=False):
		'''
        Load the graph dataset from the given directory (data_home)

        inputs:
            data_home: string
                directory path conatining a dataset (edges.tsv, node_labels.tsv)
            is_undirected: bool
                if the graph is undirected
        '''
		edges = df
		edges[2]=1
		n = df.max().max() + 1
		# print(edges)
		# print(type(edges))
		# print(n)
		# Step 3. convert the edge list to the weighted adjacency matrix
		row =  edges.iloc[:, 0]
		col =  edges.iloc[:, 1]
		weights = edges.iloc[:, 2]
		self.A = csr_matrix((weights, (row, col)), shape=(n, n), dtype=float)
		print(self.A)
		if is_undirected == True:
			self.A = self.A + self.A.T
		self.AT = self.A.T

		# Step 4. set n (# of nodes) and m (# of edges)
		self.n = self.A.shape[0]  # number of nodes
		self.m = self.A.nnz  # number of edges

	def load_node_labels(self):
		'''
        Load the node labels from the given directory (data_home)

        inputs:
            data_home: string
                directory path conatining a dataset
        '''
		self.node_labels = np.array(['U','V'])

	def iterate_HITS(self, epsilon=1e-9, maxIters=10):
			'''
	        Iterate the HITS equation to obatin the hub & authority score vectors

	        inputs:
	            epsilon: float
	                the error tolerance of the iteration
	            maxIters: int
	                the maximum number of iterations

	        outputs:
	            h: np.ndarray (n x 1 vector)
	                the final hub score vector
	            a: np.ndarray (n x 1 vector)
	                the final authority score vector
	            h_residuals: list
	                the list of hub residuals over the iteration
	            a_residuals: list
	                the list of authority residuals over the iteration

	        '''
			old_h = np.ones(self.n) 
			old_a = np.ones(self.n) 
			h_residuals = []
			a_residuals = []
			print('hub then auth')
			print(old_h)
			print(old_a)
			for t in range(maxIters):
				# h = self.A.dot(old_a)
				# a = self.AT.dot(h)

				a = self.AT.dot(old_h)
				h = self.A.dot(a)
				

				# h = h / np.linalg.norm(h, 2)
				# a = a / np.linalg.norm(a, 2)

				h = h 
				a = a 
				print('hub then auth')
				print(h)
				print(a)
				h_residual = np.linalg.norm(h - old_h, 1)
				a_residual = np.linalg.norm(a - old_a, 1)
				h_residuals.append(h_residual)
				a_residuals.append(a_residual)
				old_h = h
				old_a = a

				if h_residual < epsilon and a_residual < epsilon:
					break

			return h, a, h_residuals, a_residuals

	def rank_nodes(ranking_scores, topk=-1):
		sorted_nodes = np.flipud(np.argsort(ranking_scores))
		sorted_scores = ranking_scores[sorted_nodes]
		ranking_results = pd.DataFrame()
		ranking_results["node_id"] = sorted_nodes
		ranking_results["score"] = sorted_scores

		return ranking_results[0:topk]

	def plot_residuals(residuals, title):
		plt.semilogy(residuals, marker='o', markersize=5)
		plt.title(title)
		plt.ylim(bottom=1e-10, top=1e1)
		plt.ylabel('Residual')
		plt.xlabel('# of iterations')
		plt.grid(True)
		plt.show()

	def compute_exact_HITS(self):
			'''
	        Compute the exact hub & authority score vectors from the closed form

	        outputs:
	            h: np.ndarray (n x 1 vector)
	                the final hub score vector
	            a: np.ndarray (n x 1 vector)
	                the final authority score vector
	        '''
			h, s, a = svds(self.A, k=1)

			h = np.asarray(h).flatten()
			a = np.asarray(a).flatten()

			# since SVD is not unique, h and a could be negative according to a random seed
			# in this case, we need to make the scores non-negative and L2-normalize them out
			h = h * np.sign(h)
			h = h / np.linalg.norm(h, 2)
			a = a * np.sign(a)
			a = a / np.linalg.norm(a, 2)

			return h, a

	def rank_nodes(self, ranking_scores, topk=-1):
		'''
        Rank nodes in the order of given ranking scores.
        This function reports top-k rankings.

        inputs:
            ranking_scores: np.ndarray
                ranking score vector
            topk: int
                top-k ranking parameter, default is -1 indicating report all ranks
        '''
		sorted_nodes = np.flipud(np.argsort(ranking_scores))
		sorted_scores = ranking_scores[sorted_nodes]
		ranking_results = pd.DataFrame()
		ranking_results["node_id"] = sorted_nodes
		ranking_results["score"] = sorted_scores
		refresh(ranking_results[0:topk])
		# print(ranking_results[0:topk])

def hits_fun():
	hits = SparseHITS()
	hits.load_graph_dataset(is_undirected=False)
	hits.load_node_labels()
	h, a, h_residuals, a_residuals = hits.iterate_HITS(epsilon=1e-3, maxIters=10)
	# print(h)
	# print(a)
	print("Top-10 rankings based on the hub score vector:")
	print(hits.rank_nodes(h, topk=10))

	print("Top-10 rankings based on the authority score vector:")
	print(hits.rank_nodes(a, topk=10))


# page rank here

dampingFactor = 0.12


class Node:
	def __init__(self, id):
		self.id = id
		self.children = []
		self.parents = []
		self.pagerank = 1.0


def cmp(nodeObj):
	return nodeObj.pagerank


def pagerankUtil():
	edge=df
	numberOfEdges = len(edge)
	maxIndex = 0
	for i in range(0, numberOfEdges):
		maxIndex = max(edge.iloc[i, 0], edge.iloc[i, 1])

	adjacencyMatrix = [[0] * (maxIndex + 1) for i in range(maxIndex + 1)]

	for i in range(0, numberOfEdges):
		adjacencyMatrix[edge.iloc[i, 0]][edge.iloc[i, 1]] = 1

	graph = []
	for i in range(0, maxIndex + 1):
		graph.append(Node(i))
		graph[i].pagerank = 1 / maxIndex

	for i in range(0, numberOfEdges):
		graph[edge.iloc[i, 0]].children.append(edge.iloc[i, 1])
		graph[edge.iloc[i, 1]].parents.append(edge.iloc[i, 0])

	# create weights for artificial links
	artificialLink = (1 - dampingFactor) / maxIndex

	# update pagerank of every node in graph
	for i in range(maxIndex):
		outdegree = max(1, len(graph[i].children))

		for j in graph[i].parents:
			try:
				graph[i].pagerank += (dampingFactor * graph[j].pagerank) / outdegree
			except:
				graph[i].pagerank = 0

		graph[i].pagerank += dampingFactor * artificialLink

	# sort pages in reverse order
	sortedPages = sorted(graph, key=cmp, reverse=True)

	topIndex = []
	topPagerank = []

	for i in range(0, 10):
		topIndex.append(sortedPages[i].id)
		topPagerank.append(sortedPages[i].pagerank)

	tmp_df = pd.DataFrame(columns=["Index Of Page", "Pagerank"], data=list(zip(topIndex, topPagerank)))
	refresh(tmp_df)


def browseFiles():
	global df,filePath
	filePath = filedialog.askopenfilename(
		initialdir="/", title="Select a File", filetypes=(("all files", "*.*"), ("all files", "*.*")))
	df = pd.read_csv(filePath)
	df.to_csv("util.csv")
	refresh(df)


def freshStart():
	global Target
	df = pd.DataFrame(index=["NOTE:", "", ""], data=[
	                  "Upload a Dataset and", " select the target ", "attribute"])
	df.to_csv("util.csv")

	refresh(df)


def setTarget(col):
	global X_train, X_test, y_train, y_test
	global Target
	Target = col
	# df = pd.read_csv("util.csv")


def setY(col):
	global Y
	Y = col


def setSplittingMethod(method):
	global splitting_method
	splitting_method = method


def check():
	df = pd.read_csv("util.csv")
	print(df[Target])
	print(df[Y])


def setAttribute1(name):
	global col1
	col1 = name


def submit(name_var, passw_var):
	global K_clustures
	global K_for_tk
	name = name_var.get()
	password = passw_var.get()

	print("The name is : " + name)
	print("The password is : " + password)

	name_var.set("")
	passw_var.set("")


def show_entry_fields():
	global K_clustures,e1,min_confidence,min_support,max_length
	min_support = float(min_support.get())
	min_confidence = float(min_confidence.get())
	max_length = int(max_length.get())
	print(min_confidence, min_support, max_length)
	print(type(min_support))
	print(type(min_confidence))
	print(type(max_length))


def set_parameters():
	global e1, e2,min_support,min_confidence,max_length
	master = tk.Tk()
	master.title('For values of paramters')
	tk.Label(master, text="Enter Value of min_support").grid(row=0)
	min_support = tk.Entry(master)
	# e2 = tk.Entry(master)
	min_support.grid(row=0, column=1)
	
	tk.Label(master, text="Enter Value of min_confidence").grid(row=1)
	min_confidence = tk.Entry(master)
	min_confidence.grid(row=1, column=1)

	tk.Label(master, text="Enter Value of max_length").grid(row=2)
	max_length = tk.Entry(master)
	max_length.grid(row=2, column=1)

	tk.Button(master,
           text='Save K', command=show_entry_fields).grid(row=3,
                                                          column=1,
                                                          sticky=tk.W,
                                                          pady=4)

	tk.mainloop()

    


def appWindow():
	global root
	global Target

	global K_for_tk
	root = Tk()
	root.title("Home page")
	root['background'] = '#9ea7e6'
	width = root.winfo_screenwidth()
	height = root.winfo_screenheight()
	root.geometry("%dx%d" % (width, height))
	root.state('zoomed')

	df = pd.read_csv("util.csv")

	# Creating Menubar
	name_label = Label(root, text='Username', font=('calibre', 10, 'bold'))

	# creating a entry for input
	# name using widget Entry
	name_entry = Entry(root, textvariable=K_clustures,
                    font=('calibre', 10, 'normal'))

	menubar = Menu(root)

	Dataset = Menu(menubar, tearoff=0)
	menubar.add_cascade(label='Dataset', menu=Dataset)
	Dataset.add_command(label='Upload', command=browseFiles)
	Dataset.add_command(label='Delete', command=freshStart)
	Dataset.add_separator()
	Dataset.add_command(label='Exit', command=root.destroy)

	clusters = Menu(menubar, tearoff=0)
	menubar.add_cascade(label='Options', menu=clusters)
	# clusters.add_command(label='set parameters', command=set_parameters)
	# clusters.add_command(label='apriori', command=apriori)
	clusters.add_command(label='Page Rank', command=pagerankUtil)
	clusters.add_command(label='HITS', command=hits_fun)

	# display Menu
	root.config(menu=menubar)
	tframe = Frame(root, bg='#9ea7e6')
	tframe.pack(expand=True, fill=BOTH)
	table = TableCanvas(tframe)
	table.importCSV("util.csv", sep=',')
	table.show()

	root.mainloop()


def refresh(new_df):
	global Target
	global root

	try:
		root.destroy()
	except:
		pass

	new_df.to_csv("util.csv")
	appWindow()


if __name__ == "__main__":
	freshStart()
