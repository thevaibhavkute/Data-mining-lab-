import numpy as np
import sys


def euclideanDistance(x, y):
	squared_d = 0
	for i in range(len(x)):
		squared_d += (x - y)**2
	d = np.sqrt(squared_d)
	return d


class PartitioningClustering:
	def __init__(self, **kwargs):
		self.params = kwargs
		self.medoids_cost = []

	def initMedoids(self, data):
		self.medoids = []
		# select k indices from all data in dataset
		indexes = np.random.randint(0, len(data)-1, self.params["k"])
		# starting medoids(clusters) will be random numbers from dataset
		self.medoids = data[indexes]
		print(f"\nselected medoids >>> {self.medoids}\n")
		for i in range(0, self.params["k"]):
			self.medoids_cost.append(0)  # for each cluster medoid the cost is 0
		print(f"\nmedoids cost >>> {self.medoids_cost}\n")

	# check that new medoids is equals to the old one - if that case we have to stop the algorithm
	def isConverged(self,  new_medoids):
		return set([tuple(x) for x in self.medoids]) == set([tuple(x) for x in new_medoids])

	def updateMedoids(self, data, labels):
		self.params["has_converged"] = True
		clusters = []
		# storing data points to the current cluster key belong to
		for i in range(self.params["k"]):
			cluster = []
			for j in range(len(data)):
				if labels[j] == i:  # if the label of data is i-th k then the cluster of that data is found
					cluster.append(data[j])  # data with common labels
			clusters.append(cluster)
			print("\n============================================================")
			print("labels >>> ", labels)
			print("clusters >>> ", clusters)

		new_medoids = []
		for i in range(self.params["k"]):
			new_medoid = self.medoids[i]
			old_medoid_cost = self.medoids_cost[i]
			for j in range(len(clusters[i])):
				cur_medoids_cost = 0
				for dpoint_index in range(len(clusters[i])):
					# calculate distance between each point in a cluster to find the new medoid
					cur_medoids_cost += euclideanDistance(
						clusters[i][j], clusters[i][dpoint_index])
				# if calculated distance is less than the old one we found a new medoid with a minimum cost
				if cur_medoids_cost < old_medoid_cost:
					new_medoid = clusters[i][j]  # update medoid
					old_medoid_cost = cur_medoids_cost
			new_medoids.append(new_medoid)
			print("\nnew_medoids >>> ", new_medoids)
		if not self.isConverged(new_medoids):
			self.medoids = new_medoids
			self.params["has_converged"] = False

	def fit(self, data):

		if self.params["method"] == "kmeans":
			self.centroids = {}  # dict centroids of each cluster
			for i in range(self.params["k"]):
				# set centroids as the number of k - select the k first points of data
				self.centroids[i] = data[i]

			for i in range(self.params["max_iter"]):
				self.classifications = {}  # the result of classification for each cluster
				for i in range(self.params["k"]):
					# number of classification is based on the number of k which is the number of centroids
					self.classifications[i] = []

				for featureset in data:
					# calculate the distance from each centroid for this featureset
				distances = [np.linalg.norm(featureset - self.centroids[centroid])
                                    for centroid in self.centroids]
				# the index of the minimum distance is the one cluster that this feature will be in it
				classification = distances.index(min(distances))
				# put the featureset in its cluster - its cluster can be found based on the index of the minimum distance from each centroid because there are k centroids then k distances
				self.classifications[classification].append(featureset)

				print("--------------classifications--------------")
				print(self.classifications, "\n")

				prev_centroids = dict(self.centroids)  # save perv centroids
				for classification in self.classifications:
					# update each classified centroid based on the average of all data points in that cluster
					self.centroids[classification] = np.average(
						self.classifications[classification], axis=0)
					print("--------------centroids--------------")
					print(self.centroids, "\n")

		elif self.params["method"] == "kmedoids":
			self.initMedoids(data)
			for i in range(self.params["max_iter"]):
				cur_labels = []
				for medoid in range(self.params["k"]):
					self.medoids_cost[medoid] = 0
					for k in range(len(data)):
						d_list = []
						for j in range(self.params["k"]):
							d_list.append(euclideanDistance(self.medoids[j], data[k]))
						# the index of the minimum distance is the current label for that data - cause we're calculating k (clusters) distance between each data
						cur_labels.append(d_list.index(min(d_list)))
						# the cost of each medoid is the sum of minimum distances in k medoids for each data
						self.medoids_cost[medoid] += min(d_list)
						print(f"\nmedoids cost {medoid} = {self.medoids_cost[medoid]}")
				print(f"\ntotal medoids cost {self.medoids_cost}")
				# update medoids at the end of each iteration
				self.updateMedoids(data, cur_labels)
				if self.params["has_converged"]:
					break
			print(f"\nfinal medoids >>> {self.medoids}\n")
			return np.array(self.medoids)

	def predict(self, data):  # cluster new data
		if self.params["method"] == "kmeans":
		distances = [np.linalg.norm(data-self.centroids[centroid])
                    for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification
		elif self.params["method"] == "kmedoids":
			pred = []
			for i in range(len(data)):
				d_list = []
				for j in range(len(self.medoids)):
					# calculate each distance between each data and found medoids - the index of minimum distance is the cluster that the data should be in
					d_list.append(euclideanDistance(self.medoids[j], data[i]))
				pred.append(d_list.index(min(d_list)))
			return np.array(pred)
