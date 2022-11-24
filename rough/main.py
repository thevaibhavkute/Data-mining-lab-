# Implementon

# The k-nearest neighbor algorithm is imported from the scikit-learn package.
# Create feature and target variables.
# Split data into training and test data.
# Generate a k-NN model using neighbors value.
# Train or fit the data into the model.
# Predict the future.


# Load the training data.
# Find the optimal value for K:
# Predict a class value for new data:
# Calculate distance(X, Xi) from i=1,2,3,….,n.
# where X= new data point, Xi= training data, distance as per your chosen distance metric.
# Sort these distances in increasing order with corresponding train data.
# From this sorted list, select the top ‘K’ rows.
# Find the most frequent class from these chosen ‘K’ rows. This will be your predicted class.

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import streamlit as st

#heading
st.header('KNN Implementation')

# Loading data
irisData = load_iris()

# Create feature and target arrays
X = irisData.data
y = irisData.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

# Predict on dataset which model has not seen before
print(knn.predict(X_test))
st.write()

######################################################################################################
# Step 1: Calculate Euclidean Distance.
# Step 2: Get Nearest Neighbors.
# Step 3: Make Predictions.


# calculate the Euclidean distance between two vectors

# def euclidean_distance(row1, row2):
# 	distance = 0.0
# 	for i in range(len(row1)-1):
# 		distance += (row1[i] - row2[i])**2
# 	return sqrt(distance)

# get who are neighbors

# Locate the most similar neighbors
# def get_neighbors(train, test_row, num_neighbors):
# 	distances = list()
# 	for train_row in train:
# 		dist = euclidean_distance(test_row, train_row)
# 		distances.append((train_row, dist))
# 	distances.sort(key=lambda tup: tup[1])
# 	neighbors = list()
# 	for i in range(num_neighbors):
# 		neighbors.append(distances[i][0])
# 	return neighbors

# Make a classification prediction with neighbors

# def predict_classification(train, test_row, num_neighbors):
# 	neighbors = get_neighbors(train, test_row, num_neighbors)
# 	output_values = [row[-1] for row in neighbors]
# 	prediction = max(set(output_values), key=output_values.count)
# 	return prediction