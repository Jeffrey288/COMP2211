
## Naive Bayes Classifier

```python
import numpy as np
# CategoricalNB is for doing Naive Bayes classifier for categorical features
from sklearn.naive_bayes import CategoricalNB

# Forming training data
# Features: Blood Pressure (0=High,1=Normal,2=Low), Fever (0=High,1=Mild,2=No Fever), 
#           Diabetes (0=Yes, 1=No), Vomit (0=Yes, 1=No)
training = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [2, 0, 0, 1], [1, 1, 0, 1], [1, 2, 1, 1], 
                     [1, 2, 1, 0], [2, 2, 1, 0], [0, 1, 0, 1], [0, 2, 1, 1], [1, 1, 1, 1],
                     [0, 1, 1, 0], [2, 1, 0, 0], [2, 0, 1, 1], [1, 1, 0, 0]])
# Forming the label set
# Labels: 0=Yes, 1=No
outcome = np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1])

# This is the new data to be evaluated using the trained AI (model)
new_sample = np.array([[0, 2, 0, 0]])

# Train Naive Bayes classifier according to training, outcome
clf = CategoricalNB(alpha=1.0e-10).fit(training, outcome)
# Perform classification on new_sample
pred_class = clf.predict(new_sample)
# Get the probability estimates for new_sample
prob = clf.predict_proba(new_sample)
# Print the results
print(pred_class, prob)
```

Not covered in code:
- independent assumption of evidence (e.g. having fever won't affect the probability of having high blood pressure)
- **important:** continuous data (assume normally distributed, use probability density as probability of evidence)
- zero frequency

## K-Nearest Neighbour (KNN)

```python
from sklearn.preprocessing import LabelEncoder     # Import LabelEncoder function
from sklearn.neighbors import KNeighborsClassifier # Import KNeighborsClassifier function
import numpy as np                                 # Import NumPy

# Assign features and label variables
height = np.array([158, 158, 158, 160, 160, 163, 163, 160, 163, 165, 165, 165, 168, 168, 168, 170, 170, 170])
h_mean = np.mean(height)
h_std = np.std(height,ddof=1) # ddof=1 is to make the divisor to n-1, i.e., sample mean
height = (height - h_mean)/h_std

weight = np.array([58, 59, 63, 59, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63, 64, 68])
w_mean = np.mean(weight)
w_std = np.std(weight,ddof=1) # ddof=1 is to make the divisor to n-1, i.e., sample mean
weight = (weight - w_mean)/w_std

size = ['M','M','M','M','M','M','M','L','L','L','L','L','L','L','L','L','L','L']

# Create LabelEncoder that used to convert string labels to numbers
encoder = LabelEncoder()
# Convert string labels into numbers
label = encoder.fit_transform(size)
# Combine height and weight into single list of tuples
features = list(zip(height,weight))

# Create a KNN classifier and set K=5
model = KNeighborsClassifier(n_neighbors=5)
# Train the model using the training sets
model.fit(features, label)

# Predict output
predicted = model.predict([[(161-h_mean)/h_std,(61-w_mean)/w_std]]) # height = 161, weight = 61

# Convert number to string label
predicted = encoder.inverse_transform(predicted)
print(predicted)
```

Note: standardization is done on height and weight, and Euclidean distance is used

Not covered in code:
- choose $k=\sqrt{n}$
- distance functions
  - manhattan distance: $\sum|x_i^\text{Train}-x_i^{Test}|$, preferred for many dimensions
  - cosine distance, $1-\cos{\theta}$
  - hamming distance, no. of differences between categorical data
- cross-validation
- speed up KNN:
  - Reduce the dimension of training data (e.g., using Principle-Component Analysis).
  - Use good data structures to store the data, e.g., KD-tree.
  - Parallelizing the distance computations.

## K-Means Clustering

### Procedure
Given K, the K-Means algorithm works as follows:
1. Choose K (random) data points (seeds) to be the initial centroids (cluster centers)
   - Elbow / Silhouette methods
2. Find the distances between each data point in our training set with the K centroids
   - Euclidean distance normally
3. **Assignment**: Assign each data point to the closest centroid according to the distance found
4. **Refitting**: Re-compute the centroids using the current cluster memberships
5. If a convergence criterion is NOT met, repeat steps 2 to 4
    - minimum reassignments / change of centroids / decrease in SSE (summed error distance)

#### Proof of convergence based on SSE convergence criterion
- The change in loss function after new assignment ($z_i^*\in\argmin_{j\in{\{1,\dots,k\}}}||x_i-\mu_j||^2_2$) is given by
$$L(\mu,z^*)-L(\mu,z)=\sum^n_{i=1}\left(||x_i-\mu_{z_i^*}||^2_2-||x_i-\mu_{z_i}||^2_2\right)\leq0$$
where $z_i$ is the previous assignment.
- The change in loss function after refitting ($\mu_j^*=\dfrac{1}{|\{i:z_i=j\}|}\sum_{i:z_i=j}x_i$) is
$$L(\mu^*,z)-L(\mu,z)=\sum^k_{j=1}\left(\left(\sum_{i:z_i=j}||x_i-\mu_j^*||^2_2\right)-\left(\sum_{i:z_i=j}||x_i-\mu_j||^2_2\right)\right)\leq0$$

#### Disadvantages
- sensitive to outliers (can remove some data points, and perform random sampling)
- sensitive to initial seeds
- unapplicable to hyper-ellipsoids / hyper-spheres

### Code
```python
# Import the required libraries
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Unlabled training data
data = np.array([[19, 15, 39], [67, 19, 14], [35, 24, 35], [60, 30, 4], [65, 38, 35],
                 [49, 42, 52], [70, 46, 56], [70, 49, 55], [57, 54, 51], [68, 59, 55], 
                 [23, 62, 41], [65, 63, 52], [27, 67, 56], [47, 71, 9],  [57, 75, 5],
                 [43, 78, 17], [56, 79, 35], [40, 87, 13], [37, 97, 32], [34, 103, 23]])

# Initial centroids
init_centroids = np.array([[70, 46, 56], [27, 67, 56], [37, 97, 32]])

# Create a KMeans object by specifying
# - Number of clusters (n_clusters) = 3, initial centroids (init) = init_centroids
# - Number of time the k-means algorithm will be run with different centroid seeds (n_init) = 1
# - Maximum number of iterations of the k-means algorithm for a single run (max_iter) = 4
kmeans = KMeans(n_clusters=3, init=init_centroids, n_init=1, max_iter = 4)

kmeans.fit(data)                    # Compute k-means clustering
labels = kmeans.predict(data)       # Predict the closest cluster each sample in data belongs to
centroids = kmeans.cluster_centers_ # Get resulting centroids
fig = plt.figure(figsize = (10,10)) # Figure width = 10 inches, height = 10 inches
ax = fig.gca(projection='3d')       # Defining 3D axes so that we can plot 3D data into it

# Get boolean arrays representing entries with labels = 0, 1, and 2
a = np.array(labels == 0); b = np.array(labels == 1); c = np.array(labels == 2)

# Plot centroids with color = black, size = 50 units, transparency = 20%, and put label "Centroids"
ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c="black", s=50, alpha=0.8, label="Centroids")
# Plot data in the different clusters (1st in red, 2nd in green, 3rd blue)
ax.scatter(data[a,0], data[a,1], data[a,2], c="red", s=40, label="1st Cluster")
ax.scatter(data[b,0], data[b,1], data[b,2], c="green", s=40, label="2nd Cluster")
ax.scatter(data[c,0], data[c,1], data[c,2], c="blue", s=40, label="3rd Cluster")
ax.legend() # Show legend

ax.set_xlabel("Age")                   # Put x-axis label "Age"
ax.set_ylabel("Income (K)")            # Put y-axis label "Income (K)"
ax.set_zlabel("Expense Score (1-100)") # Put z-axis label "Expense Score (1-100)"
ax.set_title("Customer Segmentation - K-Means Clustering") # Put figure title
```

## Principal Component Analysis (PCA)

1. Construct a covariance matrix $C=\begin{bmatrix}\text{cov}(X_1,X_1)&\text{cov}(X_1,X_2)&\dots&\text{cov}(X_1,X_N)\\\ast&\text{cov}(X_2,X_2)&\dots&\text{cov}(X_2,X_N)\\\vdots&\vdots&\ddots&\vdots\\\ast&\ast&\dots&\text{cov}(X_N,X_N)\end{bmatrix}$
    - $\text{cov}(X, Y)=\dfrac{1}{n-1}\sum^n_{i=1}(X_i-\bar{x})(Y_i-\bar{y})$ describes the covariance between two variables.
    - Larger covariance means variables co-vary predictably.
2. Find its ($C$) eigenvalues $\lambda_1, \lambda_2, ..., \lambda_N$ and its corresponding eigenvectors $v_1, v_2, ..., v_N$.
3. Choose the eigenvectors with the largest eigenvalues, and these eigenvectors describe the **linear combination of the original dimensions**.

Code:
```python
import numpy as np # Import NumPy
x = np.array( [[90, 60, 90], # Original data
[90, 90, 30],
[60, 60, 60],
[60, 60, 90],
[30, 30, 30]] )
covariance_matrix = np.cov(x , rowvar = False, bias=False) # Find covariance matrix
print("Covariance Matrix:\n", covariance_matrix)
# Compute eigenvalues, eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
# Sort the eigenvalues and eigenvectors
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalue = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:,sorted_index]
print("Eigenvalues:\n",sorted_eigenvalue)
print("Eigenvectors:\n",sorted_eigenvectors)
# Form transformation matrix
W = np.array([sorted_eigenvectors[:,0], sorted_eigenvectors[:,1]]).T
print("W:\n", W)
# Compute the result, i.e., the data in reduced dimensions
y = x.dot(W)
print("y:\n", y)
```

## Artificial Neuron Network

Stopping Rules
- maximum epochs
- target accuracy
### Single Perception

- $\text{output}=f(w_1\times x_1+w_2\times x_2+\theta)$
- Back propagation:
    - $w_i=w_i+\eta(T-O)x_i$
    - $\theta=\theta+\eta(T-O)(1)$

This example trains the AND function. This problem is linearly separable.
```python
import numpy as np  # Import NumPy
from sklearn.linear_model import Perceptron # Import Perceptron class from Scikit-Learn

inputs = np.array([[0,0], [0,1], [1,0], [1,1]])  # Inputs
outputs = np.array([0, 0, 0, 1])                 # Expected outputs

# Create and fit a perceptron model
# Set learning rate (eta0)
model = Perceptron(eta0=0.2)
model.fit(inputs, outputs)

# Use the trained model to predict the outputs
predicted_outputs = model.predict([[0,0], [1,0], [1,1], [0,1]])
print(predicted_outputs) # Print the predicated outputs

print(model.coef_)       # Print the final weights
print(model.intercept_)  # Print the bias
```