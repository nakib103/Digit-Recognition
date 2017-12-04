print('----------------------RESULT----------------------------')

# import datasets from sklearn
from sklearn import datasets
# import numpy as np
import numpy as np

# load in the digits data
digits = datasets.load_digits()
print(type(digits))											# what type this dataset is?

## insight on the digit data
# get the keys of the digit data
print('keys of digit data:', digits.keys())
# print out the data
print('the data in digit data:', digits.data)
# print out the target
print('target in digit data:', digits.target)
# lastly see the description
print('Description of the digit data:', digits.DESCR)

## more insights on the digit data
# check out the data in digit date
digits_data = digits.data
print(type(digits_data))
print(digits_data.shape)
# check out the target in digit data
digits_target = digits.target
print(digits_target.shape)
print(len(np.unique(digits_target)))						# see the number of unique labels
# check out the image in digit data
digits_image = digits.images
print(digits_image.shape)

## visualize the digit dataset
# import matplotlib subpackage pyplot as plt
import matplotlib.pyplot as plt

# craeate a figure and describe its shape in inches
fig = plt.figure(figsize = (6, 6))

# adjust the subplots
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

# for each of the 64 images
for i in range(64):
	# initialize a subplot: 8*8 size at i+1 position
	ax = fig.add_subplot(8, 8, i + 1, xticks = [], yticks = [])
	# add a image on the subplot
	ax.imshow(digits_image[i], cmap = plt.cm.binary, interpolation = 'nearest')
	# label the image with target text
	ax.text(0, 7, str(digits_target[i]))

# show the plot
#plt.show()

# high dimensioanal dataset can be reduced by dimensionality reduciton technique 
# such as PCA to obtain principal features that create the most variance.

# import PCA model functions from sklearn's decomposition subpackage
from sklearn.decomposition import RandomizedPCA, PCA

# create randomize PCA model that take 2 components
randomized_pca = PCA(n_components = 2, svd_solver = 'randomized')
# fit and transform the data into the model
reduced_data_rpca = randomized_pca.fit_transform(digits_data)

# create a regular PCA
pca = PCA(n_components = 2)
# fit and transform the data into the model
reduced_data_pca = pca.fit_transform(digits_data)

# inspect the shape
print(reduced_data_rpca.shape)
print(reduced_data_pca.shape)
# print the output
print(reduced_data_rpca)
print(reduced_data_pca)

# notice that we did not provide the target array because we want to see
# if PCA can reveal different labels and we can distinct them

color = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
fig2 = plt.figure()

for i in range(len(color)):
	x = reduced_data_rpca[:, 0][digits_target == i]
	y = reduced_data_rpca[:, 1][digits_target == i]
	plt.scatter(x, y, c = color[i], edgecolor = 'black')
plt.legend(digits.target_names, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.0)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('PCA scatter plot')
#plt.show()

## clustering the data
# PCA shows some grouping between the data but there is also overlaps
# we want to see if we can cluster the data such a way that we can infer the labels
# following scikit-learn alsorithm cheat sheet we arrive at k-means (we are asssuming that
# we dont know the labels)

# preprocessing the data

# normalization : shifting the distribution of each attribute (feature) to have mean
# zero and standard deviation one
from sklearn.preprocessing import scale

data = scale(digits_data)

# splitting into train and test set
from sklearn.model_selection import train_test_split			# NOTE: cross_validation use is deprecated

x_train, x_test, y_train, y_test, image_train, image_test = train_test_split(data, digits_target, digits_image, test_size = 0.25, 
	random_state = 42) 											# 42 (any num would do) ensures that the split is always same; importany for reproducible result

# number of training samples and training features
nsample, nfeature = x_train.shape
print('number of training samples:', nsample)
print('number of training features:', nfeature)

# number of labels
n_digits = len(np.unique(y_train))
print(n_digits)

# use the kmeans
from sklearn import cluster

clf = cluster.KMeans(init = 'k-means++', n_init = 10, n_clusters = 10, random_state = 42)
clf.fit(x_train)

# visualize the result
fig3 = plt.figure(figsize = (8,3))

fig3.suptitle('Cluster center image', fontsize = 14, fontweight = 'bold')

for i in range(10):
	ax = fig3.add_subplot(2, 5, i + 1)
	ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap = plt.cm.binary)
	plt.axis('off')
#plt.show()

# predict some results
y_pred = clf.predict(x_test)

# print the first 100 instance of y predict
print(y_pred[:100])
# print the first 100 instance of y test
print(y_test[:100])
# also see how many are correct
print('the number of correct results in first 100 instances', list((y_pred[:100] == y_test[:100])).count('True'))

# visuakize the result using PCA for more insights
x_pca = PCA(n_components = 2).fit_transform(x_train)

# computed cluster and get the labels of the train data 
cluster = clf.fit_predict(x_train)									# we can also go with predic() as fit is already done

fig4, ax = plt.subplots(1, 2, figsize = (8, 4))
fig4.suptitle('Predicted versus training labels', fontsize = 14, fontweight = 'bold')
ax[0].scatter(x_pca[:, 0], x_pca[:, 1], c = cluster, edgecolor = 'black')
ax[0].set_title('Predicted training labels')
ax[1].scatter(x_pca[:,0], x_pca[:, 1], c = y_train, edgecolor = 'black')
plt.show()
# the above prediction and visualisation give us an idea that the model is not performing well

## evaluation of the model
# let us now compute some metrics to to evaluate the model exactly

# confusion metrics: used to describe the performance of a classification model
# in general an entry C(i, j) indicates the number of observations that are known to be of class i 
# but predicted as class j. We can calculate accuracy, precision etc all from this matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred))

## using another model : support vector classification
from sklearn import svm

# set up the svc model
svc_model = svm.SVC(gamma = 0.001, C = 100.0, kernel = 'linear')

# fit the model with train data and target labels
svc_model.fit(x_train, y_train)

# see the score
print('accuracy of the model:', svc_model.score(x_test, y_test))

# we arbitrarily selected the parameters in this model but it can be adjusted by grid search with cross validation

# train and test data split
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(digits_data, digits_target, test_size = 0.5, random_state = 0)

# import GridSearchCV
from sklearn.model_selection import GridSearchCV

# set parameter candidates
parameter_candidates = [{'C' : [1, 10, 100, 1000], 'kernel' : ['linear']},
	{'C' : [1, 10, 100, 1000], 'gamma' : [0.001, 0.00001], 'kernel' : ['rbf']}]

# create a classifier with this parameter candidate
clf = GridSearchCV(estimator = svm.SVC(), param_grid = parameter_candidates, n_jobs = 1)

# fit the train data and labels
clf.fit(x_train_2, y_train_2)

# show result
print('best score for training data:', clf.best_score_)
print('best C:', clf.best_estimator_.C)
print('best gamma:', clf.best_estimator_.gamma)
print('best kernel:', clf.best_estimator_.kernel)

# this search shows that it can give accuracy 0.98, more than our previous model where we arbitrarily set the value

# evaluation
print('accuracy after using parameter from grid search:', clf.score(x_test_2, y_test_2))

# calssification report
print(metrics.classification_report(y_test_2, clf.predict(x_test_2)))

# and confusion matrix
print(metrics.confusion_matrix(y_test_2, clf.predict(x_test_2)))