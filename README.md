# K-Means and K-Means++ Clustering Project

The source contains:

Implementation of k-means/k-means++ clustering algorithm accepting three arguments:
* X denoting the n data points, each represented as a real valued vector of attributes.
* k denoting the number of clusters into which the data points are to be clustered into.
* init - which can take string values “random” or “k-means++”. When init==“random”, the output follows k-means clustering. However, when init==“k-means++”, the kmeans++ initialization criterion is used for selecting the cluster centers.

The file movies.csv contains information about 4,803 movies. Each line in the file corresponds to a particular movie. The attributes are as follows:
* budget (USD)
* genres (JSON string)
* homepage (URL)
* id (movie identifier)
* keywords (JSON string)
* original language
* original title
* overview
* popularity
* production companies (JSON string)
* production countries (JSON string)
* release date
* revenue
* runtime
* spoken languages (JSON string)
* status
* tagline
* title
* vote average
* vote count
