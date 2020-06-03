# ExKMC

This repository is the official implementation of ExKMC: Expanding Explainable k-Means Clustering. 

We study algorithms for k-means clustering, focusing on a trade-off between explainability and accuracy. 
Following prior work, we partition a dataset into k clusters via a small decision tree. 
This enables us to explain each cluster assignment by a short sequence of single-feature thresholds. 
While larger trees produce more accurate clusterings, they also require more complex explanations. 
To allow flexibility, we develop a new explainable k-means clustering algorithm, ExKMC, that takes an additional parameter k' &#8805; k and outputs a decision tree with k' leaves. 
We use a new surrogate cost to efficiently expand the tree and to label the leaves with one of k clusters. 
We prove that as k' increases, the surrogate cost is non-increasing, and hence, we trade explainability for accuracy.

<img src="https://raw.githubusercontent.com/navefr/ExKMC/master/images/example.PNG">


## Installation

To install requirements:
```
pip install ExKMC
```

## Usage

```python
from ExKMC.Tree import Tree
from sklearn.datasets import make_blobs

# Create dataset
n = 100
d = 10
k = 3
X, _ = make_blobs(n, d, k, cluster_std=3.0)

# Initialize tree with up to 6 leaves, predicting 3 clusters
tree = Tree(k=k, max_leaves=2*k) 

# Construct the tree, and return cluster labels
prediction = tree.fit_predict(X)

# Tree plot saved to filename
tree.plot('filename')
```

## Notebooks
Usage examples:
* [Expand tree for Gaussians](notebooks/Example.ipynb)
* [Expand tree for text data](notebooks/Newsgroups%20example.ipynb)

## Citation
If you use ExKMC in your research we would appreciate a citation to the appropriate paper(s):

* For IMM base tree you can read our [ICML 2020 paper](https://arxiv.org/pdf/2002.12538.pdf).
   ```bash
   @article{dasgupta2020explainable,
     title={Explainable $k$-Means and $k$-Medians Clustering},
     author={Dasgupta, Sanjoy and Frost, Nave and Moshkovitz, Michal and Rashtchian, Cyrus},
     journal={arXiv preprint arXiv:2002.12538},
     year={2020}
   }
   ```
* For ExKMC expansion:
   ```bash
   will be avilable soon
   ```
## Contact
* [Nave Frost](mailto:navefrost@mail.tau.ac.il)
* [Michal Moshkovitz](https://sites.google.com/view/michal-moshkovitz)
* [Cyrus Rashtchian](https://sites.google.com/site/cyrusrashtchian/) 