# ExKMC

This repository is the official implementation of ExKMC: Expanding Explainable k-Means Clustering. 


## Installation

To install requirements:
```
pip install ExKMC
```

## Usage

```python
from ExKMC.Tree import Tree
from sklearn.datasets import make_blobs

# Initialize tree with up to 6 leaves, predicting 3 clusters
tree = Tree(k=3, max_leaves=6) 

# Construct the tree, and return cluster labels
prediction = tree.fit_predict(make_blobs(100, 10, 3))

# Tree plot saved to filename
tree.plot('filename')
```