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

## Citation
If you use ExKMC in your research we would appreciate a citation to the appropriate paper(s):

* For IMM base tree:
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
