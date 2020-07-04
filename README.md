# Supervised Contrastive Learning
## arguments
```python
# features: hidden vector of shape [bsz, n_views, f_dim]
# labels: ground truth of shape [bsz]
# masks: contrastive mask of sahpe [bsz, bsz]
# mask_{i,j}=1 if sample j has the same class as sample i
```

## Pretraining stage:
```
python3 main_supcon.py 
```
## Linear evaluation stage: 
```
python3 main_linear.py 
```
