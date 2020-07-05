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

### torch.nn.BCEWithLogitsLoss
- this loss  combines a sigmoid layer and the BCELoss in one single class 
- more numerically stable 
- take advantage  of log-sum-exp trick for numerical stability
- parameters : reduction ('mean': the sum of the  output will be  divided by the  number of elements in the output, 'sum': the  output will be  summed)
