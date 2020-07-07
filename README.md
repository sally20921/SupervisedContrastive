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
- "with  logit": it means that you are applying a softmax function to logit  numbers to normalize it 
- "logit": the vector  of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passes to a  normalization function.
-  logits typically become an input  to the softmax function

### cross-entropy loss
- KL-divergence  between two  discrete distributions: the  label distribution ( a discrete distribution of 1-hot vectors) and the  empirical distribution of logits
- each class is assigned a target (usually 1-hot) vetor and the  logits at the last layer of the  network, after  a softmax transformation, are  gradually transformed towards the  target vector 

###  contrastive loss
 -  for a given anchor point, the  first  force  pulls the anchor closer  in representation space to other points 
- second force  pushes the anchor  farther away form other points  
- computes  the  inner (dot) product between the  normalized vectors z-i and z-j(i) in 128 dimensional space 
-  the denominator has a  total  of  2n-1 terms 
- for any i, the encoder is tuned to maximize  the  numerator of the log argument  while simultaneously minimizing  its denominator 
