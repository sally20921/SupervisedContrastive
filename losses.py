from __future__ import print_function 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
'''
def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(-target*F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(-target*F.log_softmax(logits, -1), -1))
'''
'''
class NpairLoss(nn.Module):
     """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """

    def __init__(self, l2_reg = 0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg
    def forward(self, anchor, positive, target):
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)
`       
        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss
        anchors = torch.unsqueeze(anchors, dim=1) #(n,1,embedding_size)
        positives = torch.unsqueeze(positives, dim=1) #(n,1,embedding_size)

        x = torch.matmul(anchors, (negatives-positives).transpose(1,2)) #(n, 1, n-1)
        x = torch.sum(torch.exp(x),2) #(n,1)
        loss = torch.mean(torch.log(1+x))
        return loss 
'''
"""def multiclass_npairs_loss(z, y):
    z: hidden vector of shape [bsz, n_features]
    y: ground truth of shape [bsz] batch size """

"""
for a set of N  randomly sampled image/label paris, the corresponding minibatch used for  training consists of 2N pairs. 
x_2k and x_(2k-1) are two random augmentations of x_k
y_(2k-1) = y_(2k) = y_k
"""
"""
features = model(images)
"""

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels, mask):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        batch_size = features.shape[0]

        '''for Simclr,
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        returns a  2-D tensor  with ones on the  diagonal and zeros elsewhere)
        '''

        #contiguous(): returns itself if input tensor is already contiguous, otherwise it returns a new contiguous tensor by copying  data 
        labels = labels.contiguous().view(-1, 1)
        #view(*shape): returns a new tensor with the same data  as the self tensor but of  a different shape (the size -1 is inferred form other dimensions)
        mask = torch.eq(labels, labels.T).float().to(device) #mask2  
        '''indicator  function: calculate only those labels are same'''
        #T: is this  tensor with its dimensions reversed 
        #torch.eq(input, other, out=None): computes element-wise equality 
        #returns a torch.BoolTensor containing  a True at each location  where  comparison is true  

        contrast_count = features.shape[1] #n_views
        #torch.unbind(input, dim=0) -> seq
        #removes a tensor dimension 
        #returns a tuple of all slices along a given dimension 
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count
               
        # compute logits
        #torch.div(input, other, out=None) -> Tensor
        #divides each element of input input with the scalar  other and returns a new resulting tensor  
        '''anchor_feature, contrast_feature'''
        #positives
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        #torch.matmul(input,  other, out=None) -> Tensor 
        #matrix product of two tensors 

        # for numerical stability
        #torch.max(input)->Tensor 
        #returns the maximum value  of all elements in the input tensor
        '''logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()'''
        logits = achor_dot_contrast
        #detach:  declared not  to  need gradient 

        # tile mask(or mask1) 
        #repeat(*size)->Tensor
        #repeats this tensor along the specified dimensions
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases       
'''
       mask-out self-contrast cases
       torch.scatter(dim, index, src): write all values from  tensor  src into  self at  the  indices specified in  the  index tnesor. 
       torch.ones_like: returns a tensor filled with the scalar value 1, with the  same size as input
       torch.arange: returns a 1-D tensor with values from interval [start, end) with step size beginning  from start
'''

       logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask #multiply mask1 and mask2

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
