from __future__ import print_function 

import torch
import torch.nn as nn

'''
def multiclass_npairs_loss(z, y):
    z: hidden vector of shape [bsz, n_features]
    y: ground truth of shape [bsz]

'''

'''
for a set of N  randomly sampled image/label paris, the corresponding minibatch used for  training consists of 2N pairs. 
x_2k and x_(2k-1) are two random augmentations of x_k
y_(2k-1) = y_(2k) = y_k
'''


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07,base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

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
        #contiguous(): returns itself if input tensor is already contiguous, otherwise it returns a new contiguous tensor by copying  data 
        labels = labels.contiguous().view(-1, 1)
        #view(*shape): returns a new tensor with the same data  as the self tensor but of  a different shape (the size -1 is inferred form other dimensions)
        mask = torch.eq(labels, labels.T).float().to(device)
        #T: is this  tensor with its dimensions reversed 
        #torch.eq(input, other, out=None): computes element-wise equality 
        #returns a torch.BoolTensor containing  a True at each location  where  comparison is true  

        contrast_count = features.shape[1]
        #torch.unbind(input, dim=0) -> seq
        #removes a tensor dimension 
        #returns a tuple of all slices along a given dimension 
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count
               
        # compute logits
        #torch.div(input, other, out=None) -> Tensor
        #divides each element of input input with the scalar  other and returns a new resulting tensor  
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

        # tile mask
        #repeat(*size)->Tensor
        #repeats this tensor along the specified dimensions
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
       
       '''
       torch.scatter
       torch.ones_like
       torch.arange
       '''
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
