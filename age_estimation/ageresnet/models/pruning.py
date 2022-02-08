import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.prune as prune
from matplotlib import pyplot as plt
import pickle
import scipy
from scipy.stats import entropy

class MagnitudeUnstructured(prune.BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor at by magnitude.
    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the 
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        # Check range of validity of pruning amount
        prune._validate_pruning_amount_init(amount) # yes this is supposed to be an internal function, but why change what isn't broken?
        self.amount = amount

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        if nparams_toprune != 0: # k=0 not supported by torch.kthvalue
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            mask.view(-1)[topk.indices] = 0
        
        return mask

    @classmethod
    def apply(cls, module, name, amount):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """
        return super(MagnitudeUnstructured, cls).apply(
            module, name, amount=amount
        )

def magnitude_unstructured(module, name, amount):
    MagnitudeUnstructured.apply(module, name, amount)

    return module

def l1_prune_network(model,amount):
    for m in model.modules():
        #print(type(m))
        if isinstance(m, torch.nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)
            #magnitude_unstructured(m, name="weight", amount=amount)
    measure_sparsity(model)


def magnitude_prune_network(model,amount):
    for m in model.modules():
        #print(type(m))
        if isinstance(m, torch.nn.Conv2d):
            magnitude_unstructured(m, name="weight", amount=amount)
            #magnitude_unstructured(m, name="weight", amount=amount)
    measure_sparsity(model)



def random_prune_network(model,amount):
    for m in model.modules():
        #print(type(m))
        if isinstance(m, torch.nn.Conv2d):
            prune.random_unstructured(m, name="weight", amount=amount)
            #magnitude_unstructured(m, name="weight", amount=amount)
    measure_sparsity(model)


def measure_sparsity(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    total = 0
    conv_total = 0
    zero = 0
    conv_zero = 0
    for m in model.modules():
        if not hasattr(m, 'weight'):
            continue
        #t =  m.weight.data.numel()
        t =  0
        for name, para in (m.named_parameters()):
            t+= len(para.data.cpu().numpy().flatten())
        pruned = int(torch.sum(m.weight.data.eq(0)))
        if isinstance(m, torch.nn.Conv2d):
            conv_total+= t
            conv_zero += pruned
        total+=t
        zero += pruned
    print("Total patams = %d, conv params = %d, total sparsity = %.4f, Conv sparsity %.4f "%(total, conv_total, float(zero)/total, float(conv_zero)/conv_total))
    return (total, conv_total, zero, conv_zero)