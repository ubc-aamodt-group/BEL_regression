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
    #print(list(module.named_parameters()))
    #print(list(module.named_buffers()))
    #print(module.weight)
    return module


def measure_sparsity(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    total= 0
    conv_total = 0
    zero = 0
    conv_zero = 0
    for m in model.modules():
        if not hasattr(m, 'weight'):
            continue
        #t =  m.weight.data.numel()
        t =  0
        for name,para in (m.named_parameters()):
            t+= len(para.data.numpy().flatten())
        pruned = int(torch.sum(m.weight.data.eq(0)))
        if isinstance(m, torch.nn.Conv2d):
            conv_total+= t
            conv_zero += pruned
        total+=t
        zero += pruned
    print("Total patams = %d, conv params = %d, total sparsity = %.4f, Conv sparsity %.4f "%(total, conv_total, float(zero)/total, float(conv_zero)/conv_total))
    return (total, conv_total, zero, conv_zero)

def plot_distribution_pkl(model1,label):
    comp = {}
    imagenet=[]
    headpose=[]
    for n,m in model1.named_modules():
        if isinstance(m, torch.nn.Conv2d) or (isinstance(m, torch.nn.Linear) and n!= "fc_angles"):
            imagenet+=  list((m.weight.data.numpy().flatten()))
        else:
            continue
    #temp =[1,2,3,4]
    output = open("dummy.pkl", 'wb')
    pickle.dump(imagenet,output)
    output.close()
    print(output)
    output = open("poseresnet50.pkl", 'rb')
    print(output)
    weight = pickle.load(output)
    #output.close()
    print(len(weight))
    print(len(imagenet))
    n = "full network"
    #weight=headpose
    print("For layer %s, Min = %.4f, max= %.4f, length = %d"%(n,np.min(weight),np.max(weight),len(weight)))
    range_o = np.max(weight)-np.min(weight)+1
    bins = np.arange(np.min(weight)-0.5, np.max(weight)+0.5, 0.001) # fixed bin size
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

    plt.rc('font', **font)
    plt.xlim([-0.3, 0.3])

    plt.hist(weight, bins=bins,alpha=0.7, color= "blue",label="Human pose")
    #plt.title('weight distribution for layer '+n)
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    print("finished plotting")
    weight = imagenet
    print("For layer %s, Min = %.4f, max= %.4f, length = %d"%(n,np.min(weight),np.max(weight),len(weight)))
    range_o = np.max(weight)-np.min(weight)+1
    bins = np.arange(np.min(weight)-0.5, np.max(weight)+0.5, 0.001) # fixed bin size

    plt.xlim([-0.3, 0.3])

    plt.hist(weight, bins=bins,color= "red",alpha=0.7,label="ImageNet")
    plt.legend(loc=2)
    plt.savefig("plots/"+label+"_overall.pdf")
    print("finished plotting")
    plt.clf()



def plot_distribution(model,model1,label):
    comp = {}
    imagenet=[]
    headpose=[]
    var_image = {}
    var_head = {}
    kl_d = {}
    ks_d ={}
    for n,m in model1.named_modules():
        if isinstance(m, torch.nn.Conv2d) or (isinstance(m, torch.nn.Linear) and n!= "fc_angles"):
            comp[n] = m.weight.data.numpy().flatten()
            imagenet+=  list((m.weight.data.numpy().flatten()))
        else:
            continue
    layer_no= 0
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) or (isinstance(m, torch.nn.Linear) and n!= "fc_angles"):
            print(n)
            weight = m.weight.data.numpy().flatten()
            headpose+= list((m.weight.data.numpy().flatten()))
            layer_no+=1
            var_head[layer_no]=np.var(weight)
            var_image[layer_no]=np.var(comp[n])
 
            kl_d[layer_no]= scipy.stats.entropy(weight, comp[n]) 
            ks_d[layer_no]= (scipy.stats.kstest(weight, comp[n]) )[0]
        else:
            continue
        if False:
            print("For layer %s, Min = %.4f, max= %.4f, length = %d"%(n,np.min(weight),np.max(weight),len(weight)))
            range_o = np.max(weight)-np.min(weight)+1
            bins = np.arange(np.min(weight)-0.5, np.max(weight)+0.5, 0.001) # fixed bin size

            font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 14}

            plt.rc('font', **font)
            plt.xlim([np.min(weight)-0.5, np.max(weight)+0.5])

            plt.hist(weight, bins=bins,alpha=0.7, color= "blue",label="HeadPose")
            #plt.title('weight distribution for layer '+n)
            plt.xlabel('Weight')
            plt.ylabel('Frequency')
            print("finished plotting")
            weight = comp[n]
            var_image[layer_no]=np.var(weight)
            print("For layer %s, Min = %.4f, max= %.4f, length = %d"%(n,np.min(weight),np.max(weight),len(weight)))
            range_o = np.max(weight)-np.min(weight)+1
            bins = np.arange(np.min(weight)-0.5, np.max(weight)+0.5, 0.001) # fixed bin size

            plt.xlim([np.min(weight)-0.5, np.max(weight)+0.5])

            plt.hist(weight, bins=bins,color= "red",alpha=0.7,label="ImageNet")
            plt.legend(loc=2)
            plt.savefig("plots/"+label+"_"+n.replace(".","_")+".pdf")
            print("finished plotting")
            plt.clf()
    fig = plt.figure(figsize=(8,6))
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 14}

    plt.rc('font', **font)

    ax = fig.add_axes([0.13, 0.14, 0.84, 0.84]) # main axes
    print(ks_d)
    ax.plot(list(ks_d.keys()), list(ks_d.values()), marker='o',linewidth=1,linestyle="solid", color="red",markersize=3, label='KS Divergence')
    ax.set_ylabel('KS divergence')
    ax.set_xlabel('Layer Number')
    fig.savefig("plots/"+label+"_KS.pdf")
    plt.show()

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.13, 0.14, 0.84, 0.84]) # main axes
    print(kl_d)
    ax.plot(list(kl_d.keys()), list(kl_d.values()), marker='o',linewidth=1,linestyle="solid", color="tab:green",markersize=3, label='KL Divergence')
    ax.set_ylabel('KL divergence')
    ax.set_xlabel('Layer Number')

    fig.savefig("plots/"+label+"_KL.pdf")
    plt.show()


    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0.13, 0.14, 0.84, 0.84]) # main axes
    ax.plot(list(var_head.keys()), list(var_head.values()), marker='o',linewidth=1, linestyle="solid",color="tab:blue", markersize=3, label='Head pose')
    ax.plot(list(var_image.keys()), list(var_image.values()), marker='o',linewidth=1,linestyle="solid", color= "tab:orange", markersize=3, label='ImageNet')
    ax.set_ylabel('Variance')
    ax.set_xlabel('Layer Number')
    ax.legend()

    fig.savefig("plots/"+label+"_variance.pdf")
    plt.show()


    n = "full network"
    font = {'family' : 'normal',
         'weight' : 'normal',
         'size'   : 14}

    plt.rc('font', **font)
    weight=headpose
    print("For layer %s, Min = %.4f, max= %.4f, length = %d"%(n,np.min(weight),np.max(weight),len(weight)))
    range_o = np.max(weight)-np.min(weight)+1
    bins = np.arange(np.min(weight)-0.5, np.max(weight)+0.5, 0.001) # fixed bin size

    plt.xlim([-0.1, 0.1])

    plt.hist(weight, bins=bins,alpha=0.7, color= "blue",label="HeadPose")
    #plt.title('weight distribution for layer '+n)
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    print("finished plotting")
    weight = imagenet
    print("For layer %s, Min = %.4f, max= %.4f, length = %d"%(n,np.min(weight),np.max(weight),len(weight)))
    range_o = np.max(weight)-np.min(weight)+1
    bins = np.arange(np.min(weight)-0.5, np.max(weight)+0.5, 0.001) # fixed bin size

    plt.xlim([-0.1, 0.1])
    #ax = fig.add_axes([0.12, 0.12, 0.80, 0.80]) # main axes
    plt.xticks(np.arange(-0.1, 0.101, 0.05))
    plt.hist(weight, bins=bins,color= "red",alpha=0.7,label="ImageNet")
    plt.legend(loc=2)
    plt.savefig("plots/"+label+"_overall.pdf")
    print("finished plotting")
    plt.clf()


def prune_network(model,amount):
    total_params= sum([np.prod(p.size()) for p in model.parameters()])

    params_sum = 0
    params_wsum = 0
    param_c = dict()
    new_density = 1.-amount
    for name,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            #print(name)
            if name=="conv1" or ("fc_angles" in name)  or ("fc" in name) or ("multparam" in name)or ("yawm" in name)or ("pitchm" in name)or ("rollm" in name):
                continue
            for n,p in m.named_parameters():
                if "weight" in n and p.requires_grad==True:
                    p_sh = p.shape
                    p_prod = np.prod(p_sh)
                    p_sum = np.sum(p_sh)

                    params_sum += p_prod
                    param_c[name] = 1 - p_sum / p_prod
                    params_wsum += p_prod * param_c[name]
    s0 = params_sum / params_wsum
    for name,m in model.named_modules():
        #print(name,type(m))
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            #if name=="conv1" or "fc_angles" in name:
            if name=="conv1" or ("fc_angles" in name)  or ("fc" in name) or ("multparam" in name) or ("yawm" in name)or ("pitchm" in name)or ("rollm" in name):
                continue
            density =  param_c[name] * (1 - new_density) * s0
            print(density)
            prune.l1_unstructured(m, name="weight", amount=density)
            #magnitude_unstructured(m, name="weight", amount=amount)
    measure_sparsity(model)


def random_prune_network(model,amount):
    for m in model.modules():
        #print(type(m))
        if isinstance(m, torch.nn.Conv2d):
            prune.random_unstructured(m, name="weight", amount=amount)
            #magnitude_unstructured(m, name="weight", amount=amount)
    measure_sparsity(model)

def vgg_measure_sparsity(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    total= 0
    conv_total = 0
    zero = 0
    conv_zero = 0
    for lname,m in model.named_modules():
        if not hasattr(m, 'weight'):
            continue
        #t =  m.weight.data.numel()
        t =  0
        for name,para in (m.named_parameters()):
            t+= len(para.data.numpy().flatten())
        pruned = int(torch.sum(m.weight.data.eq(0)))
        if isinstance(m, torch.nn.Conv2d) or (isinstance(m, torch.nn.Linear) and lname!= "fc_angles"):
            conv_total+= t
            #print(lname)
            conv_zero += pruned
        total+=t
        zero += pruned
    print("Total patams = %d, pruinable params = %d, total sparsity = %.4f, pruinable sparsity %.4f "%(total, conv_total, float(zero)/total, float(conv_zero)/conv_total))
    return (total, conv_total, zero, conv_zero)


def vgg_prune_network(model,amount):
    for name,m in model.named_modules():
        #print(type(m))
        if isinstance(m, torch.nn.Conv2d) or (isinstance(m, torch.nn.Linear) and name!= "fc_angles"):
            prune.l1_unstructured(m, name="weight", amount=amount)
            #magnitude_unstructured(m, name="weight", amount=amount)
    vgg_measure_sparsity(model)

def random_vgg_prune_network(model,amount):
    for name,m in model.named_modules():
        #print(type(m))
        if isinstance(m, torch.nn.Conv2d) or (isinstance(m, torch.nn.Linear) and name!= "fc_angles"):
            prune.random_unstructured(m, name="weight", amount=amount)
            #magnitude_unstructured(m, name="weight", amount=amount)
    vgg_measure_sparsity(model)
