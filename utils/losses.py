import torch
import torch.nn as nn
from typing import Tuple, List
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable


# focal loss
def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


# LDAM loss
class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        # import ipdb;ipdb.set_trace();
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, MetaLinear) or isinstance(m, MetaConv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(MetaModule):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    MetaConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    MetaBatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MetaLinear_Norm(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        temp = nn.Linear(*args, **kwargs)
        temp.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.register_buffer('weight', to_var(temp.weight.data.t(), requires_grad=True))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

    def named_leaves(self):
        return [('weight', self.weight)]


class ReconstructionLoss(torch.nn.Module):
  """
  Loss function for tabular data reconstruction.
  Loss function for multimodal contrastive learning based off of the CLIP paper.
  
  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(self, 
               num_cat: int, num_con: int, cat_offsets: torch.Tensor) -> None:
    super(ReconstructionLoss, self).__init__()
    
    self.num_cat = num_cat
    self.num_con = num_con
    self.register_buffer('cat_offsets', cat_offsets, persistent=False)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, out: Tuple, y: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    import ipdb;ipdb.set_trace();
    B, _, D = out[0].shape
    # (B*N1, D)
    out_cat = out[0].reshape(B*self.num_cat, D)
    # (B, N2)
    out_con = out[1].squeeze(-1)
    target_cat = (y[:, :self.num_cat].long()+self.cat_offsets).reshape(B*self.num_cat)
    target_con = y[:, self.num_cat:]
    mask_cat = mask[:, :self.num_cat].reshape(B*self.num_cat)
    mask_con = mask[:, self.num_cat:]

    # cat loss
    prob_cat = self.softmax(out_cat)
    onehot_cat = torch.nn.functional.one_hot(target_cat, num_classes=D)
    loss_cat = -onehot_cat * torch.log(prob_cat+1e-8)
    loss_cat = loss_cat.sum(dim=1)
    loss_cat = (loss_cat*mask_cat).sum()/mask_cat.sum()   
    loss_cat = loss_cat.sum() / (loss_cat.numel())
    # con loss
    loss_con = (out_con-target_con)**2
    loss_con = (loss_con*mask_con).sum()/mask_con.sum()   
    loss_con = loss_con.sum() / (loss_con.numel())
    
    loss = (loss_cat + loss_con)/2
  
    return loss, prob_cat, target_cat, mask_cat
  

# class ReconstructionLoss_MLP(torch.nn.Module):
#     def __init__(self, num_cat: int, num_con: int, cat_offsets: torch.Tensor) -> None:
#         super(ReconstructionLoss_MLP, self).__init__()
        
#         self.num_cat = num_cat
#         self.num_con = num_con
#         self.register_buffer('cat_offsets', cat_offsets, persistent=False)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, out: Tuple, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
#         import ipdb;ipdb.set_trace();
#         B = y.shape[0]
#         out_con, out_cat = out  # 获取重构的连续和离散特征

#         target_cat = (y[:, :self.num_cat].long()+self.cat_offsets).reshape(B*self.num_cat)
#         target_con = y[:, self.num_cat:]
#         mask_cat = mask[:, :self.num_cat].reshape(B*self.num_cat)
#         mask_con = mask[:, self.num_cat:]

#         # cat loss
#         prob_cat = self.softmax(out_cat.reshape(-1))
#         onehot_cat = torch.nn.functional.one_hot(target_cat, num_classes=out_cat.shape[1])
#         loss_cat = -onehot_cat * torch.log(prob_cat+1e-8)
#         loss_cat = loss_cat.sum(dim=1)
#         loss_cat = (loss_cat*mask_cat).sum()/mask_cat.sum()   
#         loss_cat = loss_cat.sum() / (loss_cat.numel())
#         # con loss
#         loss_con = (out_con-target_con)**2
#         loss_con = (loss_con*mask_con).sum()/mask_con.sum()   
#         loss_con = loss_con.sum() / (loss_con.numel())
        
#         loss = (loss_cat + loss_con)/2
#         return loss



# class ReconstructionLoss_MLP(torch.nn.Module):
#     def __init__(self, num_cat: int, num_con: int, cat_offsets: torch.Tensor) -> None:
#         super(ReconstructionLoss_MLP, self).__init__()
        
#         self.num_cat = num_cat  
#         self.num_con = num_con 
#         self.register_buffer('cat_offsets', cat_offsets, persistent=False)  

#     def forward(self, results, y, mask=None):
#         x_cont_recon, x_cat_recon = results
#         target_cat = y[:, :self.num_cat].long()  
#         mask_cat = mask[:, :self.num_cat] 
#         loss_cat = 0
#         for i in range(self.num_cat):
#             logits = x_cat_recon[i]  
#             target = target_cat[:, i] 
#             mask_i = mask_cat[:, i]  
#             loss_i = F.cross_entropy(logits, target, reduction='none') 
#             loss_i = (loss_i * mask_i).sum() / (mask_i.sum() + 1e-8) 
#             loss_cat += loss_i
#         loss_cat = loss_cat / self.num_cat 

#         target_con = y[:, self.num_cat:]  
#         mask_con = mask[:, self.num_cat:] if mask is not None else torch.ones_like(target_con, dtype=torch.bool)
#         loss_con = (x_cont_recon - target_con) ** 2  
#         loss_con = (loss_con * mask_con).sum() / (mask_con.sum() + 1e-8)  
#         loss_con = loss_con.sum() / (loss_con.numel()) 


#         # loss_history_cat = [loss_cat.item()]  
#         # loss_history_con = [loss_con.item()]
#         # alpha = 0.9  
#         # loss_avg_cat = alpha * loss_history_cat[-1] + (1 - alpha) * loss_cat.item()
#         # loss_avg_con = alpha * loss_history_con[-1] + (1 - alpha) * loss_con.item()
#         # loss_history_cat.append(loss_avg_cat)
#         # loss_history_con.append(loss_avg_con)
#         # loss_weight = loss_avg_cat / (loss_avg_con + 1e-8)

#         print(f"===================== loss_con: {loss_con} ======================")
#         print(f"===================== loss_cat: {loss_cat} ======================")
#         # loss = (loss_cat + loss_weight * loss_con) / 2

#         loss = (loss_cat + loss_con)/2
#         prob_cat = x_cat_recon 
#         return loss, prob_cat, target_cat, mask_cat




# class ReconstructionLoss_MLP(torch.nn.Module):
#     def __init__(self, num_cat: int, num_con: int, cat_offsets: torch.Tensor) -> None:
#         super(ReconstructionLoss_MLP, self).__init__()
        
#         self.num_cat = num_cat  
#         self.num_con = num_con 
#         self.register_buffer('cat_offsets', cat_offsets, persistent=False)

#     def forward(self, results, y, mask=None):
#         x_cont_recon, x_cat_recon = results
#         target_cat = y[:, :self.num_cat].long()  
#         mask_cat = mask[:, :self.num_cat] if mask is not None else torch.ones_like(target_cat, dtype=torch.bool)
        
#         # 计算离散特征的交叉熵损失
#         loss_cat = 0
#         entropy_cat = 0  # 用于保存每个离散特征的熵值
#         for i in range(self.num_cat):
#             logits = x_cat_recon[i]  # 第i个离散特征的重建值，(B, cat_length)
#             target = target_cat[:, i]  # 第i个离散特征的真实标签，(B,)
#             mask_i = mask_cat[:, i]  # 第i个离散特征的掩码， (B,)

#             # 计算交叉熵损失
#             loss_i = F.cross_entropy(logits, target, reduction='none')  # (B,)
#             loss_i = (loss_i * mask_i).sum() / (mask_i.sum() + 1e-8)  # 计算加权平均损失
#             loss_cat += loss_i

#             # 计算熵
#             prob_cat = F.softmax(logits, dim=-1)  # 计算概率分布
#             entropy_i = -(prob_cat * torch.log(prob_cat + 1e-8)).sum(dim=-1)  # 计算熵
#             entropy_cat += entropy_i

#         loss_cat = loss_cat / self.num_cat  # 归一化离散特征的损失
#         entropy_cat = entropy_cat / self.num_cat  # 归一化离散特征的熵

#         # 计算连续特征的均方误差损失
#         target_con = y[:, self.num_cat:]  # 连续特征目标
#         mask_con = mask[:, self.num_cat:] if mask is not None else torch.ones_like(target_con, dtype=torch.bool)
#         loss_con = (x_cont_recon - target_con) ** 2  # MSE损失
#         loss_con = (loss_con * mask_con).sum() / (mask_con.sum() + 1e-8)  # 计算加权平均损失
#         loss_con = loss_con.sum() / (loss_con.numel())  # 对所有样本进行归一化

#         loss = (loss_cat + loss_con) / 2
#         prob_cat = x_cat_recon  # 离散特征的预测概率
#         return loss, prob_cat, target_cat, mask_cat




class ReconstructionLoss_MLP(nn.Module):
    def __init__(self, num_cat, num_con, opt_dict):
        super(ReconstructionLoss_MLP, self).__init__()
        self.field_lengths_tabular = torch.load(opt_dict['dataset_config']['field_lengths_tabular'])
        self.field_lengths_tabular = [int(x) for x in self.field_lengths_tabular]
        self.cat_lengths_tabular = []
        self.con_lengths_tabular = []
        for x in self.field_lengths_tabular:
            if x == 1:
                self.con_lengths_tabular.append(x)
            else:
                self.cat_lengths_tabular.append(x)
        self.num_cat = num_cat
        self.num_con = num_con 

    def forward(self, results, y, mask=None):
        # import ipdb;ipdb.set_trace();
        recon_con, recon_cat = results
        target_cat = y[:, :self.num_cat].long()  
        target_con = y[:, self.num_cat:]  
        mask_cat = mask[:, :self.num_cat]  
        mask_con = mask[:, self.num_cat:]  

        loss_cat = 0
        recon_cat_split = torch.split(recon_cat, self.cat_lengths_tabular, dim=1)  
        for i, (logits, target, mask_i) in enumerate(zip(recon_cat_split, target_cat.unbind(dim=1), mask_cat.unbind(dim=1))):
            loss_cat += (F.cross_entropy(logits, target, reduction='none') * mask_i).sum() / (mask_i.sum() + 1e-8)
        loss_cat = loss_cat / self.num_cat  

        loss_con = (recon_con - target_con) ** 2 
        loss_con = (loss_con * mask_con).sum() / (mask_con.sum() + 1e-8)  

        loss = (loss_cat + loss_con) / 2 

        cat_labels = [torch.argmax(logits, dim=1) for logits in recon_cat_split]

        return loss, cat_labels, recon_con




class KLLoss:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(self, log_probs, target_probs):
        log_probs = F.log_softmax(log_probs / self.temperature, dim=1)
        target_probs = F.softmax(target_probs / self.temperature, dim=1)
        kl_loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
        return kl_loss
    

