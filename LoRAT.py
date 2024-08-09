import math
import torch
import torch.nn as nn
import functools


def count_parameters(model, to_million=False, lora=False):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if to_million:
        total_params /= 1e6
        trainable_params /= 1e6
        if lora:
            print(f"Lora After Total Train Parameters :{trainable_params:.2f}M ")
        else:
            print(f"Lora Before Total Train Parameters :{total_params:.2f}M ")
    else:
        if lora:
            print(f"Lora After Total Train Parameters :{trainable_params:.2f} ")
        else:
            print(f"Lora Before Total Train Parameters :{total_params:.2f} ")


class lora_layer(nn.Module):
    def __init__(self, original_module, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.,
                 merge_weights: bool = True, *args, **kwargs):
        super(lora_layer, self).__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.merge_weights = merge_weights
        self.merged = False
        self.original_module = original_module
        self.mode = kwargs["mode"]

        if r > 0:
            self.lora_A = nn.Parameter(self.original_module.weight.new_zeros((r, self.original_module.weight.size(1))))
            self.lora_B = nn.Parameter(self.original_module.weight.new_zeros((self.original_module.weight.size(0), r)))
            self.scaling = self.lora_alpha / self.r
            if self.mode == 'part':
                for param in self.original_module.parameters():
                    param.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.original_module.train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.original_module.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.original_module.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            original_output = self.original_module(x)
            lora_result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0,
                                                                                                      1)) * self.scaling
            return original_output + lora_result
        else:
            return self.original_module(x)


class LoRAT(nn.Module):
    def __init__(self, model, rank=1, alpha=32, tune='qkov', mode='all'):
        super(LoRAT, self).__init__()
        self.bert = model
        self.tune = tune
        count_parameters(self.bert, to_million=True)

        if mode == 'all':
            for param in self.bert.parameters():
                param.requires_grad = False

        for name, module in self.bert.named_modules():
            if ('attention.self.query' in name and 'q' in self.tune) or \
                    ('attention.self.key' in name and 'k' in self.tune) or \
                    ('attention.self.value' in name and 'v' in self.tune) or \
                    ('attention.output.dense' in name and 'o' in self.tune):
                lora_module = lora_layer(module, r=rank, lora_alpha=alpha, mode=mode)
                parent_module = functools.reduce(getattr, name.split('.')[:-1], self.bert)
                setattr(parent_module, name.split('.')[-1], lora_module)
        count_parameters(self.bert, to_million=True, lora=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs
