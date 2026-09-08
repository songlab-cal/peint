from typing import Tuple, Optional, List

import torch
from torch import Tensor
import torch.nn as nn
from esm.model.esm2 import ESM2

from peint.models._transformer_modules import (
    FlashMHAEncoderBlock,
    ESMEncoderBlock
)

##########################
# Flash Based ESM2 Model #
##########################

class ESM2Flash(ESM2):
    '''ESM2 model with FlashAttention mechanism'''
    def __init__(self,
                num_layers: int = 30,
                embed_dim: int = 640,
                attention_heads: int = 20,
                alphabet = "ESM-1b",
                token_dropout: bool = True,
                use_bias: bool = True,
                add_bias_kv: bool = False,
                **kwargs) -> None:
        
        self.add_bias_kv = add_bias_kv
        self.dropout_p = kwargs.get('dropout_p', 0.0)
        self.attention_heads = attention_heads
        self.use_bias = use_bias
        super(ESM2Flash, self).__init__(num_layers, embed_dim, attention_heads, alphabet, token_dropout)

    def _init_submodules(self):
        super()._init_submodules()
        self.layers = nn.ModuleList(
            [
                FlashMHAEncoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim= 4 * self.embed_dim,
                    attention_heads=self.attention_heads,
                    add_bias_kv=self.add_bias_kv,
                    dropout_p=self.dropout_p,
                    use_bias = self.use_bias,
                    layer_idx=i
                )
            for i in range(self.num_layers)
        ])

    def forward(self, x, repr_layers = [], **kwargs):
        h_x = self.embed_scale * self.embed_tokens(x)

        padding_mask = x.eq(self.padding_idx)

        if self.token_dropout:
            h_x.masked_fill_((x == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (x == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            h_x = h_x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            h_x = h_x * (1 - padding_mask.unsqueeze(-1).type_as(h_x)) #multiplying by 0 the padding positions, not sure if actually needed

        hidden_representations = {}

        if 0 in repr_layers:
            hidden_representations[0] = h_x

        #mask is dealt with in the forward function of the layer
        encoder_kwargs = {'x_padding_mask': ~padding_mask} #padding mask is inverted - 1 means attend

        for layer_idx, layer in enumerate(self.layers):
            h_x = layer(h_x, **encoder_kwargs)

            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = h_x

        h_x = self.emb_layer_norm_after(h_x) #final_layer_norm
        h_x = self.lm_head(h_x)

        result = {'logits': h_x, 'representations': hidden_representations}

        return result
    

############################################
# Non A100 version of the above model ######
############################################
class ESM2Model(ESM2):
    '''ESM2 model with FlashAttention mechanism'''
    def __init__(self,
                num_layers: int = 30,
                embed_dim: int = 640,
                attention_heads: int = 20,
                alphabet = "ESM-1b",
                token_dropout: bool = True,
                use_bias: bool = True,
                add_bias_kv: bool = False,
                **kwargs) -> None:
        
        self.add_bias_kv = add_bias_kv
        self.dropout_p = kwargs.get('dropout_p', 0.0)
        self.attention_heads = attention_heads
        self.use_bias = use_bias
        super(ESM2Model, self).__init__(num_layers, embed_dim, attention_heads, alphabet, token_dropout)

    def _init_submodules(self):
        super()._init_submodules()
        self.layers = nn.ModuleList(
            [
                ESMEncoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim= 4 * self.embed_dim,
                    attention_heads=self.attention_heads,
                    add_bias_kv=self.add_bias_kv,
                    dropout_p=self.dropout_p,
                    use_bias = self.use_bias,
                    layer_idx=i
                )
            for i in range(self.num_layers)
        ])

    def forward(self, x, repr_layers = [], **kwargs):
        h_x = self.embed_scale * self.embed_tokens(x)

        padding_mask = x.eq(self.padding_idx)

        if self.token_dropout:
            h_x.masked_fill_((x == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (x == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            h_x = h_x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            h_x = h_x * (1 - padding_mask.unsqueeze(-1).type_as(h_x)) #multiplying by 0 the padding positions, not sure if actually needed

        hidden_representations = {}

        if 0 in repr_layers:
            hidden_representations[0] = h_x

        for layer_idx, layer in enumerate(self.layers):
            h_x, attn = layer(h_x, attn_mask = padding_mask)

            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = h_x

        h_x = self.emb_layer_norm_after(h_x) #final_layer_norm
        h_x = self.lm_head(h_x)

        result = {'logits': h_x, 'representations': hidden_representations}

        return result