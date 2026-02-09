import math
from typing import Tuple, Optional, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from esm.modules import gelu #use esm gelu

# Architecture constants
ROPE_BASE_FREQUENCY = 10000.0  # Base frequency for Rotary Position Embeddings
FFN_EXPANSION_FACTOR = 4  # FFN hidden dim = FFN_EXPANSION_FACTOR * embed_dim
MIN_FLASH_ATTN_COMPUTE_CAPABILITY = 8.0  # Ampere (A100, RTX 30xx), Ada (RTX 40xx), Hopper (H100)


def _check_cuda_compute_capability(min_capability: float = MIN_FLASH_ATTN_COMPUTE_CAPABILITY) -> bool:
    """Check if CUDA device supports Flash Attention (compute capability >= 8.0).

    Flash Attention requires Ampere, Ada, or Hopper GPUs (A100, RTX 3090, RTX 4090, H100).
    Turing GPUs (T4, RTX 2080) are not supported in Flash Attention 2.x.
    """
    if not torch.cuda.is_available():
        return False

    try:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        compute_capability = float(f"{capability[0]}.{capability[1]}")
        return compute_capability >= min_capability
    except Exception:
        return False


# Check both import availability AND GPU compute capability
FLASH_AVAILABLE = False
try:
    from flash_attn.bert_padding import (
        pad_input,
        IndexFirstAxis
    )
    from flash_attn import (
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_kvpacked_func
    )
    from flash_attn.layers.rotary import (
        RotaryEmbedding as FlashRotaryEmbedding,
        apply_rotary_emb_torch
    )

    index_first_axis = IndexFirstAxis.apply

    # Import succeeded, now check GPU compatibility
    if _check_cuda_compute_capability():
        FLASH_AVAILABLE = True
    else:
        print("Flash Attention installed but GPU compute capability < 8.0. "
              "Using standard PyTorch attention.")
except ImportError:
    print("Flash Attention not available. Using standard PyTorch attention.")

if FLASH_AVAILABLE:
    def unpad_input(hidden_states, attention_mask):
        """
        Recent updates to flash_attn changed the number of returns from this function.
        I'm just replicating the old behavior for consistency's sake across versions.
        Copied from
        https://github.com/Dao-AILab/flash-attention/blob/e2e4333c955b829d0e6087d27ee435f55c80d3a5/flash_attn/bert_padding.py#L98
        """
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
        return (
            index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
        )

########################################################
# Time Embedding Module (Geometric Spaced Frequencies) #
########################################################

class GeometricTimeEmbedder(nn.Module):

    def __init__(self, frequency_embedding_size=256, start=1e-5, stop=0.25):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.start=start
        self.stop=stop 
    
    def timestep_embedding(self, timesteps, dim):
        freqs = torch.tensor(np.geomspace(start=self.start, stop=self.stop, num=dim//2), dtype=timesteps.dtype).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_emb = self.timestep_embedding(t, dim = self.frequency_embedding_size)
        return t_emb


######################################
# Flash Multi-Head Attention Modules #
######################################

class RopeFlashMHA(nn.Module):
    '''Flash Multi-Head Attention module for transformer, with Rotary Embedding'''
    def __init__(self, embed_dim, num_heads, bias=True, add_bias_kv=False, dropout=0.0, self_attn =True, causal: bool = False, layer_idx=None):
        super().__init__()

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.causal = causal
        self.self_attn = self_attn
        self.layer_idx = layer_idx
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv

        hidden_size = embed_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        #NOTE: the FlashRotaryEmbedding module is used here, make sure that pos_idx_in_fp32 is set to True,
        #otherwise positional indexing will fail for large indices due to range limitations of bf16
        # i.e. 265 = 270 due to rounding
        self.rot_emb = FlashRotaryEmbedding(
            dim=self.head_dim,
            base=ROPE_BASE_FREQUENCY,
            interleaved=False,
            scale_base=None,
            pos_idx_in_fp32=True  # Important for bf16 to avoid rounding errors
        )

    def forward(self,
                x,
                y = None,
                x_padding_mask = None,
                y_padding_mask = None,
                ):
        
        Bx, Lx, Dx = x.size()
        
        if self.self_attn:
            #project x to q, k, v
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            #y comes from encoder, provides keys and values
            assert y is not None, "Cross attention requires y input"
            q = self.q_proj(x)
            k = self.k_proj(y)
            v = self.v_proj(y)

        #rescale q 
        q *= self.head_dim ** -0.5

        q = rearrange(q, 'b l (n h) -> b l n h', n=self.num_heads)
        k = rearrange(k, 'b l (n h) -> b l n h', n=self.num_heads)
        v = rearrange(v, 'b l (n h) -> b l n h', n=self.num_heads)
        
        #NOTE: flash atten's rot emb performs this in-place
        q, k = self.rot_emb(
            q,
            torch.stack([k, v], dim = 2),
            seqlen_offset = 0,
            max_seqlen = max(q.shape[1], k.shape[1]) #this is important if q, k are not the same shape
        )

        #at this point, k contains k and v, and is of shape [B, L, 2, N, H]
        if x_padding_mask is None:
            x_padding_mask = torch.ones(Bx, Lx, device=x.device, dtype=torch.bool)

        q, idx_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, x_padding_mask)

        if self.self_attn:
            k, idx_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, x_padding_mask) #k = kv

            qkv = torch.cat([q.unsqueeze(1), k], dim=1) # (total_nonpad, 3, N, H)
            out = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens_q,
                max_seqlen_q,
                dropout_p=self.dropout,
                softmax_scale=1., #q has been scaled already
                causal=self.causal,
            )
        else:
            #cross attention
            if y_padding_mask is None:
                By, Ly, Dy = y.size()
                y_padding_mask = torch.ones(By, Ly, device=y.device, dtype=torch.bool)
            k, idx_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, y_padding_mask)

            out = flash_attn_varlen_kvpacked_func(
                q,
                k,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=self.dropout,
                softmax_scale=1., #q has been scaled already
                causal=self.causal,
            )

        out = pad_input(out, idx_q, Bx, Lx) #repad
        out = rearrange(out, '... h d -> ... (h d)') #concatenate heads

        return self.out_proj(out) #linear projection
    
class FlashMHAEncoderBlock(nn.Module):
    '''Flash Multi-Head Attention Encoder Block for transformer, with Rotary Embedding.
    This implementation yields identical results to that of ESM2's MHA.
    '''
    def __init__(self, 
                 embed_dim, 
                 ffn_embed_dim,
                 attention_heads,
                 use_bias = True,
                 add_bias_kv = False,
                 dropout_p = 0.0, 
                 layer_idx=None, 
                 **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.ffn_embed_dim = ffn_embed_dim
        self.add_bias_kv = add_bias_kv
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx
        self.attention_heads = attention_heads

        #initialized submodules
        self.self_attn = RopeFlashMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            bias=use_bias,
            add_bias_kv=add_bias_kv,
            dropout=dropout_p,
            self_attn=True,
            causal=False,
            layer_idx=layer_idx,
        )

        #layer norms
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        #ffn layers
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, **kwargs):
        '''
        Forward pass using the x sequence. kwargs should contain the x padding mask as x_padding_mask.
        '''
        
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, **kwargs)

        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x
    
class FlashMHADecoderBlock(nn.Module):
    '''Flash Multi-Head Attention Decoder Block for transformer, with Rotary Embedding.
    Will perform both causal self-attention and cross-attention.'''
    def __init__(self,
                 embed_dim, 
                 ffn_embed_dim,
                 attention_heads,
                 use_bias = True,
                 add_bias_kv = False,
                 dropout_p = 0.0, 
                 layer_idx=None, 
                 **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.ffn_embed_dim = ffn_embed_dim
        self.add_bias_kv = add_bias_kv
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx
        self.attention_heads = attention_heads

        self.self_attn = RopeFlashMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            self_attn=True,
            causal=True,
            layer_idx=layer_idx,
            dropout=dropout_p
        )
        self.cross_attn = RopeFlashMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            self_attn=False,
            causal=False,
            layer_idx=layer_idx,
            dropout=dropout_p
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        
    def forward(self, x, y, **kwargs):
        '''
        Forward pass using the x sequence and the y sequence. kwargs should contain both the 
        x and y padding masks as x_padding_mask and y_padding_mask.

        The unpadding and repadding is done in the FlashMHA module so that rotary embeddings can be used.

        My notation is not ideal - the x, y refer to the x,y sequences from an x,y,t trio, but since this is
        the decoder, x is y and y is x (i.e. y provides the q, while k,v come from x).
        This is handled in the overall transformer forward pass (reassigning x to y and the padding masks accordingly).
        '''

        self_attn_kwargs = {'x_padding_mask': kwargs.get('x_padding_mask', None)}

        #causal self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            x=x,
            **self_attn_kwargs
        )
        x = x + residual

        #cross attention
        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attn(
            x=x,
            y=y,
            **kwargs
        )
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x
    
##########################
## KV Caching Functions ##
##########################

class KVCached_MHCA(RopeFlashMHA):
    '''For single token decoding - q will be shape B x 1 x D'''
    def __init__(self,
                embed_dim: int,
                num_heads: int,
                max_seq_len: int,
                bias: bool=True, 
                dropout: float =0.0,
                layer_idx: Optional[int]=None):
        super().__init__(
            embed_dim = embed_dim,
            num_heads = num_heads,
            bias=bias,
            self_attn=False,
            causal=False,
            dropout=dropout,
            layer_idx=layer_idx)
        
        self.max_seq_len = max_seq_len
        self.kv_cache = None
        self.cache_size = 0

    def init_kv_cache(self, batch_size):
        # Initialize the KV cache with empty tensors
        self.kv_cache = torch.empty(batch_size, self.max_seq_len, 2, self.num_heads, self.head_dim, device=self.q_proj.weight.device)
        self.cache_size = 0

    def forward(self, x, y=None, x_padding_mask=None, y_padding_mask=None, decoder_cache_size=0):
        Bx, Lx, Dx = x.size()

        if self.kv_cache is None or self.kv_cache.size(0) != Bx:
            self.init_kv_cache(Bx)

        q = self.q_proj(x)
        q *= self.head_dim ** -0.5
        q = rearrange(q, 'b l (n h) -> b l n h', n=self.num_heads)

        if y is not None:
            # Update KV cache
            By, Ly, Dy = y.size()
            k = self.k_proj(y)
            v = self.v_proj(y)
            k = rearrange(k, 'b l (n h) -> b l n h', n=self.num_heads)
            v = rearrange(v, 'b l (n h) -> b l n h', n=self.num_heads)

            #rotate q and k,v here
            q, kv = self.rot_emb(q, torch.stack([k,v], dim=2), seqlen_offset=0, max_seqlen = self.max_seq_len)
            
            self.kv_cache[:, self.cache_size:self.cache_size+Ly] = kv
            self.cache_size += Ly

        else:
            # Use cached KV
            kv = self.kv_cache[:, :self.cache_size].to(q.dtype)

            #rotate just q, use the cached_cos_sin
            #check if need to update
            if Lx + decoder_cache_size > self.max_seq_len:
                self.rot_emb._update_cos_sin_cache(Lx + decoder_cache_size, device = q.device, dtype=q.dtype)
            
            cos, sin = self.rot_emb._cos_cached, self.rot_emb._sin_cached
            q = apply_rotary_emb_torch(q, cos[decoder_cache_size], sin[decoder_cache_size]) #this gets the seqlen offset for you

        if x_padding_mask is None:
            x_padding_mask = torch.ones(Bx, Lx, device=x.device, dtype=torch.bool)

        q, idx_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, x_padding_mask)

        if y_padding_mask is None:
            y_padding_mask = torch.ones(Bx, self.cache_size, device=x.device, dtype=torch.bool)

        kv, idx_k, cu_seqlens_k, max_seqlen_k = unpad_input(kv, y_padding_mask)

        out = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=self.dropout,
            softmax_scale=1.,
            causal=self.causal,
        )

        out = pad_input(out, idx_q, Bx, Lx)
        out = rearrange(out, '... h d -> ... (h d)')

        return self.out_proj(out)

class EncoderCachedFlashMHCA(RopeFlashMHA):
    '''Cache the encoder, pass in a full y sequence for likelihood evaluation.'''
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 max_seq_len: int,
                 bias: bool = True,
                 dropout: float = 0.0,
                 layer_idx: Optional[int] = None):
        
        super().__init__(embed_dim = embed_dim,
                         num_heads = num_heads,
                         bias=bias,
                         self_attn=False,
                         causal=False,
                         dropout=dropout,
                         layer_idx=layer_idx)
        
        self.max_seq_len = max_seq_len
        self.kv_cache = None
        self.cache_size = 0

    def init_kv_cache(self, batch_size):
        # Initialize the KV cache with empty tensors
        self.kv_cache = torch.empty(batch_size, self.max_seq_len, 2, self.num_heads, self.head_dim, device=self.q_proj.weight.device)
        self.cache_size = 0

    def forward(self, x, y=None, x_padding_mask=None, y_padding_mask=None):
        Bx, Lx, Dx = x.size()

        if self.kv_cache is None or self.kv_cache.size(0) < Bx:
            self.init_kv_cache(Bx)

        q = self.q_proj(x)
        q *= self.head_dim ** -0.5
        q = rearrange(q, 'b l (n h) -> b l n h', n=self.num_heads)

        if y is not None:
            # Update KV cache
            By, Ly, Dy = y.size()
            k = self.k_proj(y)
            v = self.v_proj(y)
            k = rearrange(k, 'b l (n h) -> b l n h', n=self.num_heads)
            v = rearrange(v, 'b l (n h) -> b l n h', n=self.num_heads)
            
            kv = torch.stack([k, v], dim=2)
            
            self.kv_cache[:, self.cache_size:self.cache_size+Ly] = kv
            self.cache_size += Ly
        else:
            # Use cached KV
            #note that batch size may be smaller than max during final batch
            kv = self.kv_cache[:Bx, :self.cache_size].to(q.dtype)

        q, kv = self.rot_emb(q, kv, seqlen_offset=0, max_seqlen=max(q.shape[1], kv.shape[1]))

        if x_padding_mask is None:
            x_padding_mask = torch.ones(Bx, Lx, device=x.device, dtype=torch.bool)

        q, idx_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, x_padding_mask)

        if y_padding_mask is None:
            y_padding_mask = torch.ones(Bx, self.cache_size, device=x.device, dtype=torch.bool)

        kv, idx_k, cu_seqlens_k, max_seqlen_k = unpad_input(kv, y_padding_mask)

        out = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=self.dropout,
            softmax_scale=1.,
            causal=self.causal,
        )

        out = pad_input(out, idx_q, Bx, Lx)
        out = rearrange(out, '... h d -> ... (h d)')

        return self.out_proj(out)
    
class KVCached_MHSA(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 max_seq_len: int,
                 bias: bool =True,
                 dropout: float =0.0,
                 causal: bool =True, 
                 layer_idx: Optional[int] = None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.causal = causal
        self.dropout = dropout
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.rot_emb = FlashRotaryEmbedding(
            dim=self.head_dim,
            base=ROPE_BASE_FREQUENCY,
            interleaved=False,
            scale_base=None,
            pos_idx_in_fp32=True
        )

        self.kv_cache = None
        self.cache_size = 0

    def init_kv_cache(self, batch_size, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        self.kv_cache = torch.empty(batch_size, seq_len, 2, self.num_heads, self.head_dim, device=self.q_proj.weight.device)
        self.cache_size = 0

    def forward(self, x, x_padding_mask=None):
        batch_size, seq_len, _ = x.shape

        if self.kv_cache is None or self.kv_cache.size(0) != batch_size:
            self.init_kv_cache(batch_size)

        assert self.cache_size + seq_len <= self.kv_cache.size(1), "KV cache is full"

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q *= self.head_dim ** -0.5

        q = rearrange(q, 'b l (n h) -> b l n h', n=self.num_heads)
        k = rearrange(k, 'b l (n h) -> b l n h', n=self.num_heads)
        v = rearrange(v, 'b l (n h) -> b l n h', n=self.num_heads)

        # Apply rotary embeddings - returns k (kv)
        q, k = self.rot_emb(q, torch.stack([k,v], dim=2), seqlen_offset=self.cache_size)

        # Update KV cache
        self.kv_cache[:, self.cache_size:self.cache_size+seq_len] = k

        # Combine current and cached KV
        k = self.kv_cache[:, :self.cache_size+seq_len].to(q.dtype)

        # Update cache size
        self.cache_size += seq_len

        output = flash_attn_kvpacked_func(
            q = q,
            kv = k,
            dropout_p = self.dropout,
            softmax_scale = 1., #q already rescaled
            causal = self.causal,
        )
        output = rearrange(output, 'b l h d -> b l (h d)')

        return self.out_proj(output)

    def reset_kv_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.zero_()
        self.cache_size = 0

class KV_CachedFlashMHADecoderBlock(nn.Module):
    '''Flash Multi-Head Attention Decoder Block for transformer, with Rotary Embedding.
    Will perform both causal self-attention and cross-attention.'''
    def __init__(self,
                 embed_dim: int, 
                 ffn_embed_dim: int,
                 attention_heads: int,
                 use_bias: bool = True,
                 add_bias_kv: bool = False,
                 dropout_p: float = 0.0,
                 max_encoder_seq_len: int = 1024,
                 max_decoder_seq_len: int = 1024,
                 layer_idx: Optional[int] = None,
                 **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.ffn_embed_dim = ffn_embed_dim
        self.add_bias_kv = add_bias_kv
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx
        self.attention_heads = attention_heads
        self.max_encoder_seq_len = max_encoder_seq_len
        self.max_decoder_seq_len = max_decoder_seq_len

        self.self_attn = KVCached_MHSA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            bias=use_bias,
            causal=True,
            layer_idx=layer_idx,
            dropout=dropout_p,
            max_seq_len= max_decoder_seq_len #can decode longer
        )

        self.cross_attn = KVCached_MHCA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            max_seq_len = max_encoder_seq_len,
            layer_idx=layer_idx,
            dropout=dropout_p
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        
    def forward(self, x, y, **kwargs):

        self_attn_kwargs = {'x_padding_mask': kwargs.get('x_padding_mask', None)}

        #causal self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            x=x,
            **self_attn_kwargs
        )
        x = x + residual

        #cross attention
        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attn(
            x=x,
            y=y,
            decoder_cache_size = self.self_attn.cache_size-1, #-1 because it gets updated in the prior step
            **kwargs
        )
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x
    
class EncoderCachedFlashMHADecoderBlock(nn.Module):
    '''Caches the encoder output for likelihood evaluation. Assumes that y is passed in as the full y sequence.
    Can still be used for generation, but will be slower than the fully KV cached version, as you perform a full
    attention pass on the entire y rather than the last token.'''
    def __init__(self,
                 embed_dim, 
                 ffn_embed_dim,
                 attention_heads,
                 use_bias = True,
                 add_bias_kv = False,
                 dropout_p = 0.0, 
                 layer_idx=None,
                 max_seq_len: int = 1024,
                 **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.ffn_embed_dim = ffn_embed_dim
        self.add_bias_kv = add_bias_kv
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx
        self.attention_heads = attention_heads
        self.max_seq_len = max_seq_len

        self.self_attn = RopeFlashMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            self_attn=True,
            causal=True,
            layer_idx=layer_idx,
            dropout=dropout_p
        )

        self.cross_attn = EncoderCachedFlashMHCA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            max_seq_len = max_seq_len,
            layer_idx=layer_idx,
            dropout=dropout_p
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        
    def forward(self, x, y, **kwargs):

        self_attn_kwargs = {'x_padding_mask': kwargs.get('x_padding_mask', None)}

        #causal self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            x=x,
            **self_attn_kwargs
        )
        x = x + residual

        #cross attention
        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attn(
            x=x,
            y=y,
            **kwargs
        )
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x

##############################################################
## Non-Flash Versions of the MHA and Encoder/Decoder Blocks ##
##############################################################

class TorchRotaryEmbedding(FlashRotaryEmbedding):
    def __init__(self,
            dim: int,
            base: float = ROPE_BASE_FREQUENCY,
            interleaved: bool = False,
            scale_base=None,
            pos_idx_in_fp32: bool = True
    ):
        super(TorchRotaryEmbedding, self).__init__(
            dim,
            base=base,
            interleaved=interleaved,
            scale_base=scale_base,
            pos_idx_in_fp32=pos_idx_in_fp32)

    def forward(
        self,
        q,
        k,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) if kv is none,
             else it's just q of shape (batch, seqlen, nheads, headdim)
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        seqlen_q = q.shape[1]
        seqlen_k = k.shape[1]
        self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        q = apply_rotary_emb_torch(
            q,
            self._cos_cached[:seqlen_q,:],
            self._sin_cached[:seqlen_q,:],
        )
        #self._update_cos_sin_cache(seqlen_k, device=k.device, dtype=k.dtype)
        k = apply_rotary_emb_torch(
            k,
            self._cos_cached[:seqlen_k,:],
            self._sin_cached[:seqlen_k,:],
            interleaved = False
        )
        return q, k

class VanillaMHA(nn.Module):
    '''Vanilla Attention module for transformer using RoPe
    
    Args:
    head_dim: int - the dimension of the attention head
    num_heads: int - the number of attention heads
    causal: bool - whether to use causal attention (in decoder self-attention)
    '''
    def __init__(self, embed_dim, num_heads, bias = True, add_bias_kv = False, dropout = 0.0, self_attn=True, causal: bool = False, layer_idx= None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx
        self.dropout = dropout
        self.causal = causal
        self.self_attn = self_attn

        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.rot_emb = TorchRotaryEmbedding(
            dim=self.head_dim,
            base=ROPE_BASE_FREQUENCY,
            interleaved=False,
            scale_base=None,
            pos_idx_in_fp32=True  # Important for bf16 to avoid rounding errors
        )

    def forward(self,
                x,
                y = None,
                attention_mask = None,
                ):

        if self.self_attn:
            #project x to q, k, v
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            #y comes from encoder, provides keys and values
            assert y is not None, "Cross attention requires y input"
            q = self.q_proj(x)
            k = self.k_proj(y)
            v = self.v_proj(y)

        #rescale q 
        q *= self.head_dim ** -0.5

        q = rearrange(q, 'b l (n h) -> b l n h', n=self.num_heads, h = self.head_dim)
        k = rearrange(k, 'b l (n h) -> b l n h', n=self.num_heads, h = self.head_dim)
        v = rearrange(v, 'b l (n h) -> b l n h', n=self.num_heads, h = self.head_dim)

        #rotary here
        q, k = self.rot_emb(q, k, max_seqlen = max(q.shape[1], k.shape[1]))

        q = rearrange(q, 'b l n h -> b n l h')
        k = rearrange(k, 'b l n h -> b n l h')
        v = rearrange(v, 'b l n h -> b n l h')
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) #q already scaled

        if attention_mask is not None:
            #attention mask should be positions to mask
            dots = dots.masked_fill(attention_mask[:, None, None, :], float('-inf'))
        
        if self.causal:
            i,j = dots.shape[-2:]
            mask = torch.ones(i,j, device=dots.device, dtype=bool).triu(j-i + 1)
            dots = dots.masked_fill(mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b n l h -> b l (n h)', n=self.num_heads, h=self.head_dim)

        return self.out_proj(out), attn

class ESMEncoderBlock(nn.Module):
    '''Vanilla Multi-Head Attention Encoder Block for transformer, compatible with ESM2.
    Notably, this module is identical to the FlashMHAEncoderBlock, but does not use FLASH
    Positional encoding is handled using Rotary Embeddings.
    '''
    def __init__(self, embed_dim, attention_heads, ffn_embed_dim, use_bias = True, dropout_p = 0.0, layer_idx = None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads

        self.self_attn = VanillaMHA(
            embed_dim = embed_dim,
            num_heads = attention_heads,
            bias = use_bias,
            dropout = dropout_p,
            self_attn = True,
            layer_idx = layer_idx
        )
        
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, attn_mask):
        #atten block
        residual = x
        x = self.self_attn_layer_norm(x)
        x, att = self.self_attn(x, attention_mask = attn_mask)
        x = x + residual
        #MLP block
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x, att
    
class ESMDecoderBlock(nn.Module):
    '''Vanilla Multi-Head Attention Decoder Block for transformer, compatible with ESM2.'''
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, use_bias = True, add_bias_kv = False, dropout_p = 0.0, layer_idx= None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = attention_heads
        self.head_dim = embed_dim // attention_heads
        self.layer_idx = layer_idx
        self.dropout = dropout_p


        self.self_attn = VanillaMHA(
            embed_dim = embed_dim,
            num_heads = attention_heads,
            bias = use_bias,
            dropout = dropout_p,
            self_attn = True,
            causal = True,
            layer_idx = layer_idx
        )

        self.cross_attn = VanillaMHA(
            embed_dim = embed_dim,
            num_heads = attention_heads,
            bias = use_bias,
            dropout = dropout_p,
            self_attn = False,
            layer_idx = layer_idx,
            causal = False
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, y, x_attn_mask=None, y_attn_mask=None):
        #self attend block
        residual = x
        x = self.self_attn_layer_norm(x)
        x, self_att = self.self_attn(x, x, x_attn_mask) #here the x is the decoding sequence (y)
        x = x + residual
        #cross attend block
        residual = x
        x = self.cross_attn_layer_norm(x)
        x, cross_att = self.cross_attn(x, y, y_attn_mask)
        x = x + residual
        #MLP block
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x, (self_att, cross_att)