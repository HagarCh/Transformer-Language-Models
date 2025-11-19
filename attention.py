from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads = 1):
    d_head = input_vector_dim // n_heads
    return nn.Linear(input_vector_dim, d_head * 3)

def kqv(x, linear):
    B, N, D =  x.size()
    kqv = linear(x)
    k, q, v = torch.chunk(kqv, 3, dim=-1)
    return k, q, v

def attention_scores(a, b):
    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2
    
    scale = math.sqrt(D1)
    A = torch.matmul(b, a.transpose(-2,-1)) / scale
    return A

def create_causal_mask(embed_dim, n_heads, max_context_len):
    mask = torch.tril(torch.ones(1, max_context_len, max_context_len, dtype=torch.bool))
    return mask

def self_attention(v, A, mask = None):
    # As usual, the dimensions of v and of sa are (b x n x d).
    A = A.masked_fill(mask == 0, float("-inf"))
    A_softmax = nn.functional.softmax(A, dim=-1)
    sa = torch.matmul(A_softmax, v)
    return sa, A_softmax


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa, att_weights = self_attention(v, att, attention_mask)
    return sa, att_weights

def multi_head_attention_layer(x, kqv_matrices, mask):
    B, N, D = x.size()
    n_heads = len(kqv_matrices)
    sliced_mask = mask[:, :N, :N]
    #sa = torch.cat([self_attention_layer(x, kqv_matrices[i], sliced_mask) for i in range(n_heads)], dim=-1)
    sa_list = []
    attn_list = []
    for i in range(n_heads):
        sa_i, attn_i = self_attention_layer(x, kqv_matrices[i], sliced_mask)
        sa_list.append(sa_i)
        attn_list.append(attn_i)

    sa = torch.cat(sa_list, dim=-1)
    attn_weights = torch.stack(attn_list, dim=1)

    assert sa.size() == x.size()

    return sa, attn_weights


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        sa, attn_weights = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        sa = self.proj(sa)
        return sa, attn_weights
