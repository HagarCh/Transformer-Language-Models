from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals

    def forward(self, inputs):
        if self.with_residuals:
            x = inputs
            x = self.layer_norm_1(x)
            x, attn_weights = self.causal_attention(x)
            x1 = x + inputs
            x = self.layer_norm_2(x1)
            x = self.mlp(x) + x1
            return x, attn_weights

        else:
            x = inputs
            x = self.layer_norm_1(x)
            x, attn_weights = self.causal_attention(x)
            x = self.layer_norm_2(x)
            x = self.mlp(x)
            return x, attn_weights

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        B, N = x.size()
        tok_embeddings = self.token_embeddings(x)
        positions = torch.arange(N, device=x.device).unsqueeze(0)
        pos_embeddings = self.position_embeddings(positions)
        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs, return_attn=False):
        x = self.embed(inputs)
        all_attn_weights = [] if return_attn else None

        for layer in self.layers:
            x, attn_weights = layer(x)
            if return_attn:
                all_attn_weights.append(attn_weights)

        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        if return_attn:
            return logits, all_attn_weights

        return logits

    def init_weights(self):
        # initialize weights
        for pn, p in self.named_parameters():
            if isinstance(p, nn.LayerNorm):
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.ones_(p.weight)
            elif isinstance(p, nn.Linear):
                torch.nn.init.zeros_(p.bias)
                nn.init.kaiming_uniform_(p.weight, nonlinearity='relu')
                # torch.nn.init.xavier_uniform_(p.weight)
            elif isinstance(p, nn.Embedding):
                torch.nn.init.zeros_(p.bias)
                # torch.nn.init.xavier_uniform_(p.weight)
                nn.init.kaiming_uniform_(p.weight, nonlinearity='relu')



    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                # logits = self(torch.tensor([feed_to_lm], dtype=torch.int32))
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32).to(next(self.parameters()).device))
                logits_for_last_token = logits[0][-1]
                # distribution_for_last_token = F.softmax(logits_for_last_token)
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
            feed_to_lm = prefix[:]
            generated = []
            with torch.no_grad():
                while len(generated) < max_tokens_to_generate:
                    if len(feed_to_lm) > self.max_context_len:
                        feed_to_lm = feed_to_lm[-self.max_context_len:]
                    device = next(self.parameters()).device
                    logits = self(torch.tensor([feed_to_lm], dtype=torch.int32, device=device))
                    logits_for_last_token = logits[0][-1]
                    # apply temperature
                    logits_for_last_token = logits_for_last_token / temperature

                    # top-k filtering
                    topk_values, topk_indices = torch.topk(logits_for_last_token, topK)
                    topk_probs = F.softmax(topk_values, dim=-1)
                    sampled_index = torch.multinomial(topk_probs, num_samples=1)
                    sampled_token = topk_indices[sampled_index]

                    generated.append(sampled_token)
                    feed_to_lm.append(sampled_token)
            return generated

