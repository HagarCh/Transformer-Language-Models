from __future__ import annotations
import torch
import os
import datetime
from torch import nn
from torch import optim
from transformer import TransformerLM
import data
import lm
import matplotlib.pyplot as plt
import seaborn as sns


def create_attention_maps(model, tokenizer, device, sentence, output_dir="attention_maps"):

    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    tokens = tokenizer.tokenize(sentence)
    token_strs = tokenizer.detokenize(tokens)
    input_tensor = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        logits, attn_weights = model(input_tensor, return_attn=True)

    for layer_idx, layer_attn in enumerate(attn_weights):  # layers
        for head_idx in range(layer_attn.shape[1]):  # heads
            attn = layer_attn[0, head_idx].cpu().numpy()
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn,
                        xticklabels=token_strs,
                        yticklabels=token_strs,
                        cmap="Blues",
                        cbar_kws={'label': 'Attention Weight'})

            plt.yticks(rotation=0)
            plt.title(f"Layer {layer_idx+1}, Head {head_idx+1}")
            plt.xlabel("Key (attended to)")
            plt.ylabel("Query (attending)")
            plt.tight_layout()
            filename = f"{output_dir}/layer{layer_idx+1}_head{head_idx+1}.png"
            plt.savefig(filename)
            plt.close()

    print(f"Saved all attention maps to: {output_dir}/")

def save_res(model, num_batches, params):
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"transformer_final_{params['n_layers']}L_{params['n_heads']}H_{params['embed_size']}D_{params['batch_size']}B_{num_batches}steps_{timestamp}_Heb.pt"
    save_path = os.path.join(save_dir, model_name)

    # Save only model weights
    torch.save(model.state_dict(), save_path)
    print(f"[Final weights saved to {save_path}]")

def load_model(data_path, params_path):
     # load
    seq_len = 128
    n_layers = 4
    n_heads = 8
    embed_size = 1000
    mlp_hidden_size = embed_size * 4
    data_path = data_path
    tokenizer, tokenized_data = data.load_data(data_path)
    model: torch.nn.Module = TransformerLM(
            n_layers,
            n_heads,
            embed_size,
            seq_len,
            tokenizer.vocab_size(),
            mlp_hidden_size,
            with_residuals = True,
        )
    model.load_state_dict(torch.load(params_path))
    model.eval()  # optional: switch to eval mode

    return model, tokenizer


if __name__ == '__main__':
    seq_len = 128
    batch_size = 128
    lan = "Eng"

    if lan == "Heb":
        data_path = "heb-data/"
    elif lan == "Eng":
        data_path = "data/"

    n_layers = 4
    n_heads = 8
    embed_size = 1000
    mlp_hidden_size = embed_size * 4
    weight_decay = 0.01
    learning_rate = 0.0004
    gradient_clipping = 1.0
    num_batches_to_train = 10000


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    tokenizer, tokenized_data = data.load_data(data_path)
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model: torch.nn.Module = TransformerLM(
            n_layers,
            n_heads,
            embed_size,
            seq_len,
            tokenizer.vocab_size(),
            mlp_hidden_size,
            with_residuals=True,
        )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95], weight_decay=weight_decay)
    model.train()
    num_batches = 0

    for batch in data.batch_items(data_iter, batch_size):
        if num_batches >= num_batches_to_train:
            break
        num_batches += 1

        batch_x, batch_y = lm.batch_to_labeled_samples(batch)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = lm.compute_loss(logits, batch_y)
        # parameters update
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        if num_batches % 10 == 0:
            print(f"Seen {num_batches} batches. Last loss: {loss.item():.4f}")
            if num_batches % 100 == 0:
                for _ in range(1):
                    model.eval()
                    if lan == "Eng":
                        prompt = tokenizer.tokenize("Hello")
                    elif lan == "Heb":
                        prompt = tokenizer.tokenize("שלום")
                    sampled = model.better_sample_continuation(prompt, max_tokens_to_generate=500, temperature=0.5, topK=5)
                    #sampled = model.sample_continuation(prompt, 500)
                    sampled = tokenizer.detokenize(sampled)
                    model.train()
                    print(f"Model sample: '''{sampled}'''")
                print("")


    ######## Interpretability ############
    '''
    save_res(model, num_batches, {
        'n_layers': n_layers,
        'n_heads': n_heads,
        'embed_size': embed_size,
        'batch_size': batch_size,
        })

    create_attention_maps(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sentence="The cat sat on the mat.",
        output_dir="attention_maps/sentence1"
    )
    create_attention_maps(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sentence="She gave him a book.",
        output_dir="attention_maps/sentence2"
    )
    create_attention_maps(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sentence="If it rains, we will stay home.",
        output_dir="attention_maps/sentence3"
    )
    '''


