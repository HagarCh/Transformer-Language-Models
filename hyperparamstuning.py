import optuna
import torch
from transformer import TransformerLM
import data
import lm
from torch import optim
import pandas as pd

def objective(trial):
    # Sample hyperparameters from broader, realistic ranges
    n_layers = trial.suggest_int("n_layers", 2, 8)
    n_heads = trial.suggest_int("n_heads", 2, 8)
    head_dim = trial.suggest_categorical("head_dim", [32, 48, 64, 96])
    embed_size = n_heads * head_dim
    mlp_hidden_size = embed_size * 4
    learning_rate = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    seq_len = 128
    batch_size = 64
    num_batches_to_train = 5000
    data_path = "code-and-data/heb-data/"
    
    # Load data
    tokenizer, tokenized_data = data.load_data(data_path)
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    # Initialize model
    model = TransformerLM(
        n_layers=n_layers,
        n_heads=n_heads,
        embed_size=embed_size,
        max_context_len=seq_len,
        vocab_size=tokenizer.vocab_size(),
        mlp_hidden_size=mlp_hidden_size,
        with_residuals=True,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay)

    total_loss = 0
    for i, batch in enumerate(data.batch_items(data_iter, batch_size)):
        if i >= num_batches_to_train:
            break
        batch_x, batch_y = lm.batch_to_labeled_samples(batch)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = lm.compute_loss(logits, batch_y)

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches_to_train
    return avg_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1000))
    study.optimize(objective, n_trials=30)

    for i, trial in enumerate(study.trials):
        print(f"Trial {i}:")
        print(f"  Value (loss): {trial.value:.4f}")
        print(f"  Params: {trial.params}")

    df = study.trials_dataframe()
    df.to_csv("optuna_trials.csv", index=False)

    print("Best trial:")
    print(study.best_trial)