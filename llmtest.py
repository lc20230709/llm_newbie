import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import mlflow
import optuna

# Step 1: Toy Dataset

input_data = "./input/"

sentences = []
for filename in os.listdir(input_data):
    file_path = os.path.join(input_data,filename)
    with open(file_path,"r") as file:
        content = file.read()
        sentences += content.split("\n")

# Build vocabulary
words = sorted(set(" ".join(sentences).split()))
vocab_size = len(words)
word_to_idx = {word: i for i, word in enumerate(words)}
idx_to_word = {i: word for i, word in enumerate(words)}

# Convert sentences to token indices
data = [[word_to_idx[word] for word in sentence.split()] for sentence in sentences]


# Step 3: Model Definition
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([self.TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    class TransformerBlock(nn.Module):
        def __init__(self, n_embd, n_head, dropout):
            super().__init__()
            self.sa = nn.MultiheadAttention(n_embd, n_head, dropout=dropout)
            self.ffwd = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x.transpose(0, 1)
            attn_output, _ = self.sa(x, x, x)
            x = x + attn_output
            x = self.ln1(x.transpose(0, 1))
            ffwd_out = self.ffwd(x)
            x = self.ln2(x + ffwd_out)
            return x

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(t, device=device))
        x = tok_emb + pos_emb
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss



# Step 4: Data Preparation
def get_batch(data, batch_size, block_size):
    batch = []
    targets = []
    for _ in range(batch_size):
        sentence = data[torch.randint(len(data), (1,)).item()]
        # Ensure sentence is long enough
        if len(sentence) < block_size + 1:
            sentence = sentence + [0] * (block_size + 1 - len(sentence))  # Pad with 0s
        start_idx = torch.randint(0, len(sentence) - block_size, (1,)).item()
        seq = sentence[start_idx:start_idx + block_size]
        target = sentence[start_idx + 1:start_idx + block_size + 1]
        batch.append(seq)
        targets.append(target)
    return torch.tensor(batch), torch.tensor(targets)

def train_model(trial):
    n_head = trial.suggest_int("n_head", 1, 4)
    base_embed = trial.suggest_int("base_embed", 16, 32)
    n_embd = base_embed * n_head
    n_layer = trial.suggest_int("n_layer", 1, 3)
    block_size = trial.suggest_int("block_size", 3, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 1, 4)


    with mlflow.start_run():
        mlflow.log_param("n_embd", n_embd)
        mlflow.log_param("n_head", n_head)
        mlflow.log_param("n_layer", n_layer)
        mlflow.log_param("block_size", block_size)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("batch_size", batch_size)
        model = SimpleLLM(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        for epoch in range(50):
            model.train()
            X, y = get_batch(data, batch_size, block_size)
            logits, loss = model(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mlflow.log_metric("final_loss", loss.item())

        return loss.item()

study = optuna.create_study(direction="minimize")
study.optimize(train_model, n_trials=10)

# 输出最佳参数
print("best parameters:", study.best_params)
print("best loss:", study.best_value)