import torch
import torch.nn as nn
from torch.nn import functional as F
import os


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

# Step 2: Hyperparameters
n_embd = 32  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 2  # Number of transformer layers
block_size = 5  # Reduced to fit short sentences (max words per sentence is 6)
dropout = 0.2  # Dropout rate
batch_size = 2  # Small batch size


# Step 3: Model Definition
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([self.TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])

        # Output layer
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
            # Self-attention (transpose for multihead attention: [seq_len, batch, embd])
            x = x.transpose(0, 1)
            attn_output, _ = self.sa(x, x, x)
            x = x + attn_output
            x = self.ln1(x.transpose(0, 1))  # Back to [batch, seq_len, embd]
            # Feedforward
            ffwd_out = self.ffwd(x)
            x = self.ln2(x + ffwd_out)
            return x

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # Embed tokens and positions
        tok_emb = self.token_embedding(idx)  # (batch, time, embedding)
        pos_emb = self.position_embedding(torch.arange(t, device=device))  # (time, embedding)
        x = tok_emb + pos_emb

        # Apply transformer blocks
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        # Final output
        x = self.ln_f(x)
        logits = self.head(x)

        # Compute loss if targets are provided
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


# Step 5: Training
model = SimpleLLM(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
epochs = 100

for epoch in range(epochs):
    model.train()
    X, y = get_batch(data, batch_size, block_size)
    logits, loss = model(X, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Step 6: Generate Text
def generate(model, start_idx, max_new_tokens):
    model.eval()
    idx = torch.tensor([start_idx], dtype=torch.long)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx
        # Pad if shorter than block_size
        if idx_cond.size(1) < block_size:
            padding = torch.zeros((1, block_size - idx_cond.size(1)), dtype=torch.long)
            idx_cond = torch.cat((padding, idx_cond), dim=1)
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # Last token's logits
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)

    return [idx_to_word[i.item()] for i in idx[0]]


# Test generation
start_idx = [word_to_idx["the"]]
generated = generate(model, start_idx, max_new_tokens=5)
print("Generated text:", " ".join(generated))