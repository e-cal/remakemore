# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# %matplotlib inline

# %% [markdown]
# # Data

# %% [markdown]
# ### Load Data

# %%
words = open("names.txt", "r").read().splitlines()

# %%
words[:10]

# %%
print(f"num words: {len(words)}")
print(f"shortest: {min(len(w) for w in words)}")
print(f"longest: {max(len(w) for w in words)}")

# %% [markdown]
# ### Char Maps

# %%
chars = [chr(i) for i in range(97, 97 + 25 + 1)]  # 'a' ... 'z'
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0  # start/end indicator
itos = {i: s for s, i in stoi.items()}

# %% [markdown]
# # Models

# %% [markdown]
# ## Bigrams
# Ordered letter pairs


# %% [markdown]
# ### Count Bigrams

# %%
N = torch.zeros((27, 27), dtype=torch.int32)

for word in words:
    chrs = ["."] + list(word) + ["."]
    for c1, c2 in zip(chrs, chrs[1:]):
        # bigram = (c1, c2)
        N[stoi[c1], stoi[c2]] += 1

# %% [markdown]
# ### Visualization

# %%
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="YlOrRd")
plt.axis("off")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")


# %% [markdown]
# ### Probability Distribution

# %%
# convert each row into its sum (for each row, sum of columns [dim 1])
# keepdim keeps it 2D rather than squeezing out to 1D
s = N.sum(dim=1, keepdim=True)
s, s.shape

# %%
P = (N + 1).float()  # +1 is model smoothing
P /= P.sum(dim=1, keepdim=True)
P[0].sum()

# %%
def gen_name(num=1, seed=2147483647):
    """bigram-based, weighted sampling name generator"""
    g = torch.Generator().manual_seed(seed)
    names = []

    for _ in range(num):
        name = ""
        ix = 0  # start with start token [row]
        while True:  # run until a name is generated
            p = P[ix]  # probability distribution for the token's bigrams
            # get next token by sampling from prev token's prob distribution
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()
            name += itos[ix]  # type: ignore
            if ix == 0:  # end token
                break
        names.append(name)

    return names


# %%
gen_name(10)


# %% [markdown]
# ### Evaluation

# %% [markdown]
# Goal: maximize likelihood (normalized negative log likelihood) of the data w.r.t. model parameters
#
# > max likelihood \
# > ≡ max log likelihood (log is monotonic) \
# > ≡ min negative log likelihood \
# > ≡ min average log likelihood
# >
# > $log(abc) = log(a) + log(b) + log(c)$

# %%
def nll(word):
    """Get the average negative log likelihood of a word."""
    log_likelihood = 0
    n = 0
    chrs = ["."] + list(word) + ["."]
    for c1, c2 in zip(chrs, chrs[1:]):
        prob = P[stoi[c1], stoi[c2]]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

    return -log_likelihood / (len(chrs) - 1)


# %%
for word in ["andrej", "ethan", "daniel", "asotnehu", "test", "sam", "andrejq"]:
    print(f"nll({word})={nll(word):.4f}")

# %% [markdown]
# ## Bigram Neural Network

# %% [markdown]
# ### Get Training Data

# %%
# get x (first char in bigram) and y (second char)

x, y = [], []

for word in words:
    # for word in words[:1]:
    chrs = ["."] + list(word) + ["."]
    for c1, c2 in zip(chrs, chrs[1:]):
        x.append(stoi[c1])
        y.append(stoi[c2])

x = torch.tensor(x)
y = torch.tensor(y)

# %%
# one hot encoding
xenc = F.one_hot(x, num_classes=27).float()

# %% [markdown]
# ### Neural Network

# %% [markdown]
# #### Broken Down Steps

# %%
# initialize weights
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g)

# %%
# Forward pass (differentiable)

# exponenitiation gets rid of negatives (-> n<1) and increases positives,
# analagous to counts

logits = xenc @ W  # log counts
# softmax activation function
counts = logits.exp()  # counts
out = counts / counts.sum(dim=1, keepdims=True)  # probability

# %%
out[0]


# %% [markdown]
# #### Class

# %%
class BigramNN:
    def __init__(self, seed=2147483647):
        self.g = torch.Generator().manual_seed(seed)

    def init_weights(self):
        self.W = torch.randn((27, 27), generator=self.g, requires_grad=True)

    def softmax(self, logits):
        counts = logits.exp()
        prob = counts / counts.sum(dim=1, keepdims=True)
        return prob

    def forward(self, x):
        logits = x @ self.W
        prob = self.softmax(logits)
        return prob

    def predict(self, x):
        xenc = F.one_hot(x, num_classes=27).float()
        prob = self.forward(xenc)
        return torch.argmax(prob).item()

    def fit(self, x, y, lr=1, epochs=100):
        self.init_weights()
        xenc = F.one_hot(x, num_classes=27).float()

        for _ in range(epochs):
            # forward
            pred = self.forward(xenc)
            loss = self.loss(pred, y)

            # backward
            self.W.grad = None
            loss.backward()
            self.W.data += -lr * self.W.grad  # type: ignore

        pred = self.forward(xenc)
        loss = self.loss(pred, y)
        print(f"Final loss: {loss.item():.4f}")

    def loss(self, probs, y):
        likelihood = probs[torch.arange(y.nelement()), y]
        loss = -likelihood.log().mean()  # nll
        return loss


# %%
model = BigramNN()

model.fit(x, y, lr=50, epochs=200)
