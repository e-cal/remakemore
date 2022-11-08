# %%
import torch
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
P = N / N.sum(dim=1, keepdim=True)
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

# %%
