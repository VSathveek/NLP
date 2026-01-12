import re
import random
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


with open("enwik9", "r", encoding="utf-8") as f:
    raw_text = f.read()

sentences = [
    clean_text(s).split()
    for s in raw_text.split("\n")
    if len(s.split()) > 5
]


MIN_COUNT = 5
WINDOW_SIZE = 5
EMBEDDING_DIM = 300
NEG_SAMPLES = 5
EPOCHS = 3
BATCH_SIZE = 512


word_counts = Counter(word for sent in sentences for word in sent)
vocab = {w: i for i, (w, c) in enumerate(word_counts.items()) if c >= MIN_COUNT}
id2word = {i: w for w, i in vocab.items()}
vocab_size = len(vocab)


counts = np.array([word_counts[id2word[i]] for i in range(vocab_size)])
unigram_dist = counts ** 0.75
unigram_dist /= unigram_dist.sum()


def generate_pairs(sentences):
    pairs = []
    for sent in sentences:
        sent = [vocab[w] for w in sent if w in vocab]
        for i, target in enumerate(sent):
            context_window = (
                sent[max(0, i - WINDOW_SIZE): i]
                + sent[i + 1: i + WINDOW_SIZE + 1]
            )
            for context in context_window:
                pairs.append((target, context))
    return pairs


pairs = generate_pairs(sentences)


class SGNS(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)
        self._init_embeddings()

    def _init_embeddings(self):
        initrange = 0.5 / EMBEDDING_DIM
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.zero_()

    def forward(self, target, context, negative):
        v = self.in_embed(target)
        u = self.out_embed(context)
        neg_u = self.out_embed(negative)

        pos_score = torch.sum(v * u, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        neg_score = torch.bmm(neg_u, v.unsqueeze(2)).squeeze()
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1)

        return -(pos_loss + neg_loss).mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SGNS(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


for _ in range(EPOCHS):
    random.shuffle(pairs)

    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i + BATCH_SIZE]
        if not batch:
            continue

        targets = torch.tensor([t for t, _ in batch]).to(device)
        contexts = torch.tensor([c for _, c in batch]).to(device)

        negatives = np.random.choice(
            vocab_size,
            size=(len(batch), NEG_SAMPLES),
            p=unigram_dist
        )
        negatives = torch.tensor(negatives).to(device)

        optimizer.zero_grad()
        loss = model(targets, contexts, negatives)
        loss.backward()
        optimizer.step()


embeddings = model.in_embed.weight.detach().cpu().numpy()

with open("sgns_embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, vocab, id2word), f)


gensim_model = Word2Vec(
    sentences=sentences,
    vector_size=EMBEDDING_DIM,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    negative=NEG_SAMPLES,
    workers=4
)


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


for word in ["king", "queen", "man", "woman"]:
    if word in vocab and word in gensim_model.wv:
        sgns_vec = embeddings[vocab[word]]
        gensim_vec = gensim_model.wv[word]
        print(word, cosine(sgns_vec, gensim_vec))


def analogy(a, b, c):
    if a not in vocab or b not in vocab or c not in vocab:
        return None

    vec = embeddings[vocab[b]] - embeddings[vocab[a]] + embeddings[vocab[c]]
    sims = cosine_similarity([vec], embeddings)[0]
    best = np.argsort(-sims)

    for idx in best:
        word = id2word[idx]
        if word not in {a, b, c}:
            return word


print("SGNS:", analogy("man", "king", "woman"))
print(
    "Gensim:",
    gensim_model.wv.most_similar(
        positive=["king", "woman"],
        negative=["man"]
    )[0]
)


def bias_score(target, a, b):
    if target not in vocab or a not in vocab or b not in vocab:
        return None
    return cosine(
        embeddings[vocab[target]],
        embeddings[vocab[a]] - embeddings[vocab[b]]
    )


gender_pairs = [("he", "she"), ("man", "woman"), ("father", "mother")]
profession_words = ["doctor", "nurse", "engineer", "teacher", "programmer"]

for word in profession_words:
    scores = [bias_score(word, a, b) for a, b in gender_pairs]
    print(word, scores)
