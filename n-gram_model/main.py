from collections import defaultdict
import requests
import re

N = 5

ngram_counts = defaultdict(int)
context_counts = defaultdict(int)
vocab = set()


def vocabulary(sentences):
    for sentence in sentences:
        words = sentence.split()
        if len(words) < N:
            continue
        for i in range(len(words) - N + 1):
            ngram = tuple(words[i:i + N])
            context = tuple(words[i:i + N - 1])
            ngram_counts[ngram] += 1
            context_counts[context] += 1
            vocab.add(words[i + N - 1])


def compute(context, word):
    ngram = context + (word,)
    if ngram_counts[ngram] == 0:
        return 0
    return ngram_counts[ngram] / context_counts[context]


def compute_prob(context):
    max_prob = 0
    next_word = ""
    for word in vocab:
        prob = compute(context, word)
        if prob > max_prob:
            max_prob = prob
            next_word = word
    return next_word


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s.!?]', '', text)
    return text


def prepare_sentences(text):
    sentences = re.split(r'[.!?]', text)
    return [s.strip() for s in sentences if s.strip()]


def generate(text, limit=25):
    words = text.lower().split()
    output = words[:]
    while len(output) < limit + len(words):
        context = tuple(output[-(N - 1):])
        word = compute_prob(context)
        if not word:
            break
        output.append(word)
    return " ".join(output)


url = "https://www.gutenberg.org/files/1342/1342-0.txt"
raw = requests.get(url).text
raw = raw.split("*** START OF")[1].split("*** END OF")[0]

clean = clean_text(raw)
sentences = prepare_sentences(clean)
vocabulary(sentences)


samples = [
    "The day was very",
    "She could not help",
    "It was impossible to"
]

for s in samples:
    print("\n\n\n")
    print(generate(s))
