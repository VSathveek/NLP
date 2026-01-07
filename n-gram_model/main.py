from collections import defaultdict
import requests
import re
import time

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
    sents = re.split(r'[.!?]', text)
    return [s.strip() for s in sents if s.strip()]


def fetch_book(book_id):
    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.text
        except:
            time.sleep(1)
    return ""



book_ids = [
    "1342", 
    "84",   
    "1661", 
    "11",    
    "2701",  
]

all_sentences = []

for bid in book_ids:
    raw = fetch_book(bid)
    if not raw:
        print(f"Failed {bid}")
        continue
    if "*** START OF" in raw and "*** END OF" in raw:
        raw = raw.split("*** START OF")[1].split("*** END OF")[0]
    clean = clean_text(raw)
    sents = prepare_sentences(clean)
    all_sentences.extend(sents)
    print(f"Loaded book {bid}, sentences: {len(sents)}")


vocabulary(all_sentences)
print(f"Total vocab: {len(vocab)}")


seeds = [
    "The day was very",
    "She could not help",
    "It was impossible to",
    "In the beginning",
    "He had never"
]

results = []
for s in seeds:
    words = s.lower().split()
    generated = words[:]
    
    for _ in range(1000):
        if len(generated) < N - 1:
            break
        context = tuple(generated[-(N - 1):])
        nxt = compute_prob(context)
        if not nxt:
            break
        generated.append(nxt)
    out = " ".join(generated)
    results.append(out)


with open("output.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n\n")

print("wrote to output")
