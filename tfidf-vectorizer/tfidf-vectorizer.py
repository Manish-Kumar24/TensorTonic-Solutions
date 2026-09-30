import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    tokenized = [document.lower().split() for document in documents]
    vocabulary = sorted({token for tokens in tokenized for token in tokens})
    index = {token: position for position, token in enumerate(vocabulary)}
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=float)
    document_frequency = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    for row, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        for token, count in counts.items():
            tf = count / len(tokens)
            idf = math.log(len(documents) / document_frequency[token])
            matrix[row, index[token]] = tf * idf
    return {"tfidf_matrix": matrix, "vocabulary": vocabulary}