from rank_bm25 import BM25Okapi

# Our documents (each split into words)
docs = [
    "Python is a programming language",
    "Streamlit builds web apps",
    "Python is used for AI and machine learning"
]
tokenized_docs = [doc.lower().split() for doc in docs]

# Build the BM25 search
bm25 = BM25Okapi(tokenized_docs)

# Search
query = "python ai".split()
scores = bm25.get_scores(query)

for doc, score in zip(docs, scores):
    print(round(score, 2), "->", doc)