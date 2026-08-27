import re


def _tokens(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in text.split() if len(t) > 1]


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were",
    "be", "been", "by", "at", "as", "it", "its", "that", "this", "these", "those", "from", "not",
    "no", "but", "if", "then", "than", "so", "such", "any", "all", "can", "may", "shall", "will",
    "would", "could", "should", "has", "have", "had", "do", "does", "did", "under", "into", "also",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        self.tf: list[dict] = []
        df: dict = {}
        for d in docs:
            counts: dict = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        import math
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}

    def score(self, query: list[str], index: int) -> float:
        s = 0.0
        tf = self.tf[index]
        dl = self.doc_len[index]
        for t in query:
            if t not in tf:
                continue
            idf = self.idf.get(t, 0.0)
            f = tf[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            s += idf * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query_tokens: list[str], top_k: int = 10):
        scored = [(i, self.score(query_tokens, i)) for i in range(self.N)]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


class TfidfIndex:
    def __init__(self, docs: list[list[str]]):
        import math
        self.docs = [self._tf(d) for d in docs]
        self.N = len(docs)
        df: dict = {}
        for d in self.docs:
            for t in d:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(self.N / n) + 1.0 for t, n in df.items()}
        self.vecs = [self._norm_vec(tf) for tf in self.docs]

    @staticmethod
    def _tf(tokens: list[str]) -> dict:
        tf: dict = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return tf

    def _norm_vec(self, tf: dict) -> dict:
        v = {t: (f * self.idf.get(t, 1.0)) ** 0.5 for t, f in tf.items() if t in self.idf}
        norm = sum(w * w for w in v.values()) ** 0.5
        if norm > 0:
            v = {t: w / norm for t, w in v.items()}
        return v

    def embed_query(self, tokens: list[str]) -> dict:
        return self._norm_vec({t: 1 for t in tokens})

    @staticmethod
    def cosine(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(w * b.get(t, 0.0) for t, w in a.items())

    def search(self, query_tokens: list[str], top_k: int = 10):
        qv = self.embed_query(query_tokens)
        scored = [(i, TfidfIndex.cosine(qv, vec)) for i, vec in enumerate(self.vecs)]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
