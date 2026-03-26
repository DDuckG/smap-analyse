from __future__ import annotations
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from smap.providers.base import EmbeddingProvider, EmbeddingPurpose, ProviderMetadata, ProviderProvenance, SimilarityMatch, TopicArtifact, TopicAssignment, TopicDiscoveryResult, TopicDocument, TopicProvider, ZeroShotClassifierProvider
from smap.providers.cache import EmbeddingCacheStore
TOKEN_RE = re.compile('\\w+', flags=re.UNICODE)
_TOPIC_STOPWORDS = {'and', 'but', 'cho', 'cua', 'hay', 'khong', 'la', 'nhe', 'roi', 'that', 'the', 'this', 'voi'}

def _tokenize(text):
    return [token.lower() for token in TOKEN_RE.findall(text)]

def _tokenize_set(text):
    return set(_tokenize(text))

def _cosine_similarity(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum((a * b for a, b in zip(left, right, strict=True)))
    left_norm = math.sqrt(sum((a * a for a in left)))
    right_norm = math.sqrt(sum((b * b for b in right)))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)

def _normalize_vector(values):
    norm = math.sqrt(sum((value * value for value in values)))
    if norm == 0.0:
        return tuple((0.0 for _ in values))
    return tuple((value / norm for value in values))

class TokenOverlapEmbeddingProvider(EmbeddingProvider):

    def __init__(self, *, cache_store=None, vector_size=64):
        self.version = 'token-overlap-v2'
        self.provenance = ProviderProvenance(provider_kind='embedding', provider_name='token_overlap', provider_version=self.version, model_id='token-overlap', device='cpu')
        self.cache_store = cache_store
        self.vector_size = vector_size
        self._memory_cache: dict[tuple[str, str], tuple[float, ...]] = {}

    def embed_texts(self, texts, *, purpose=EmbeddingPurpose.PASSAGE):
        unique_texts = list(dict.fromkeys(texts))

        def memory_key(value):
            return (purpose.value, value)
        cached_vectors = {text: self._memory_cache[memory_key(text)] for text in unique_texts if memory_key(text) in self._memory_cache}
        missing_from_memory = [text for text in unique_texts if text not in cached_vectors]
        store_hits = self.cache_store.load_many(model_id=self.provenance.model_id, purpose=purpose.value, texts=missing_from_memory) if self.cache_store is not None and missing_from_memory else {}
        for text, vector in store_hits.items():
            cached_vectors[text] = vector
            self._memory_cache[memory_key(text)] = vector
        pending_cache_entries: list[dict[str, object]] = []
        for text in unique_texts:
            cached = cached_vectors.get(text)
            if cached is not None:
                continue
            vector = self._hash_embedding(text, purpose)
            cached_vectors[text] = vector
            self._memory_cache[memory_key(text)] = vector
            if self.cache_store is not None:
                pending_cache_entries.append({'model_id': self.provenance.model_id, 'purpose': purpose.value, 'text': text, 'vector': vector})
        if self.cache_store is not None and pending_cache_entries:
            self.cache_store.store_many(pending_cache_entries)
        return [cached_vectors[text] for text in texts]

    def rank_candidates(self, text, candidates, *, purpose=EmbeddingPurpose.LINKING, top_k=5):
        query_vector = self.embed_texts([text], purpose=purpose)[0]
        candidate_items = list(candidates.items())
        candidate_vectors = self.embed_texts([candidate_text for _, candidate_text in candidate_items], purpose=EmbeddingPurpose.PASSAGE)
        ranked = [SimilarityMatch(candidate_id=candidate_id, score=round(_cosine_similarity(query_vector, candidate_vector), 6), candidate_text=candidate_text) for (candidate_id, candidate_text), candidate_vector in zip(candidate_items, candidate_vectors, strict=True)]
        return sorted(ranked, key=lambda item: (-item.score, item.candidate_id))[:top_k]

    def best_match(self, text, candidates):
        ranked = self.rank_candidates(text, candidates, top_k=1)
        return ranked[0] if ranked else None

    def _hash_embedding(self, text, purpose):
        values = [0.0 for _ in range(self.vector_size)]
        weights = Counter(_tokenize(text))
        purpose_bias = 1 if purpose == EmbeddingPurpose.QUERY else 2
        for token, count in weights.items():
            digest = hashlib.sha256(f'{purpose.value}:{token}'.encode()).digest()
            for index in range(0, len(digest), 2):
                bucket = digest[index] % self.vector_size
                sign = 1.0 if digest[index + 1] % 2 == 0 else -1.0
                values[bucket] += sign * float(count * purpose_bias)
        return _normalize_vector(values)

class KeywordZeroShotProvider(ZeroShotClassifierProvider):
    version = 'keyword-zero-shot-v1'

    def classify(self, text, labels):
        text_tokens = Counter(TOKEN_RE.findall(text.lower()))
        scores: dict[str, float] = {}
        for label in labels:
            label_tokens = TOKEN_RE.findall(label.lower())
            if not label_tokens:
                scores[label] = 0.0
                continue
            hits = sum((1 for token in label_tokens if text_tokens[token] > 0))
            scores[label] = hits / len(label_tokens)
        return scores

@dataclass(slots=True)
class FallbackTopicProvider(TopicProvider):
    version: str = 'fallback-topic-v1'
    provenance: ProviderProvenance = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'provenance', ProviderProvenance(provider_kind='topic', provider_name='fallback_topic', provider_version=self.version, model_id='fallback-topic-rules', device='cpu'))

    def discover(self, documents, *, embeddings=None):
        del embeddings
        assignments: list[TopicAssignment] = []
        docs_by_topic: dict[str, list[TopicDocument]] = defaultdict(list)
        for document in documents:
            topic_key, topic_label, confidence, metadata = self._topic_for_document(document)
            assignments.append(TopicAssignment(document_id=document.document_id, topic_key=topic_key, topic_label=topic_label, confidence=confidence, representative=False, metadata=metadata))
            docs_by_topic[topic_key].append(document)
        artifacts: list[TopicArtifact] = []
        for topic_key, topic_docs in sorted(docs_by_topic.items()):
            top_terms = self._top_terms(topic_docs)
            representative_ids = tuple((document.document_id for document in topic_docs[:3]))
            topic_label = topic_key.replace('_', ' ').title()
            assignment_sources = Counter((self._topic_for_document(document)[3].get('source', 'fallback') for document in topic_docs))
            artifacts.append(TopicArtifact(topic_key=topic_key, topic_label=topic_label, top_terms=tuple(top_terms), representative_document_ids=representative_ids, topic_size=len(topic_docs), provider_provenance=self.provenance, metadata={'mode': 'fallback', 'primary_source': assignment_sources.most_common(1)[0][0] if assignment_sources else 'fallback'}))
        return TopicDiscoveryResult(assignments=assignments, artifacts=artifacts)

    def _topic_for_document(self, document):
        hashtags = [token[1:] for token in document.text.split() if token.startswith('#') and len(token) > 2]
        if hashtags:
            topic_key = hashtags[0].lower().replace('-', '_')
            return (topic_key, topic_key.replace('_', ' ').title(), 0.72, {'source': 'hashtag'})
        tokens = [token for token in _tokenize(document.normalized_text) if len(token) > 3 and token not in _TOPIC_STOPWORDS]
        if tokens:
            topic_key = Counter(tokens).most_common(1)[0][0]
            return (topic_key, topic_key.title(), 0.55, {'source': 'keyword'})
        return ('misc', 'Misc', 0.35, {'source': 'fallback'})

    def _top_terms(self, documents):
        counts = Counter((token for document in documents for token in _tokenize(document.normalized_text) if len(token) > 3 and token not in _TOPIC_STOPWORDS))
        return [term for term, _ in counts.most_common(5)] or ['misc']
