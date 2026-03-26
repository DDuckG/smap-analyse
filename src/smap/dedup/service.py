from __future__ import annotations
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from smap.dedup.models import DedupAnalysisResult, DedupClusterRecord
from smap.normalization.models import MentionRecord

@dataclass
class _UnionFind:
    parent: dict[str, str]

    def find(self, item):
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

class DeduplicationService:

    def __init__(self, *, exact_enabled=True, near_enabled=True, min_text_length=10, word_shingle_size=3, char_shingle_size=5, num_perm=64, num_bands=16, near_similarity_threshold=0.82, max_bucket_size=256):
        self.exact_enabled = exact_enabled
        self.near_enabled = near_enabled
        self.min_text_length = min_text_length
        self.word_shingle_size = word_shingle_size
        self.char_shingle_size = char_shingle_size
        self.num_perm = max(num_perm, 8)
        self.num_bands = max(1, min(num_bands, self.num_perm))
        self.rows_per_band = max(1, self.num_perm // self.num_bands)
        self.near_similarity_threshold = near_similarity_threshold
        self.max_bucket_size = max_bucket_size

    def annotate(self, mentions):
        if not mentions:
            return ([], DedupAnalysisResult())
        exact_clusters = self._exact_clusters(mentions) if self.exact_enabled else []
        exact_members = {mention_id for cluster in exact_clusters for mention_id in cluster.mention_ids}
        near_clusters = self._near_clusters(mentions, exact_members=exact_members) if self.near_enabled else []
        annotations = self._annotation_map([*exact_clusters, *near_clusters])
        updated_mentions = [mention.model_copy(update=annotations.get(mention.mention_id, {})) for mention in mentions]
        return (updated_mentions, DedupAnalysisResult(mentions_updated=sum((1 for update in annotations.values() if update)), exact_cluster_count=sum((1 for cluster in exact_clusters if cluster.cluster_size > 1)), near_cluster_count=sum((1 for cluster in near_clusters if cluster.cluster_size > 1)), clusters=[*exact_clusters, *near_clusters]))

    def _annotation_map(self, clusters):
        updates: dict[str, dict[str, object]] = {}
        for cluster in clusters:
            weight = round(1.0 / max(cluster.cluster_size, 1), 6)
            for mention_id in cluster.mention_ids:
                updates[mention_id] = {'dedup_cluster_id': cluster.dedup_cluster_id, 'dedup_kind': cluster.dedup_kind, 'dedup_representative_mention_id': cluster.representative_mention_id, 'dedup_cluster_size': cluster.cluster_size, 'dedup_similarity': cluster.similarity_proxy, 'dedup_weight': weight}
        return updates

    def _exact_clusters(self, mentions):
        groups: dict[str, list[MentionRecord]] = defaultdict(list)
        for mention in mentions:
            fingerprint = self._fingerprint(mention.normalized_text_compact)
            if fingerprint is None:
                continue
            groups[fingerprint].append(mention)
        clusters: list[DedupClusterRecord] = []
        for fingerprint, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            representative = self._representative(members)
            clusters.append(DedupClusterRecord(dedup_cluster_id=f'exact:{fingerprint[:16]}', dedup_kind='exact', representative_mention_id=representative.mention_id, representative_text=representative.raw_text, mention_ids=[member.mention_id for member in members], cluster_size=len(members), similarity_proxy=1.0, text_fingerprint=fingerprint))
        return clusters

    def _near_clusters(self, mentions, *, exact_members):
        candidates = [mention for mention in mentions if mention.mention_id not in exact_members and len(mention.normalized_text_compact) >= self.min_text_length]
        if len(candidates) < 2:
            return []
        shingles = {mention.mention_id: self._shingles(mention.normalized_text_compact) for mention in candidates}
        signatures = {mention_id: self._minhash_signature(items) for mention_id, items in shingles.items() if items}
        if len(signatures) < 2:
            return []
        buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
        for mention_id, signature in signatures.items():
            for band_index in range(self.num_bands):
                start = band_index * self.rows_per_band
                end = start + self.rows_per_band
                band = signature[start:end]
                if not band:
                    continue
                buckets[band_index, band].append(mention_id)
        union_find = _UnionFind(parent={mention_id: mention_id for mention_id in signatures})
        for bucket in buckets.values():
            if len(bucket) < 2 or len(bucket) > self.max_bucket_size:
                continue
            for left_index in range(len(bucket)):
                for right_index in range(left_index + 1, len(bucket)):
                    left_id = bucket[left_index]
                    right_id = bucket[right_index]
                    similarity = self._jaccard(shingles[left_id], shingles[right_id])
                    if similarity >= self.near_similarity_threshold:
                        union_find.union(left_id, right_id)
        if len(signatures) <= 64:
            signature_ids = list(signatures)
            for left_index in range(len(signature_ids)):
                for right_index in range(left_index + 1, len(signature_ids)):
                    left_id = signature_ids[left_index]
                    right_id = signature_ids[right_index]
                    similarity = self._jaccard(shingles[left_id], shingles[right_id])
                    if similarity >= self.near_similarity_threshold:
                        union_find.union(left_id, right_id)
        grouped: dict[str, list[MentionRecord]] = defaultdict(list)
        by_id = {mention.mention_id: mention for mention in candidates}
        for mention_id in signatures:
            grouped[union_find.find(mention_id)].append(by_id[mention_id])
        clusters: list[DedupClusterRecord] = []
        for group_members in grouped.values():
            if len(group_members) < 2:
                continue
            representative = self._representative(group_members)
            similarity = self._cluster_similarity(group_members, shingles)
            cluster_hash = self._cluster_hash(group_members)
            clusters.append(DedupClusterRecord(dedup_cluster_id=f'near:{cluster_hash[:16]}', dedup_kind='near', representative_mention_id=representative.mention_id, representative_text=representative.raw_text, mention_ids=sorted((member.mention_id for member in group_members)), cluster_size=len(group_members), similarity_proxy=similarity, text_fingerprint=cluster_hash))
        return sorted(clusters, key=lambda item: (-item.cluster_size, item.dedup_cluster_id))

    def _fingerprint(self, text):
        compact = ' '.join(text.split())
        if len(compact) < self.min_text_length:
            return None
        return hashlib.blake2b(compact.encode('utf-8'), digest_size=20).hexdigest()

    def _shingles(self, text):
        tokens = [token for token in text.split() if token]
        if len(tokens) >= self.word_shingle_size:
            return {' '.join(tokens[index:index + self.word_shingle_size]) for index in range(len(tokens) - self.word_shingle_size + 1)}
        compact = text.replace(' ', '')
        if len(compact) >= self.char_shingle_size:
            return {compact[index:index + self.char_shingle_size] for index in range(len(compact) - self.char_shingle_size + 1)}
        return {text} if text else set()

    def _minhash_signature(self, shingles):
        hashed = [int(hashlib.blake2b(shingle.encode('utf-8'), digest_size=8).hexdigest(), 16) for shingle in shingles]
        if not hashed:
            return ()
        signature: list[int] = []
        for seed in range(self.num_perm):
            signature.append(min(((value ^ (seed + 1) * 11400714819323198485) & 18446744073709551615 for value in hashed)))
        return tuple(signature)

    def _jaccard(self, left, right):
        if not left and (not right):
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return round(len(left & right) / len(union), 4)

    def _representative(self, mentions):
        return sorted(mentions, key=lambda mention: (-len(mention.normalized_text_compact), mention.depth, mention.mention_id))[0]

    def _cluster_similarity(self, mentions, shingles):
        if len(mentions) < 2:
            return 1.0
        total = 0.0
        pair_count = 0
        for left_index in range(len(mentions)):
            for right_index in range(left_index + 1, len(mentions)):
                total += self._jaccard(shingles[mentions[left_index].mention_id], shingles[mentions[right_index].mention_id])
                pair_count += 1
        if pair_count <= 0:
            return 1.0
        return round(total / pair_count, 4)

    def _cluster_hash(self, mentions):
        payload = '|'.join(sorted((mention.normalized_text_compact for mention in mentions)))
        return hashlib.blake2b(payload.encode('utf-8'), digest_size=20).hexdigest()
