from __future__ import annotations
from collections import defaultdict
from smap.contracts.uap import UAPType
from smap.normalization.models import MentionRecord
from smap.threads.models import MentionContext, ThreadBundle, ThreadEdge, ThreadSummary

def _score_for_ranking(mention):
    if mention.sort_score is not None:
        return mention.sort_score
    if mention.likes is not None:
        return float(mention.likes)
    return 0.0

def _resolve_root_mention(root_id, thread_mentions, mentions_by_id):
    root_mention = mentions_by_id.get(root_id)
    if root_mention is not None:
        return root_mention
    return min(thread_mentions, key=lambda item: (item.depth, 0 if item.parent_id is None or item.parent_id not in mentions_by_id else 1, item.mention_id))

def build_threads(mentions):
    mentions_by_id = {mention.mention_id: mention for mention in mentions}
    by_root = defaultdict(list)
    children_by_parent = defaultdict(list)
    edges = []
    for mention in mentions:
        by_root[mention.root_id].append(mention)
        if mention.parent_id:
            children_by_parent[mention.parent_id].append(mention)
            edges.append(ThreadEdge(root_id=mention.root_id, parent_id=mention.parent_id, child_id=mention.mention_id, depth=mention.depth))
    contexts = []
    summaries = []
    for root_id, thread_mentions in by_root.items():
        sorted_mentions = sorted(thread_mentions, key=lambda item: (item.depth, item.mention_id))
        root_mention = _resolve_root_mention(root_id, thread_mentions, mentions_by_id)
        comments = [item for item in thread_mentions if item.uap_type is UAPType.COMMENT]
        replies = [item for item in thread_mentions if item.uap_type is UAPType.REPLY]
        ranked_comments = sorted(comments, key=_score_for_ranking, reverse=True)
        summaries.append(ThreadSummary(root_id=root_id, total_mentions=len(thread_mentions), total_descendants=max(len(thread_mentions) - 1, 0), max_depth_observed=max((item.depth for item in thread_mentions)), comment_count=len(comments), reply_count=len(replies), top_comment_ids=[item.mention_id for item in ranked_comments[:5]], top_comment_scores=[_score_for_ranking(item) for item in ranked_comments[:5]]))
        for mention in sorted_mentions:
            lineage_ids = []
            current_parent = mention.parent_id
            while current_parent:
                lineage_ids.append(current_parent)
                parent_mention = mentions_by_id.get(current_parent)
                current_parent = parent_mention.parent_id if parent_mention is not None else None
            lineage_ids.reverse()
            parent_text = mentions_by_id[mention.parent_id].raw_text if mention.parent_id is not None and mention.parent_id in mentions_by_id else None
            sibling_ids = [sibling.mention_id for sibling in children_by_parent.get(mention.parent_id or '', []) if sibling.mention_id != mention.mention_id]
            direct_child_ids = [child.mention_id for child in children_by_parent.get(mention.mention_id, [])]
            context_parts = [root_mention.raw_text]
            if parent_text and parent_text != root_mention.raw_text:
                context_parts.append(parent_text)
            if mention.raw_text not in context_parts:
                context_parts.append(mention.raw_text)
            contexts.append(MentionContext(mention_id=mention.mention_id, root_id=root_id, parent_id=mention.parent_id, lineage_ids=lineage_ids, sibling_ids=sibling_ids, direct_child_ids=direct_child_ids, context_text=' | '.join(context_parts), root_text=root_mention.raw_text, parent_text=parent_text))
    return ThreadBundle(summaries=summaries, edges=edges, contexts=contexts)

def conversation_slice(mention_id, contexts, window=3):
    for context in contexts:
        if context.mention_id == mention_id:
            trimmed_lineage = context.lineage_ids[-window:]
            return MentionContext(mention_id=context.mention_id, root_id=context.root_id, parent_id=context.parent_id, lineage_ids=trimmed_lineage, sibling_ids=context.sibling_ids[:window], direct_child_ids=context.direct_child_ids[:window], context_text=context.context_text, root_text=context.root_text, parent_text=context.parent_text)
    return None
