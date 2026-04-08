from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel

from smap.normalization.models import MentionRecord
from smap.threads.models import MentionContext


class MentionEnricher(Protocol):
    name: str

    def enrich(
        self,
        mention: MentionRecord,
        context: MentionContext | None,
    ) -> Sequence[BaseModel]: ...
