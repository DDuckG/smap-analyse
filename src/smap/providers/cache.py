from __future__ import annotations
import hashlib
import sqlite3
from array import array
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path

class EmbeddingCacheStore:

    def __init__(self, root):
        self.root = root
        self._connection: sqlite3.Connection | None = None
        self._pending_access_updates: set[str] = set()
        self._access_flush_threshold = 256
        if self.root.suffix:
            self.db_path = self.root
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.db_path = self.root / 'embeddings.sqlite3'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = self._connect()
        self._connection.execute('\n            CREATE TABLE IF NOT EXISTS embeddings (\n                cache_key TEXT PRIMARY KEY,\n                model_id TEXT NOT NULL,\n                purpose TEXT NOT NULL,\n                text_hash TEXT NOT NULL,\n                text TEXT NOT NULL,\n                vector_blob BLOB NOT NULL,\n                created_at TEXT DEFAULT CURRENT_TIMESTAMP,\n                last_accessed_at TEXT DEFAULT CURRENT_TIMESTAMP\n            )\n            ')
        self._connection.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_model_purpose_text_hash ON embeddings(model_id, purpose, text_hash)')
        self._connection.commit()

    def close(self):
        connection = getattr(self, '_connection', None)
        if connection is None:
            return
        self._flush_access_updates()
        with suppress(sqlite3.Error):
            connection.close()
        self._connection = None

    def __del__(self):
        self.close()

    def _connect(self):
        connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA synchronous=NORMAL')
        connection.execute('PRAGMA temp_store=MEMORY')
        return connection

    def _cache_key(self, *, model_id, purpose, text):
        return hashlib.sha256(f'{model_id}\n{purpose}\n{text}'.encode()).hexdigest()

    def _text_hash(self, text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def load(self, *, model_id, purpose, text):
        if self._connection is None:
            return None
        cache_key = self._cache_key(model_id=model_id, purpose=purpose, text=text)
        row = self._connection.execute('SELECT vector_blob FROM embeddings WHERE cache_key = ?', (cache_key,)).fetchone()
        if row is None:
            return None
        self._mark_accessed(cache_key)
        return self._deserialize_vector(row[0])

    def store(self, *, model_id, purpose, text, vector):
        self.store_many([{'model_id': model_id, 'purpose': purpose, 'text': text, 'vector': vector}])

    def store_many(self, entries):
        if self._connection is None:
            return
        if not entries:
            return
        payload = [(self._cache_key(model_id=str(entry['model_id']), purpose=str(entry['purpose']), text=str(entry['text'])), str(entry['model_id']), str(entry['purpose']), self._text_hash(str(entry['text'])), str(entry['text']), self._serialize_vector(tuple((float(value) for value in self._vector_values(entry['vector']))))) for entry in entries]
        self._connection.executemany('\n            INSERT INTO embeddings(cache_key, model_id, purpose, text_hash, text, vector_blob, created_at, last_accessed_at)\n            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)\n            ON CONFLICT(cache_key) DO UPDATE SET\n                vector_blob = excluded.vector_blob,\n                last_accessed_at = CURRENT_TIMESTAMP\n            ', payload)
        self._connection.commit()

    def load_many(self, *, model_id, purpose, texts):
        if not texts or self._connection is None:
            return {}
        key_to_text = {self._cache_key(model_id=model_id, purpose=purpose, text=text): text for text in texts}
        placeholders = ','.join(('?' for _ in key_to_text))
        rows = self._connection.execute(f'SELECT cache_key, vector_blob FROM embeddings WHERE cache_key IN ({placeholders})', tuple(key_to_text)).fetchall()
        if not rows:
            return {}
        self._mark_accessed(*(str(row[0]) for row in rows))
        return {key_to_text[str(cache_key)]: self._deserialize_vector(vector_blob) for cache_key, vector_blob in rows if str(cache_key) in key_to_text}

    def _mark_accessed(self, *cache_keys):
        if self._connection is None or not cache_keys:
            return
        self._pending_access_updates.update(cache_keys)
        if len(self._pending_access_updates) >= self._access_flush_threshold:
            self._flush_access_updates()

    def _flush_access_updates(self):
        if self._connection is None or not self._pending_access_updates:
            return
        self._connection.executemany('UPDATE embeddings SET last_accessed_at = CURRENT_TIMESTAMP WHERE cache_key = ?', [(cache_key,) for cache_key in sorted(self._pending_access_updates)])
        self._connection.commit()
        self._pending_access_updates.clear()

    def _serialize_vector(self, vector):
        return array('d', vector).tobytes()

    def _deserialize_vector(self, payload):
        values = array('d')
        values.frombytes(payload)
        return tuple((float(value) for value in values))

    def _vector_values(self, value):
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return value
        raise TypeError('Embedding cache entries must provide vector values as a list or tuple of floats.')
