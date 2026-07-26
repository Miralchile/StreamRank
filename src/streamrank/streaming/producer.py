from __future__ import annotations

import json
from typing import Mapping


class KafkaEventProducer:
    def __init__(self, bootstrap_servers: str, topic: str):
        try:
            from kafka import KafkaProducer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install StreamRank with the 'streaming' extra") from exc
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )

    def send(self, payload: Mapping[str, object], event_id: str | None = None) -> None:
        envelope = dict(payload)
        if event_id:
            envelope["_event_id"] = event_id
        self.producer.send(self.topic, envelope).get(timeout=10)
