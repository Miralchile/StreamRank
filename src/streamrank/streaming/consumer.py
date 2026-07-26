from __future__ import annotations

import json
import logging
import time
from dataclasses import fields
from typing import Callable, Iterable

from streamrank.domain import Interaction

LOGGER = logging.getLogger(__name__)


def interaction_from_payload(payload: dict[str, object]) -> Interaction:
    if not isinstance(payload, dict):
        raise ValueError("event payload must be a JSON object")
    allowed = {field.name for field in fields(Interaction)}
    event = Interaction(
        **{key: value for key, value in payload.items() if key in allowed}  # type: ignore[arg-type]
    )
    for name in ("user_id", "item_id", "event_time_ms"):
        value = getattr(event, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    for name in ("is_click", "long_view", "is_like", "is_hate", "is_follow", "is_rand"):
        if getattr(event, name) not in (0, 1):
            raise ValueError(f"{name} must be binary")
    return event


def replay_events(
    events: Iterable[Interaction],
    callback: Callable[[Interaction, str, int], bool],
    speed: float = 0.0,
) -> int:
    """Replay historical logs; speed=0 runs without wall-clock waiting."""
    ordered = sorted(events, key=lambda event: event.event_time_ms)
    processed = 0
    previous_time: int | None = None
    for index, event in enumerate(ordered):
        if speed > 0 and previous_time is not None:
            delay = max(0.0, (event.event_time_ms - previous_time) / 1000.0 / speed)
            time.sleep(min(delay, 1.0))
        event_id = f"replay:{index}:{event.user_id}:{event.item_id}:{event.event_time_ms}"
        processed += int(callback(event, event_id, event.event_time_ms))
        previous_time = event.event_time_ms
    return processed


class KafkaEventConsumer:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str = "streamrank-features"):
        try:
            from kafka import KafkaConsumer, KafkaProducer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install StreamRank with the 'streaming' extra") from exc
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            enable_auto_commit=False,
            value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
        )
        self.dlq_topic = f"{topic}.dlq"
        self.dlq_producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )

    def run(self, callback: Callable[[Interaction, str, int], bool]) -> None:
        for message in self.consumer:
            try:
                event = interaction_from_payload(message.value)
            except (TypeError, ValueError) as exc:
                LOGGER.error(
                    "dropping invalid event topic=%s partition=%s offset=%s error=%s",
                    message.topic,
                    message.partition,
                    message.offset,
                    exc,
                )
                self.dlq_producer.send(
                    self.dlq_topic,
                    {
                        "source_topic": message.topic,
                        "partition": message.partition,
                        "offset": message.offset,
                        "error": str(exc),
                        "payload": message.value,
                    },
                ).get(timeout=10)
                self.consumer.commit()
                continue
            event_id = str(
                message.value.get("_event_id")
                or f"{message.topic}:{message.partition}:{message.offset}"
            )
            callback(event, event_id, int(time.time() * 1000))
            self.consumer.commit()
