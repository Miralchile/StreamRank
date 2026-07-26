from __future__ import annotations

import logging
import os

from streamrank.online.state import RedisOnlineState
from streamrank.streaming.consumer import KafkaEventConsumer


def main() -> int:
    logging.basicConfig(level=os.getenv("STREAMRANK_LOG_LEVEL", "INFO"))
    state = RedisOnlineState(os.getenv("STREAMRANK_REDIS_URL", "redis://localhost:6379/0"))
    state.ping()
    consumer = KafkaEventConsumer(
        os.getenv("STREAMRANK_KAFKA_BOOTSTRAP", "localhost:19092"),
        os.getenv("STREAMRANK_EVENT_TOPIC", "streamrank.events"),
        os.getenv("STREAMRANK_CONSUMER_GROUP", "streamrank-features"),
    )
    consumer.run(state.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
