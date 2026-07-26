from __future__ import annotations

from collections import defaultdict, deque

from streamrank.domain import Interaction


class InMemoryOnlineState:
    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        self.histories: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=max_history))
        self.timed_histories: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.processed_event_ids: set[str] = set()
        self.consumer_lag_ms = 0

    def apply(
        self, event: Interaction, event_id: str | None = None, now_ms: int | None = None
    ) -> bool:
        dedupe_key = event_id or (
            f"{event.logging_policy}:{event.user_id}:{event.item_id}:{event.event_time_ms}"
        )
        if dedupe_key in self.processed_event_ids:
            return False
        self.processed_event_ids.add(dedupe_key)
        if event.is_click or event.long_view or event.is_like:
            self.histories[event.user_id].append(event.item_id)
            self.timed_histories[event.user_id].append((event.event_time_ms, event.item_id))
        if now_ms is not None:
            self.consumer_lag_ms = max(0, now_ms - event.event_time_ms)
        return True

    def history(self, user_id: int, before_ms: int | None = None) -> list[int]:
        if before_ms is not None:
            return [
                item_id
                for event_time_ms, item_id in self.timed_histories[user_id]
                if event_time_ms < before_ms
            ]
        return list(self.histories[user_id])


class RedisOnlineState:
    """Redis adapter with idempotent event application and bounded histories."""

    def __init__(self, redis_url: str, max_history: int = 200):
        try:
            import redis
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install StreamRank with the 'serving' extra") from exc
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.max_history = max_history

    def apply(
        self, event: Interaction, event_id: str | None = None, now_ms: int | None = None
    ) -> bool:
        dedupe_key = event_id or (
            f"{event.logging_policy}:{event.user_id}:{event.item_id}:{event.event_time_ms}"
        )
        positive = int(bool(event.is_click or event.long_view or event.is_like))
        lag = max(0, now_ms - event.event_time_ms) if now_ms is not None else -1
        member = f"{event.event_time_ms}|{event.item_id}|{dedupe_key}"
        script = """
        if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
        redis.call('SET', KEYS[1], '1', 'EX', 604800)
        if ARGV[1] == '1' then
          redis.call('RPUSH', KEYS[2], ARGV[2])
          redis.call('LTRIM', KEYS[2], -tonumber(ARGV[3]), -1)
          redis.call('ZADD', KEYS[3], tonumber(ARGV[4]), ARGV[5])
          redis.call('ZREMRANGEBYRANK', KEYS[3], 0, -tonumber(ARGV[3]) - 1)
        end
        if tonumber(ARGV[6]) >= 0 then redis.call('SET', KEYS[4], ARGV[6]) end
        return 1
        """
        applied = self.client.eval(
            script,
            4,
            f"streamrank:event:{dedupe_key}",
            f"streamrank:history:{event.user_id}",
            f"streamrank:timed_history:{event.user_id}",
            "streamrank:consumer_lag_ms",
            positive,
            event.item_id,
            self.max_history,
            event.event_time_ms,
            member,
            lag,
        )
        return bool(applied)

    def history(self, user_id: int, before_ms: int | None = None) -> list[int]:
        if before_ms is not None:
            members = self.client.zrangebyscore(
                f"streamrank:timed_history:{user_id}", "-inf", f"({before_ms}"
            )
            return [int(member.split("|", 2)[1]) for member in members]
        return [int(value) for value in self.client.lrange(f"streamrank:history:{user_id}", 0, -1)]

    @property
    def consumer_lag_ms(self) -> int:
        value = self.client.get("streamrank:consumer_lag_ms")
        return int(value or 0)

    def ping(self) -> bool:
        return bool(self.client.ping())
