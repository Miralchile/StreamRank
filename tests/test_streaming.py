import unittest

from streamrank.streaming.consumer import interaction_from_payload, replay_events


class StreamingTests(unittest.TestCase):
    def test_payload_validation_rejects_poison_message(self):
        with self.assertRaisesRegex(ValueError, "is_click must be binary"):
            interaction_from_payload(
                {"user_id": 1, "item_id": 2, "event_time_ms": 3, "is_click": 2}
            )

    def test_replay_is_event_time_ordered_and_idempotence_ready(self):
        later = interaction_from_payload({"user_id": 1, "item_id": 2, "event_time_ms": 20})
        earlier = interaction_from_payload({"user_id": 1, "item_id": 1, "event_time_ms": 10})
        seen: list[tuple[int, str]] = []

        def callback(event, event_id, now_ms):
            self.assertEqual(now_ms, event.event_time_ms)
            seen.append((event.item_id, event_id))
            return True

        self.assertEqual(replay_events([later, earlier], callback), 2)
        self.assertEqual([item_id for item_id, _ in seen], [1, 2])
        self.assertEqual(len({event_id for _, event_id in seen}), 2)


if __name__ == "__main__":
    unittest.main()
