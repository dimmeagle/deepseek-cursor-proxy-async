from __future__ import annotations

import unittest

from deepseek_cursor_proxy.reasoning_store import ReasoningStore, conversation_scope
from deepseek_cursor_proxy.streaming import (
    CursorReasoningDisplayAdapter,
    StreamAccumulator,
    fold_reasoning_into_content,
    strip_tool_tags,
)


class StreamAccumulatorTests(unittest.TestCase):
    def test_accumulates_reasoning_content_and_tool_call_deltas(self) -> None:
        store = ReasoningStore(":memory:")
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Need ",
                        },
                    }
                ]
            }
        )
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "context.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stream",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path"',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ':"README.md"}'},
                                }
                            ],
                        },
                    }
                ]
            }
        )

        scope = conversation_scope([{"role": "user", "content": "read README"}])
        stored = accumulator.store_reasoning(store, scope)

        self.assertGreater(stored, 0)
        self.assertEqual(
            store.get(f"scope:{scope}:tool_call:call_stream"), "Need context."
        )
        store.close()

    def test_stores_reasoning_when_choice_finishes_before_done(self) -> None:
        store = ReasoningStore(":memory:")
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Need a tool.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stream",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

        scope = conversation_scope([{"role": "user", "content": "lookup"}])
        stored = accumulator.store_finished_reasoning(store, scope)

        self.assertGreater(stored, 0)
        self.assertEqual(
            store.get(f"scope:{scope}:tool_call:call_stream"), "Need a tool."
        )
        self.assertEqual(accumulator.store_reasoning(store, scope), 0)
        store.close()

    def test_stores_same_streaming_choice_under_multiple_scopes(self) -> None:
        store = ReasoningStore(":memory:")
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Need a tool.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stream",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

        first_scope = conversation_scope([{"role": "user", "content": "full"}])
        second_scope = conversation_scope([{"role": "user", "content": "active"}])
        first_stored = accumulator.store_finished_reasoning(store, first_scope)
        second_stored = accumulator.store_finished_reasoning(store, second_scope)

        self.assertGreater(first_stored, 0)
        self.assertGreater(second_stored, 0)
        self.assertEqual(
            store.get(f"scope:{first_scope}:tool_call:call_stream"), "Need a tool."
        )
        self.assertEqual(
            store.get(f"scope:{second_scope}:tool_call:call_stream"), "Need a tool."
        )
        store.close()

    def test_stores_tool_call_reasoning_before_finish_reason(self) -> None:
        store = ReasoningStore(":memory:")
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Need a tool.",
                        },
                    }
                ]
            }
        )
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stream",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"query"',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )

        scope = conversation_scope([{"role": "user", "content": "lookup"}])
        stored = accumulator.store_ready_reasoning(store, scope)

        self.assertGreater(stored, 0)
        self.assertEqual(
            store.get(f"scope:{scope}:tool_call:call_stream"), "Need a tool."
        )

        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ':"README"}'},
                                }
                            ],
                        },
                    }
                ]
            }
        )

        self.assertGreater(accumulator.store_ready_reasoning(store, scope), 0)
        store.close()

    def test_stores_empty_reasoning_content_when_stream_field_is_present(
        self,
    ) -> None:
        store = ReasoningStore(":memory:")
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_empty",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

        scope = conversation_scope([{"role": "user", "content": "lookup"}])
        stored = accumulator.store_finished_reasoning(store, scope)

        self.assertGreater(stored, 0)
        self.assertEqual(store.get(f"scope:{scope}:tool_call:call_empty"), "")
        self.assertEqual(accumulator.messages()[0]["reasoning_content"], "")
        store.close()

    def test_returns_accumulated_messages_for_logging(self) -> None:
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Think.",
                            "content": "Answer.",
                        },
                    }
                ]
            }
        )

        self.assertEqual(
            accumulator.messages(),
            [
                {
                    "role": "assistant",
                    "content": "Answer.",
                    "reasoning_content": "Think.",
                }
            ],
        )


class CursorReasoningDisplayAdapterTests(unittest.TestCase):
    def test_mirrors_reasoning_content_into_details_content(self) -> None:
        adapter = CursorReasoningDisplayAdapter()
        reasoning_chunk = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Need context."},
                    "finish_reason": None,
                }
            ],
        }
        answer_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Final answer."},
                    "finish_reason": None,
                }
            ],
        }

        adapter.rewrite_chunk(reasoning_chunk)
        adapter.rewrite_chunk(answer_chunk)

        reasoning_delta = reasoning_chunk["choices"][0]["delta"]
        answer_delta = answer_chunk["choices"][0]["delta"]
        self.assertEqual(reasoning_delta["reasoning_content"], "Need context.")
        self.assertEqual(
            reasoning_delta["content"],
            "<details>\n<summary>Thinking</summary>\n\nNeed context.",
        )
        self.assertEqual(answer_delta["content"], "\n</details>\n\nFinal answer.")

    def test_can_mirror_reasoning_content_into_legacy_think_content(self) -> None:
        adapter = CursorReasoningDisplayAdapter(collapsible=False)
        reasoning_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Need context."},
                    "finish_reason": None,
                }
            ],
        }
        answer_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Final answer."},
                    "finish_reason": None,
                }
            ],
        }

        adapter.rewrite_chunk(reasoning_chunk)
        adapter.rewrite_chunk(answer_chunk)

        self.assertEqual(
            reasoning_chunk["choices"][0]["delta"]["content"], "<think>\nNeed context."
        )
        self.assertEqual(
            answer_chunk["choices"][0]["delta"]["content"],
            "\n</think>\n\nFinal answer.",
        )

    def test_closes_thinking_block_before_tool_calls(self) -> None:
        adapter = CursorReasoningDisplayAdapter()
        adapter.rewrite_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "Need a tool."},
                    }
                ]
            }
        )
        tool_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ]
        }

        adapter.rewrite_chunk(tool_chunk)

        self.assertEqual(
            tool_chunk["choices"][0]["delta"]["content"], "\n</details>\n\n"
        )

    def test_flush_chunk_closes_unfinished_thinking_block_at_done(self) -> None:
        adapter = CursorReasoningDisplayAdapter()
        adapter.rewrite_chunk(
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "Still thinking."},
                    }
                ],
            }
        )

        closing_chunk = adapter.flush_chunk("deepseek-v4-pro")

        self.assertIsNotNone(closing_chunk)
        assert closing_chunk is not None
        self.assertEqual(closing_chunk["model"], "deepseek-v4-pro")
        self.assertEqual(
            closing_chunk["choices"][0]["delta"]["content"], "\n</details>\n\n"
        )
        self.assertIsNone(adapter.flush_chunk("deepseek-v4-pro"))


class FoldReasoningTests(unittest.TestCase):
    def test_fold_reasoning_into_non_streaming_content(self) -> None:
        """Non-streaming responses mirror reasoning_content into a visible
        <details> block, matching the streaming layout."""
        payload = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "thinking",
                    },
                }
            ]
        }
        fold_reasoning_into_content(payload, collapsible=True)
        self.assertEqual(
            payload["choices"][0]["message"]["content"],
            "<details>\n<summary>Thinking</summary>\n\nthinking\n</details>\n\nanswer",
        )

    def test_fold_reasoning_skips_empty_reasoning(self) -> None:
        payload = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "",
                    },
                }
            ]
        }
        fold_reasoning_into_content(payload, collapsible=True)
        self.assertEqual(payload["choices"][0]["message"]["content"], "answer")


class StripToolTagsTests(unittest.TestCase):
    def test_removes_tool_comment_block(self) -> None:
        text = "before <tool_comment>\nsome comment\n</tool_comment> after"
        self.assertEqual(strip_tool_tags(text), "before  after")

    def test_removes_tool_use_block(self) -> None:
        text = "prefix <tool_use>\n{\"name\": \"read\"}\n</tool_use> suffix"
        self.assertEqual(strip_tool_tags(text), "prefix  suffix")

    def test_removes_multiple_blocks(self) -> None:
        text = (
            "start "
            "<tool_comment>note</tool_comment> "
            "middle "
            "<tool_use>action</tool_use> "
            "end"
        )
        self.assertEqual(strip_tool_tags(text), "start  middle  end")

    def test_removes_block_with_attributes(self) -> None:
        text = '<tool_comment type="info">attr test</tool_comment>'
        self.assertEqual(strip_tool_tags(text), "")

    def test_case_insensitive(self) -> None:
        text = "<TOOL_COMMENT>CAPS</TOOL_COMMENT> <Tool_Use>Mixed</Tool_Use>"
        self.assertEqual(strip_tool_tags(text), " ")

    def test_nested_strips_first_complete_pair(self) -> None:
        # Non-greedy regex matches from first <tool_comment> to first
        # </tool_comment>, leaving the unpaired outer closing tag.
        text = (
            "<tool_comment>"
            "outer <tool_comment>inner</tool_comment>"
            "</tool_comment>"
        )
        # Removes: <tool_comment>outer <tool_comment>inner</tool_comment>
        self.assertEqual(strip_tool_tags(text), "</tool_comment>")

    def test_preserves_normal_text(self) -> None:
        text = "regular text without any tags"
        self.assertEqual(strip_tool_tags(text), text)

    def test_empty_string(self) -> None:
        self.assertEqual(strip_tool_tags(""), "")

    def test_only_partial_tag_left_intact(self) -> None:
        text = "<tool_comment>unclosed"
        self.assertEqual(strip_tool_tags(text), text)

    def test_multiline_block(self) -> None:
        text = (
            "<tool_comment>\n"
            "line 1\n"
            "line 2\n"
            "</tool_comment>\n"
            "visible"
        )
        self.assertEqual(strip_tool_tags(text), "\nvisible")


if __name__ == "__main__":
    unittest.main()
