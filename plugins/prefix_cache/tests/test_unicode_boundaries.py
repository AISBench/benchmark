import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ais_bench_prefix_cache.artifacts import read_jsonl, validate_artifacts
from ais_bench_prefix_cache.errors import ArtifactValidationError
from ais_bench_prefix_cache.generation import (
    CanonicalPrefix,
    GSMRecord,
    _repair_token_segment,
    build_canonical_prefixes,
    build_prompt,
)
from ais_bench_prefix_cache.pipeline import _build_prompt_with_seed_retry, _plan_request_shape, prepare_scenario
from ais_bench_prefix_cache.scenario import load_scenario
from tests.test_core import scenario_dict


class ByteTokenizer:
    """Deterministic byte-level tokenizer; arbitrary cuts can break UTF-8."""
    all_special_ids = list(range(32))

    def __len__(self):
        return 256

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8", errors="replace")


def record(text, index=0):
    return GSMRecord(index, text, hashlib.sha256(text.encode()).hexdigest())


class UnicodeBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()

    def test_suffix_repair_preserves_fixed_prefix_seed_and_length(self):
        canonical = CanonicalPrefix("g0", "P", (80,), "hash", (), ())
        for source, length in (("A😀B", 4), ("A汉B", 3)):
            with self.subTest(source=source):
                repairs = []
                text, tokens, indices, hashes = build_prompt(
                    self.tokenizer, canonical, 1, [33], [record(source)], length + 2,
                    safe_ids=[32, 46], boundary_repairs=repairs,
                )
                self.assertEqual(tokens[:2], (80, 33))
                self.assertEqual(len(tokens), length + 2)
                self.assertEqual(self.tokenizer.encode(text), list(tokens))
                self.assertNotIn("\ufffd", text)
                self.assertEqual(repairs[0]["region"], "suffix")
                self.assertEqual(repairs[0]["padding_tokens"], length - 1)
                self.assertEqual(indices, (0,))
                self.assertEqual(hashes, (record(source).question_sha256,))

    def test_normal_and_literal_replacement_character_are_not_modified(self):
        canonical = CanonicalPrefix("g0", "P", (80,), "hash", (), ())
        for source in ("normal text", "\ufffd", "汉😀"):
            with self.subTest(source=source), patch(
                "ais_bench_prefix_cache.generation._repair_token_segment",
                side_effect=AssertionError("normal fast path must not repair"),
            ):
                expected = [80, 33] + self.tokenizer.encode(source)
                repairs = []
                text, tokens, _, _ = build_prompt(
                    self.tokenizer, canonical, 1, [33], [record(source)], len(expected),
                    safe_ids=[32], boundary_repairs=repairs,
                )
                self.assertEqual(list(tokens), expected)
                self.assertEqual(text, "P!" + source)
                self.assertEqual(repairs, [])

    def test_bad_suffix_is_repaired_before_regenerating_seed(self):
        canonical = CanonicalPrefix("g0", "P", (80,), "hash", (), ())
        seeds = {"r0": (33,)}
        repairs = []
        with patch("ais_bench_prefix_cache.pipeline.build_unique_seed", side_effect=AssertionError("must keep seed")):
            text, tokens, _, _ = _build_prompt_with_seed_retry(
                self.tokenizer, canonical, 1, seeds, "r0", [record("A😀B")],
                6, [32, 46], 1, 42, boundary_repairs=repairs,
            )
        self.assertEqual(seeds, {"r0": (33,)})
        self.assertEqual(tokens[:2], (80, 33))
        self.assertEqual(self.tokenizer.encode(text), list(tokens))
        self.assertEqual(len(repairs), 1)

    def test_intermediate_canonical_cut_and_following_continuation_are_repaired(self):
        sources = {"g0": [record("A😀B")], "g1": [record("C😀D", 1)]}
        canonical = build_canonical_prefixes(
            self.tokenizer, sources, {"g0": 8, "g1": 8}, 4,
            prefix_lengths={"g0": [4, 8], "g1": [4, 8]}, safe_ids=[32, 46],
        )
        for item in canonical.values():
            self.assertEqual(len(item.token_ids), 8)
            self.assertTrue(item.boundary_repairs)
            for length in (4, 8):
                prefix = item.token_ids[:length]
                text = self.tokenizer.decode(prefix)
                self.assertEqual(self.tokenizer.encode(text), list(prefix))
                self.assertNotIn("\ufffd", text)
            # Fixing the following orphan byte must not rewrite the first cut.
            self.assertEqual(item.boundary_repairs[1]["start_token"], 4)
            self.assertGreater(item.boundary_repairs[1]["removed_head_tokens"], 0)
        self.assertNotEqual(canonical["g0"].token_ids[:4], canonical["g1"].token_ids[:4])

    def test_repair_does_not_introduce_a_group_first_block_collision(self):
        sources = {"g0": [record("😀a")], "g1": [record("😁b", 1)]}
        canonical = build_canonical_prefixes(
            self.tokenizer, sources, {"g0": 8, "g1": 8}, 2,
            prefix_lengths={"g0": [2, 8], "g1": [2, 8]}, safe_ids=[32],
        )
        self.assertNotEqual(canonical["g0"].token_ids[:2], canonical["g1"].token_ids[:2])
        for item in canonical.values():
            for length in (2, 8):
                prefix = item.token_ids[:length]
                self.assertEqual(self.tokenizer.encode(self.tokenizer.decode(prefix)), list(prefix))

    def test_repair_has_a_fixed_attempt_budget(self):
        class UnstableTokenizer(ByteTokenizer):
            calls = 0

            def encode(self, text, add_special_tokens=False):
                self.calls += 1
                return [1]

        tokenizer = UnstableTokenizer()
        with self.assertRaisesRegex(ArtifactValidationError, "unable to repair suffix token boundary"):
            _repair_token_segment(tokenizer, [80], [65] * 1024, [32, 33, 34, 35], "suffix")
        self.assertLessEqual(tokenizer.calls, 64)

    def test_prepare_keeps_the_planned_hits_and_shared_structure(self):
        for mode in ("cold", "warmup"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "gsm.jsonl").write_text(
                    "".join(json.dumps({"question": text}, ensure_ascii=False) + "\n"
                            for text in ("A😀B汉", "C😁D字", "E🚀F题", "G🧪H目")),
                    encoding="utf-8",
                )
                data = scenario_dict(root, mode=mode, dp_size=2)
                data["requests"]["input_length"] = {
                    "mode": "explicit", "values": [33, 41, 49, 57, 65, 73, 81, 89],
                }
                source = root / "scenario.json"
                source.write_text(json.dumps(data), encoding="utf-8")
                lengths, _, _, _, _, solve = _plan_request_shape(load_scenario(source), load_scenario(source).data)
                paths = prepare_scenario(source, tokenizer_loader=lambda _: self.tokenizer)
                self.assertTrue(validate_artifacts(paths.manifest)["ok"])
                rows = read_jsonl(paths.full)
                manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
                self.assertEqual([row["shared_prefix_tokens"] for row in rows], list(solve.shared_prefix_tokens))
                self.assertEqual(sum(row["theoretical_hit_tokens"] for row in rows), solve.effective_hit_tokens)
                seeds = set()
                previous = {}
                cached_blocks = {}
                for warm in manifest["warmup"]["plan"]:
                    warm_tokens = self.tokenizer.encode(warm["prompt"])
                    cached_blocks.setdefault(warm["dp_rank"], set()).update(
                        tuple(warm_tokens[:end]) for end in range(4, len(warm_tokens) + 1, 4)
                    )
                for index, (row, length) in enumerate(zip(rows, lengths)):
                    tokens = self.tokenizer.encode(row["question"])
                    prefix_length = row["shared_prefix_tokens"]
                    self.assertEqual(len(tokens), length)
                    self.assertNotIn("\ufffd", row["question"])
                    self.assertEqual(prefix_length + row["seed_tokens"] + row["natural_suffix_tokens"]
                                     + row.get("suffix_padding_tokens", 0), length)
                    seeds.add(tuple(tokens[prefix_length:prefix_length + row["seed_tokens"]]))
                    prefix = tokens[:prefix_length]
                    earlier = previous.get(row["group_id"])
                    if earlier is not None:
                        common = min(len(prefix), len(earlier))
                        self.assertEqual(prefix[:common], earlier[:common])
                    previous[row["group_id"]] = prefix
                    rank = row["dp_rank"] if mode == "cold" else index % 2
                    cache = cached_blocks.setdefault(rank, set())
                    hit = 0
                    for end in range(4, len(tokens) + 1, 4):
                        if tuple(tokens[:end]) not in cache:
                            break
                        hit = end
                    self.assertEqual(hit, row["theoretical_hit_tokens"])
                    cache.update(tuple(tokens[:end]) for end in range(4, len(tokens) + 1, 4))
                self.assertEqual(len(seeds), len(rows))
                self.assertTrue(any(row.get("boundary_repairs") for row in rows))
                for warm in manifest["warmup"]["plan"]:
                    self.assertEqual(len(self.tokenizer.encode(warm["prompt"])), warm["input_tokens"])
                    self.assertFalse(warm["included_in_formal_statistics"])


if __name__ == "__main__":
    unittest.main()
