from __future__ import annotations

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "sentence-transformers is not available in the current Python environment. "
        "Please run: python -m pip install -U sentence-transformers"
    ) from e

from dataclasses import dataclass
import traceback
from pathlib import Path

LOCAL_MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "models" / "all-MiniLM-L6-v2"
)

print("LOCAL_MODEL_PATH =", LOCAL_MODEL_PATH)
print("EXISTS =", Path(LOCAL_MODEL_PATH).exists())
from src.components.semantic_matcher import SemanticMatcher
import os
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "qianfan.baidubce.com,localhost,127.0.0.1"
@dataclass
class DummyItem:
    text: str


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_pairwise_similarity_shape_and_range() -> None:
    print_header("test_pairwise_similarity_shape_and_range")

    matcher = SemanticMatcher(
        model_name=LOCAL_MODEL_PATH,
        similarity_backend="sentence_transformers",
        cache_embeddings=False,
    )

    texts_a = [
        "agent A and B disagree on the final answer",
        "the hourly rate is wrong",
    ]
    texts_b = [
        "agent A B disagree on final answer",
        "the hourly rate is incorrect",
        "the weather is sunny today",
    ]

    sim = matcher.pairwise_similarity(texts_a, texts_b)

    print("Similarity matrix:")
    print(sim)

    assert_true(sim.shape == (2, 3), f"Expected shape (2, 3), got {sim.shape}")
    assert_true((sim <= 1.0 + 1e-8).all(), "Similarity values should be <= 1")
    assert_true((sim >= -1.0 - 1e-8).all(), "Similarity values should be >= -1")

    print("PASS")


def test_pairwise_similarity_prefers_related_texts() -> None:
    print_header("test_pairwise_similarity_prefers_related_texts")

    matcher = SemanticMatcher(
        model_name=LOCAL_MODEL_PATH,
        similarity_backend="sentence_transformers",
        cache_embeddings=False,
    )

    texts_a = ["agent A and B disagree on the final answer"]
    texts_b = [
        "agent A B disagree on final answer",
        "the weather is sunny today",
    ]

    sim = matcher.pairwise_similarity(texts_a, texts_b)

    print("Similarity matrix:")
    print(sim)

    assert_true(
        sim[0, 0] > sim[0, 1],
        f"Expected semantically related text to have higher similarity, got {sim[0,0]} <= {sim[0,1]}",
    )

    print("PASS")


def test_greedy_match_texts_basic() -> None:
    print_header("test_greedy_match_texts_basic")

    matcher = SemanticMatcher(
        model_name=LOCAL_MODEL_PATH,
        similarity_backend="sentence_transformers",
        cache_embeddings=False,
    )

    texts_a = [
        "agent A and B disagree on the final answer",
        "the hourly rate is wrong",
    ]
    texts_b = [
        "agent A B disagree on final answer",
        "the hourly rate is incorrect",
        "the weather is sunny today",
    ]

    matches = matcher.greedy_match_texts(texts_a, texts_b, threshold=0.55)

    print("Matches:")
    for m in matches:
        print(m)

    matched_pairs = {(m.prev_index, m.curr_index) for m in matches}

    assert_true((0, 0) in matched_pairs, "Expected pair (0,0) to match")
    assert_true((1, 1) in matched_pairs, "Expected pair (1,1) to match")
    assert_true(len(matches) == 2, f"Expected 2 matches, got {len(matches)}")

    print("PASS")


def test_greedy_match_items_with_unmatched() -> None:
    print_header("test_greedy_match_items_with_unmatched")

    matcher = SemanticMatcher(
        model_name=LOCAL_MODEL_PATH,
        similarity_backend="sentence_transformers",
        cache_embeddings=False,
    )

    prev_items = [
        DummyItem("agent A and B disagree on the final answer"),
        DummyItem("the hourly rate is wrong"),
    ]
    curr_items = [
        DummyItem("agent A B disagree on final answer"),
        DummyItem("the weather is sunny today"),
    ]

    result = matcher.greedy_match_items(
        prev_items=prev_items,
        curr_items=curr_items,
        text_getter=lambda x: x.text,
        threshold=0.55,
    )

    print("Matches:")
    for m in result.matches:
        print(m)

    print("Unmatched prev:", result.unmatched_prev_indices)
    print("Unmatched curr:", result.unmatched_curr_indices)

    assert_true(len(result.matches) == 1, f"Expected 1 match, got {len(result.matches)}")
    assert_true(result.matches[0].prev_index == 0, "Expected prev index 0 to be matched")
    assert_true(result.matches[0].curr_index == 0, "Expected curr index 0 to be matched")
    assert_true(result.unmatched_prev_indices == [1], f"Unexpected unmatched prev: {result.unmatched_prev_indices}")
    assert_true(result.unmatched_curr_indices == [1], f"Unexpected unmatched curr: {result.unmatched_curr_indices}")

    print("PASS")


def test_prepare_text_cleanup_effect() -> None:
    print_header("test_prepare_text_cleanup_effect")

    matcher = SemanticMatcher(
        model_name=LOCAL_MODEL_PATH,
        similarity_backend="sentence_transformers",
        cache_embeddings=False,
    )

    texts_a = ["   Answer   is   10   "]
    texts_b = ["answer is 10"]

    sim = matcher.pairwise_similarity(texts_a, texts_b)

    print("Similarity matrix:")
    print(sim)

    assert_true(sim.shape == (1, 1), f"Expected shape (1,1), got {sim.shape}")
    assert_true(sim[0, 0] > 0.90, f"Expected very high similarity, got {sim[0,0]}")

    print("PASS")


def run_all_tests() -> None:
    tests = [
        test_pairwise_similarity_shape_and_range,
        test_pairwise_similarity_prefers_related_texts,
        test_greedy_match_texts_basic,
        test_greedy_match_items_with_unmatched,
        test_prepare_text_cleanup_effect,
    ]

    passed = 0
    failed = 0

    print("\nRunning SemanticMatcher tests with sentence-transformers backend only...")

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\nFAILED: {test_func.__name__}")
            print(f"Reason: {e}")
            traceback.print_exc()

    print("\n" + "#" * 80)
    print(f"SUMMARY: passed={passed}, failed={failed}, total={len(tests)}")
    print("#" * 80)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    # force-load once so environment problems show up immediately
    SentenceTransformer("C:\\Users\\chen\\Desktop\\MY_MAD\\models\\all-MiniLM-L6-v2")
    run_all_tests()