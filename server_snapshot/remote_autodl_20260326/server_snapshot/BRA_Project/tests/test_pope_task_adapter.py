from __future__ import annotations

from uniground_v2.task_adapter import PopeTaskAdapter


def test_extract_object_label_strips_articles():
    adapter = PopeTaskAdapter()
    assert adapter.extract_object_label("Is there a snowboard in the image?") == "snowboard"
    assert adapter.extract_object_label("Is there an elephant in the image?") == "elephant"
    assert adapter.extract_object_label("Does this image contain the bus?") == "bus"


def test_build_runtime_context_uses_explicit_object_metadata():
    adapter = PopeTaskAdapter()
    context = adapter.build_runtime_context(
        "Is there a snowboard in the image?",
        split="popular",
        controller_mode="verifier",
        prompt_token_count=17,
        label="yes",
    )
    assert context["task_name"] == "pope"
    assert context["task_family"] == "binary_verification"
    assert context["controller_mode"] == "verifier"
    assert context["decision_mode"] == "answer_labels"
    assert context["decision_scope"] == "answer_step_only"
    assert context["retrieval_scope"] == "task_query"
    assert context["pope_split"] == "popular"
    assert context["object_label"] == "snowboard"
    assert context["task_query_text"] == "a photo of snowboard"
    assert context["retrieval_query_text"] == "a photo of snowboard"
    assert context["yes_hypothesis"] == "a photo containing snowboard"
    assert context["no_hypothesis"] == "a photo without snowboard"
    assert context["hypothesis_text_by_label"]["yes"] == "a photo containing snowboard"
    assert context["answer_labels"] == ["yes", "no"]
    assert context["prompt_token_count"] == 17
