from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import get_model_info
from litellm.llms.anthropic.chat.transformation import AnthropicConfig

REPO_ROOT: Final = Path(__file__).parents[2]
MODELS: Final = ("vertex_ai/claude-haiku-4-5", "vertex_ai/claude-haiku-4-5@20251001")
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/haiku-4-5
# ("Maximum output tokens: 64,000") — the same cap every non-vertex claude-haiku-4-5 entry already carries.
HAIKU_4_5_MAX_OUTPUT_TOKENS: Final = 64000
COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])


def _cost_map_entry(path: Path, model: str) -> dict[str, object]:
    return COST_MAP_ADAPTER.validate_json(path.read_bytes())[model]


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("model", MODELS)
def test_vertex_ai_claude_haiku_4_5_output_cap_matches_google(model: str) -> None:
    info = get_model_info(model=model, custom_llm_provider="vertex_ai")
    assert info["litellm_provider"] == "vertex_ai-anthropic_models"
    assert info["max_input_tokens"] == 200000
    assert info["max_output_tokens"] == HAIKU_4_5_MAX_OUTPUT_TOKENS
    assert info["max_tokens"] == HAIKU_4_5_MAX_OUTPUT_TOKENS

    # The anthropic transformation fills a missing request max_tokens from this value.
    assert AnthropicConfig.get_max_tokens_for_model(model) == HAIKU_4_5_MAX_OUTPUT_TOKENS


@pytest.mark.usefixtures("local_model_cost_map")
def test_vertex_ai_claude_haiku_4_5_matches_the_first_party_entry() -> None:
    first_party = get_model_info(model="claude-haiku-4-5", custom_llm_provider="anthropic")
    for model in MODELS:
        vertex = get_model_info(model=model, custom_llm_provider="vertex_ai")
        assert vertex["max_output_tokens"] == first_party["max_output_tokens"]
        assert vertex["max_tokens"] == first_party["max_tokens"]


@pytest.mark.parametrize("model", MODELS)
def test_vertex_ai_claude_haiku_4_5_entry_and_backup_match(model: str) -> None:
    main_entry = _cost_map_entry(REPO_ROOT / "model_prices_and_context_window.json", model)
    backup_entry = _cost_map_entry(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json", model)

    assert main_entry["max_output_tokens"] == HAIKU_4_5_MAX_OUTPUT_TOKENS
    assert backup_entry == main_entry
