"""BrainConfig for the OpenCode Go provider: default model and effort values."""

import pytest
from pydantic import ValidationError

from sellbird.config import (
    BRAIN_PROVIDERS,
    OPENCODE_GO_DEFAULT_MODEL,
    OPENCODE_GO_REASONING_EFFORTS,
    BrainConfig,
)


def test_opencode_go_is_a_provider():
    assert "opencode_go" in BRAIN_PROVIDERS
    assert BrainConfig(provider="opencode_go").provider == "opencode_go"


def test_default_model_is_deepseek():
    assert OPENCODE_GO_DEFAULT_MODEL == "deepseek-v4.1-flash"
    assert BrainConfig(provider="opencode_go").model == "deepseek-v4.1-flash"


def test_explicit_model_wins():
    cfg = BrainConfig(provider="opencode_go", model="deepseek-v4-pro")
    assert cfg.model == "deepseek-v4-pro"


def test_max_effort_allowed_only_for_opencode_go():
    assert "max" in OPENCODE_GO_REASONING_EFFORTS
    cfg = BrainConfig(provider="opencode_go", reasoning_effort="MAX")
    assert cfg.reasoning_effort == "max"
    with pytest.raises(ValidationError):
        BrainConfig(provider="codex", reasoning_effort="max")
    with pytest.raises(ValidationError):
        BrainConfig(reasoning_effort="max")  # claude default


def test_unknown_effort_rejected_for_opencode_go():
    with pytest.raises(ValidationError):
        BrainConfig(provider="opencode_go", reasoning_effort="turbo")


def test_claude_and_codex_models_unchanged():
    assert BrainConfig().model == ""
    assert BrainConfig(provider="codex").model == ""
