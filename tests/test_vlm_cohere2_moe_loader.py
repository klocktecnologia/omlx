# SPDX-License-Identifier: Apache-2.0
"""Tests for the mlx-vlm text-only load path and its model-type dispatch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from omlx.engine import vlm as vlm_module
from omlx.engine.vlm import (
    VLMBatchedEngine,
    _load_text_model_with_tokenizer_fallback,
)
from omlx.exceptions import InvalidRequestError


class _FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    eos_token_ids = None
    pad_token = None


class _FakeDetokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _FakeStoppingCriteria:
    def __init__(self, eos_token_ids, tokenizer):
        self.eos_token_ids = eos_token_ids
        self.tokenizer = tokenizer


def test_cohere2_moe_loader_uses_upstream_processor(monkeypatch, tmp_path):
    import mlx_vlm.utils as vlm_utils

    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[2]))
    processor = object()

    monkeypatch.setattr(vlm_utils, "get_model_path", lambda model_name: tmp_path)
    monkeypatch.setattr(vlm_utils, "load_model", lambda *a, **k: model)
    monkeypatch.setattr(vlm_utils, "load_processor", lambda *a, **k: processor)

    loaded_model, loaded_processor = _load_text_model_with_tokenizer_fallback("cohere")

    assert loaded_model is model
    assert loaded_processor is processor


def test_cohere2_moe_loader_falls_back_to_tokenizer(monkeypatch, tmp_path):
    import mlx_vlm.tokenizer_utils as tokenizer_utils
    import mlx_vlm.utils as vlm_utils
    import transformers

    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[7]))
    tokenizer = _FakeTokenizer()

    monkeypatch.setattr(vlm_utils, "get_model_path", lambda model_name: tmp_path)
    monkeypatch.setattr(vlm_utils, "load_model", lambda *a, **k: model)

    def fail_processor(*args, **kwargs):
        raise ValueError("no processor")

    monkeypatch.setattr(vlm_utils, "load_processor", fail_processor)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: tokenizer,
    )
    monkeypatch.setattr(
        tokenizer_utils,
        "load_tokenizer",
        lambda *a, **k: _FakeDetokenizer,
    )
    monkeypatch.setattr(vlm_utils, "StoppingCriteria", _FakeStoppingCriteria)

    loaded_model, loaded_processor = _load_text_model_with_tokenizer_fallback("cohere")

    assert loaded_model is model
    assert loaded_processor is tokenizer
    assert tokenizer.pad_token == "<eos>"
    assert isinstance(tokenizer.detokenizer, _FakeDetokenizer)
    assert isinstance(tokenizer.stopping_criteria, _FakeStoppingCriteria)
    assert tokenizer.stopping_criteria.eos_token_ids == [7]


def test_cohere2_moe_rejects_image_input():
    engine = VLMBatchedEngine("cohere")
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type=vlm_module.COHERE2_MOE_MODEL_TYPE)
    )

    with pytest.raises(InvalidRequestError, match="text-only"):
        engine._prepare_vision_inputs(
            [{"role": "user", "content": "describe"}],
            images=[object()],
        )


def test_cohere2_moe_rejects_audio_input():
    engine = VLMBatchedEngine("cohere")
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type=vlm_module.COHERE2_MOE_MODEL_TYPE)
    )

    with pytest.raises(InvalidRequestError, match="text-only"):
        engine._prepare_vision_inputs(
            [{"role": "user", "content": "transcribe"}],
            images=[],
            audio=[("samples", 16000)],
        )


BOLT_TEMPLATE = Path(__file__).parent / "fixtures" / "nemotron_bolt_chat_template.jinja"


def _write_dangling_processor_checkpoint(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path / "tokenizer.json"))
    (path / "config.json").write_text(
        json.dumps({"model_type": "nemotron_h_puzzle", "eos_token_id": 1})
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "<eos>",
                "auto_map": {
                    "AutoProcessor": "processing_missing.MissingProcessor",
                },
            }
        )
    )
    shutil.copy(BOLT_TEMPLATE, path / "chat_template.jinja")


def test_text_loader_uses_tokenizer_when_processor_module_is_missing(
    monkeypatch, tmp_path
):
    import mlx_vlm.utils as vlm_utils

    _write_dangling_processor_checkpoint(tmp_path)
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[1]))
    monkeypatch.setattr(vlm_utils, "load_model", lambda *a, **k: model)

    loaded_model, processor = _load_text_model_with_tokenizer_fallback(str(tmp_path))

    assert loaded_model is model
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    assert "<tools>" in rendered
    assert "get_weather" in rendered
    assert processor.detokenizer is not None
    assert processor.stopping_criteria.eos_token_ids == [1]


def test_text_loader_propagates_tokenizer_error_when_both_loads_fail(
    monkeypatch, tmp_path
):
    import mlx_vlm.utils as vlm_utils

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "nemotron_h_puzzle"})
    )
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[1]))
    monkeypatch.setattr(vlm_utils, "load_model", lambda *a, **k: model)

    def broken_processor(*args, **kwargs):
        raise ValueError("processor cannot load")

    monkeypatch.setattr(vlm_utils, "load_processor", broken_processor)

    with pytest.raises((OSError, ValueError)) as excinfo:
        _load_text_model_with_tokenizer_fallback(str(tmp_path))

    assert "processor cannot load" not in str(excinfo.value)


class _LoadFinishedError(Exception):
    pass


async def _start_and_capture(monkeypatch, tmp_path, model_type):
    """Run engine start up to the point it holds the loaded model/processor."""
    import mlx_vlm.utils as vlm_utils

    import omlx.models.vlm as models_vlm
    from omlx.utils import model_loading

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    text_model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[1]))
    text_processor = object()
    full_model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[1]))
    full_processor = object()

    monkeypatch.setattr(vlm_utils, "get_model_path", lambda name: tmp_path)
    monkeypatch.setattr(vlm_utils, "load_model", lambda *a, **k: text_model)
    monkeypatch.setattr(vlm_utils, "load_processor", lambda *a, **k: text_processor)
    monkeypatch.setattr(vlm_utils, "load", lambda *a, **k: (full_model, full_processor))
    monkeypatch.setattr(
        model_loading, "maybe_apply_pre_load_patches", lambda *a, **k: None
    )
    monkeypatch.setattr(
        model_loading, "maybe_load_custom_quantization", lambda *a, **k: None
    )

    def stop_after_load(model):
        raise _LoadFinishedError

    monkeypatch.setattr(models_vlm, "restore_bonsai_quantized_modules", stop_after_load)

    engine = VLMBatchedEngine(str(tmp_path))
    with pytest.raises(_LoadFinishedError):
        await engine.start()
    return engine, (text_model, text_processor), (full_model, full_processor)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["nemotron_h_puzzle", "cohere2_moe"])
async def test_engine_loads_text_only_types_without_the_vlm_loader(
    monkeypatch, tmp_path, model_type
):
    engine, text, _full = await _start_and_capture(monkeypatch, tmp_path, model_type)

    assert (engine._vlm_model, engine._processor) == text


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["nemotron_h_omni", "nemotron_h"])
async def test_engine_keeps_the_vlm_loader_for_other_nemotron_types(
    monkeypatch, tmp_path, model_type
):
    engine, _text, full = await _start_and_capture(monkeypatch, tmp_path, model_type)

    assert (engine._vlm_model, engine._processor) == full
