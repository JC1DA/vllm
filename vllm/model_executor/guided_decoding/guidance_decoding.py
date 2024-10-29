from enum import Enum
from vllm.model_executor.guided_decoding.guidance_logits_processors import (
    GuidanceLogitsProcessor
)
from vllm.sampling_params import GuidedDecodingParams
from transformers import PreTrainedTokenizerBase
from typing import Tuple, Union
from json import dumps as json_dumps
from re import escape as regex_escape

class GuidedDecodingMode(Enum):
    JSON = "json"
    REGEX = "regex"
    CHOICE = "choice"
    GRAMMAR = "grammar"


def get_local_guidance_guided_decoding_logits_processor(
    guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizerBase
) -> Union[GuidanceLogitsProcessor, None]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    """
    guide, mode = _get_guide_and_mode(guided_params)
    if not guide or not mode:
        return None

    return _get_logits_processor(
        guide, tokenizer, mode, guided_params.whitespace_pattern
    )


def _get_guide_and_mode(
    guided_params: GuidedDecodingParams,
) -> Union[Tuple[str, GuidedDecodingMode], Tuple[None, None]]:
    if guided_params.json:
        if isinstance(guided_params.json, dict):
            # turn dict into hashable string
            json = json_dumps(guided_params.json)
        else:
            json = guided_params.json
        return json, GuidedDecodingMode.JSON
    elif guided_params.regex:
        return guided_params.regex, GuidedDecodingMode.REGEX
    elif guided_params.choice:
        # choice just uses regex
        choices = [regex_escape(str(choice)) for choice in guided_params.choice]
        choices_regex = "(" + "|".join(choices) + ")"
        return choices_regex, GuidedDecodingMode.CHOICE
    # elif guided_params.grammar:
    #     return guided_params.grammar, GuidedDecodingMode.GRAMMAR
    # elif guided_params.json_object:
    #     return JSON_GRAMMAR, GuidedDecodingMode.GRAMMAR
    else:
        return None, None


def _get_logits_processor(
    guide: str,
    tokenizer: PreTrainedTokenizerBase,
    mode: GuidedDecodingMode,
    whitespace_pattern: Union[str, None],
) -> Union[GuidanceLogitsProcessor, None]:
    if mode == GuidedDecodingMode.JSON:
        return GuidanceLogitsProcessor(mode.value, guide, tokenizer, whitespace_pattern)
    elif mode == GuidedDecodingMode.REGEX or mode == GuidedDecodingMode.CHOICE:
        return GuidanceLogitsProcessor(mode.value, guide, tokenizer)
    elif mode == GuidedDecodingMode.GRAMMAR:
        return GuidanceLogitsProcessor(mode.value, guide, tokenizer)
    else:
        raise ValueError(f"Unknown guided decoding mode {mode}")
