# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_types import StructuredOutputBackend
from vllm.v1.structured_output.grammar import Grammar, StructuredOutputOptions

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


class XgrammarBackend(StructuredOutputBackend):

    def __init__(self, vllm_config: VllmConfig):
        tokenizer_group = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)  # type: ignore[arg-type]
        tokenizer_group.ping()
        self.vllm_config = vllm_config
        self.vocab_size = vllm_config.model_config.get_vocab_size()

        tokenizer = tokenizer_group.get_lora_tokenizer(None)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=self.vocab_size)
        self.compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)

    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> Grammar:
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(grammar_spec,
                                                    any_whitespace=False)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_builtin_json_grammar()
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = self.compiler.compile_regex(grammar_spec)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})")

        return Grammar(
            matcher=xgr.GrammarMatcher(ctx),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)
