# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import multiprocessing
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import LazyLoader
from vllm.v1.structured_output.grammar import Grammar, StructuredOutputOptions

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch
    import xgrammar as xgr

    from vllm.v1.request import Request
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


class StructuredOutputBackend(ABC):

    @abstractmethod
    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> Grammar:
        pass

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        pass


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

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: Optional[StructuredOutputBackend] = None
        self.vllm_config = vllm_config

        self._grammar_bitmask: Optional[torch.Tensor] = None

        # The default max_workers if not specified is the number of CPUs * 5,
        # which is way too high since these tasks are CPU-bound, not I/O bound.
        # We also know we would never dominate CPU usage with just grammar
        # compilation, so we set it to half the number of CPUs.
        max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def grammar_init(self, request: Request) -> None:
        if request.structured_output_request is None:
            return

        # Initialize the backend the first time it is needed.
        if self.backend is None:
            backend_name = request.sampling_params.guided_decoding.backend_name
            if backend_name == "xgrammar":
                self.backend = XgrammarBackend(self.vllm_config)
            else:
                raise ValueError(
                    f"Unsupported structured output backend: {backend_name}")

        grammar: Future[Grammar] = self.executor.submit(
            self._async_create_grammar, request, self.backend)
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]

    def _async_create_grammar(self, request: Request,
                              backend: StructuredOutputBackend) -> Grammar:
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def grammar_bitmask(
        self,
        requests: dict[str, Request],
        structured_output_request_ids: dict[str, int],
        batch_len: int,
    ) -> Optional[npt.NDArray[np.int32]]:
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        if self._grammar_bitmask is None:
            assert self.backend is not None
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                self.vllm_config.scheduler_config.max_num_seqs)

        # Fill the bitmask using the index of each request equal to its
        # position in the batch. Resize the bitmask down to the size of
        # the batch.
        bitmask_tensor = self._grammar_bitmask
        for req_id, batch_index in structured_output_request_ids.items():
            request = requests[req_id].structured_output_request
            assert request is not None and request.grammar is not None
            if not request.grammar.matcher.is_terminated():
                request.grammar.fill_bitmask(bitmask_tensor, batch_index)
        if batch_len < self._grammar_bitmask.shape[0]:
            bitmask_tensor = self._grammar_bitmask[:batch_len]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()
