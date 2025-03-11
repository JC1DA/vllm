# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from vllm.v1.structured_output.grammar import Grammar, StructuredOutputOptions


class StructuredOutputBackend(ABC):

    @abstractmethod
    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> Grammar:
        pass

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int):
        pass
