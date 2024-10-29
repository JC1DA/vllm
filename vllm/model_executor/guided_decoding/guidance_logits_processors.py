import os
import time
import guidance
import torch
import json
import numpy as np
import llguidance  # type: ignore[import-untyped]
from pydantic import BaseModel
from typing import Type, Union, List
from guidance.models import TransformersTokenizer
from guidance._schema import LLInterpreterResponse
from transformers import PreTrainedTokenizerBase

class GuidanceLogitsProcessor:
    metadata = {}

    def __init__(
        self,
        mode: str,
        guide: Union[dict, Type[BaseModel], str],
        tokenizer: PreTrainedTokenizerBase,
        whitespace_pattern: Union[str, None] = None,
    ) -> None:
        self.mode = mode
        self.initialized = False
        self.guide = guide
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer.name_or_path
        self.whitespace_pattern = whitespace_pattern

        self.is_stopped = False
        self.pending_ff_tokens: list[int] = []
        self.new_sampling = False

    def initialize(self):
        if self.initialized:
            return
        
        if self.mode.lower() == "json":
            if isinstance(self.guide, str):
                schema = json.loads(self.guide)
            else:
                schema = self.guide
        
            self.schema = guidance.json(schema=schema, temperature=0.0)
            self.serialized_grammar = self.schema.ll_serialize()
        elif self.mode.lower() in ["regex", "choice"]:
            self.serialized_grammar = guidance.gen(regex=self.guide, temperature=0.0).ll_serialize()
        elif self.mode.lower() == "grammar":
            # TODO: what is the correct format for grammar?
            self.serialized_grammar = self.guide

        t0 = time.time()
        if "guidance_tokenizer" not in self.metadata:
            self.metadata["guidance_tokenizer"] = TransformersTokenizer(
                model=self.tokenizer.name_or_path, transformers_tokenizer=self.tokenizer
            )        
        self.guidance_tokenizer = self.metadata["guidance_tokenizer"]
        print("GuidanceTokenizer init time:", time.time() - t0)

        t0 = time.time()
        if "ll_tokenizer" not in self.metadata:            
            self.metadata["ll_tokenizer"] = llguidance.LLTokenizer(
                llguidance.TokenizerWrapper(self.guidance_tokenizer)
            )
        self.ll_tokenizer = self.metadata["ll_tokenizer"]
        print("LLTokenizer init time:", time.time() - t0)
        
        t0 = time.time()
        self.ll_interpreter = llguidance.LLInterpreter(
            self.ll_tokenizer,
            json.dumps(self.serialized_grammar),
            enable_backtrack=False,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )
        print("LLInterpreter init time:", time.time() - t0)
        
        self.initialized = True

    def __call__(
        self,
        prompt_tokens_ids: List[int],
        past_tokens_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        self.initialize()

        if self.is_stopped:
            return logits

        try:
            if len(past_tokens_ids) == 0:
                self.ll_interpreter.process_prompt(prompt_tokens_ids)

            if self.new_sampling:
                backtrack, ff_tokens = self.ll_interpreter.post_process(
                    past_tokens_ids[-1]
                )
                if len(ff_tokens) > 0 and backtrack == 0:
                    ff_tokens = ff_tokens[1:]
                self.pending_ff_tokens.extend(ff_tokens)
                self.new_sampling = False

                # backtrack is disabled by default
                # assert backtrack == 0

            if len(self.pending_ff_tokens) > 0:
                ff_token = self.pending_ff_tokens.pop(0)
                masked_logits = torch.zeros_like(
                    logits, dtype=logits.dtype, device=logits.device
                )
                masked_logits[ff_token] = 200.0
                return masked_logits

            mask, resp = self.ll_interpreter.mid_process()
            # if mask is None:
            #     self.ll_interpreter.post_process(None)

            r = LLInterpreterResponse.model_validate_json(resp)
            # response = r.progress.to_engine_call_response()

            if r.stop:
                mask = torch.zeros_like(
                    logits, dtype=logits.dtype, device=logits.device
                )
                mask[self.guidance_tokenizer.eos_token_id] = 200.0
                self.is_stopped = True
            elif mask is None:
                mask = (
                    torch.zeros_like(logits, dtype=logits.dtype, device=logits.device)
                )
            else:
                mask = np.frombuffer(mask, dtype=np.uint8)
                mask = torch.tensor(mask, dtype=logits.dtype, device=logits.device)

            if mask.shape[0] != logits.shape[0]:
                # Some model has extra tokens that are not in the vocabulary
                mask = torch.cat(
                    [
                        mask,
                        torch.ones(
                            logits.shape[0] - mask.shape[0],
                            device=logits.device,
                        )
                        * 200.0,
                    ]
                )

            masked_logits = logits + mask
            self.new_sampling = True
        except Exception as e:
            print("Error in GuidanceJsonLogitsProcessor: ", e)
            raise e

        return masked_logits
