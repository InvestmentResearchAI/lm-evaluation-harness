import torch
import lit_gpt
from typing import Optional, List, Tuple, Any
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model


@register_model("litgpt")
class LitGPT(LM):
    def __init__(
        self,
        model: lit_gpt.GPT,
        tokenizer: lit_gpt.Tokenizer,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        max_length: int = None,
        top_k: int = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = model.device
        self._config = model.config
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_length  = max_length or model.max_seq_length
        self.top_k = top_k

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls()

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return self._device

    @property
    def config(self):
        return self._config
    
    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id
    
    def tok_encode(
        self, string: str, left_truncate_len=None
    ) -> List[int]:
        encoding = self.tokenizer.encode(string).to(torch.int64)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding[1:]
    
    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)

        # whole_enc = self.tok_encode(context + continuation)
        # context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                context_enc, continuation_enc = (
                    [self.eot_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
            new_reqs.append(((context, continuation), context_enc, continuation_enc))
        return self._loglikelihood_tokens(new_reqs)
        
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]]
    ) -> List[Tuple[float, bool]]:
        # NOTE: LitGPT doesn't implement batched inference, so we won't either until they do
        
        res = []
        for key, context_enc, continuation_enc in requests:
            # Calculate log likelihood for continuation tokens
            inp = torch.cat((context_enc, continuation_enc))[-(self.max_length + 1) : -1].to(self.device)
            (inplen,) = inp.shape
            input_pos = torch.arange(0, inplen, device=self.device)
            logits = self._model(inp.view([1, -1]), input_pos)
            logits = torch.nn.functional.log_softmax(logits, dim=-1)[0]
            
            contlen = len(continuation_enc)
            logits = logits[inplen - contlen : inplen]
            logits = logits.unsqueeze(0)
            
            # Check if per-token argmax is exactly equal to continuation
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = continuation_enc.unsqueeze(0)  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            
            logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
            
            # Answer: (log prob, is-exact-match)
            answer = (float(logits.sum()), bool(max_equal))
            res.append(answer)
            self.cache_hook.add_partial("loglikelihood", key, answer)
        
        return res
