import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pdb

class LMClass(BaseLM):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model
        config = AutoConfig.from_pretrained(
            args.model, attn_implementation=args.attn_implementation
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)
        # 모델은 fp16으로 로드합니다.
        self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu', torch_dtype=torch.float16)
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

    @classmethod
    def from_pretrained(cls, save_dir, args=None):
        """
        저장된 디렉토리에서 모델과 토크나이저를 재로딩하는 클래스 메서드.
        args가 필요한 경우 전달하거나, 기본값으로 설정할 수 있습니다.
        """
        if args is None:
            # 필요한 인자들에 대해 기본 설정을 하거나, 별도 인자 처리가 필요함
            parser = argparse.ArgumentParser()
            parser.add_argument("--model", type=str, default=save_dir)
            parser.add_argument("--batch_size", type=int, default=8)
            parser.add_argument("--attn_implementation", type=str, default="eager")
            args = parser.parse_args([])
            args.model = save_dir
        instance = cls(args)
        # 저장된 디렉토리에서 모델과 토크나이저 불러오기
        instance.model = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.float16)
        instance.tokenizer = AutoTokenizer.from_pretrained(save_dir, use_fast=False, legacy=False)
        instance.seqlen = instance.model.config.max_position_embeddings
        instance.model.eval()
        return instance

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # end of text token
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: torch tensor of shape [batch, sequence]
        returns: logits tensor [batch, sequence, vocab]
        """
        with torch.no_grad():
            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(self._model_call(batch), dim=-1).cpu()
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )