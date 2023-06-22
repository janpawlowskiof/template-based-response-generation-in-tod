from transformers import MT5ForConditionalGeneration, MT5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from app.utils import CACHE_PATH


def get_pretrained_model(model_type: str = "mt5", model_path: str = "google/mt5-small") -> MT5ForConditionalGeneration:
    match model_type:
        case "mt5":
            return MT5ForConditionalGeneration.from_pretrained(model_path, cache_dir=CACHE_PATH)
        case "t5":
            return AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=CACHE_PATH)
        case "gpt" | "mgpt":
            return AutoModelForCausalLM.from_pretrained(model_path, cache_dir=CACHE_PATH)
        case _:
            raise ValueError(f"Unknown model type {model_type}!")



def get_tokenizer(tokenizer_type: str = "mt5", tokenizer_path: str = "google/mt5-small"):
    match tokenizer_type:
        case "mt5":
            return MT5Tokenizer.from_pretrained(tokenizer_path, cache_dir=CACHE_PATH, truncation_side="left")
        case "t5":
            return AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=CACHE_PATH, truncation_side="left")
        case "gpt" | "mgpt":
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=CACHE_PATH, truncation_side="left")
            tokenizer.add_special_tokens(
                {
                    "pad_token": "[PAD]",
                    "sep_token": "[SEP]",
                    "bos_token": "[BOS]",
                    "eos_token": "[EOS]",
                }
            )
            # for training this needs to be right or else model does not learn to properly use eos token!
            # set this manually to "left" for generation only
            tokenizer.padding_side = "right"
            return tokenizer
        case _:
            raise ValueError(f"Unknown tokenizer type {tokenizer_type}!")
