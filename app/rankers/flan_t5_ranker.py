from typing import Dict, List
from app.utils import preprocessor_slot_description_to_value
import torch

from transformers import AutoTokenizer

from app.model.pretrained_models import get_pretrained_model, get_tokenizer


class FlanT5Sacc:
    def __init__(self, size: str, device) -> None:
        self.name = f"flan-t5-{size}"
        self.device = device
        model_path = f"google/flan-t5-{size}"
        self.model = get_pretrained_model(model_type="t5", model_path=model_path).eval().to(self.device)
        self.tokenizer: AutoTokenizer = get_tokenizer(tokenizer_type="t5", tokenizer_path=model_path)
        self.generation_config = {
            "max_new_tokens": 16,
        }

    def calculate(self, utterance: str, metadata: Dict) -> Dict[str, bool]:
        with torch.no_grad():
            parts = metadata["parts"]
            outputs = self.are_parts_correct_for_utterance(utterance, parts=parts)
            return {
                part["slot"]: output
                for part, output in zip(parts, outputs)
            }

    def are_parts_correct_for_utterance(self, utterance: str, parts: List[Dict]) -> List[bool]:
        utterance = self.preprocess_utterance(utterance)
        prompts = [
            self.get_prompt(utterance=utterance, part=part)
            for part in parts
        ]
        tokenized = self.tokenizer(
            prompts,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        output_tensor = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **self.generation_config)
        outputs = self.tokenizer.batch_decode(sequences=output_tensor, skip_special_tokens=True)
        assert len(outputs) == len(parts)
        outputs = [self.model_output_to_bool(output) for output in outputs]
        return outputs

    def get_prompt(self, utterance: str, part: Dict) -> str:
        template = self.preprocess_template(part["template"])
        if part["act"] in ["INFORM", "OFFER", "CONFIRM"]:
            return f'{utterance}\nCan we infer the following?\n{template}\n\n["yes", "no"]'
        elif part["act"] == "REQUEST":
            return f'{utterance}\nDo we ask the following question?\n"{template}"\n\["yes", "no"]'
        else: 
            raise RuntimeError(f"Unknown act {part['act']}")

    def preprocess_utterance(self, utterance: str) -> str:
        return utterance

    def preprocess_template(self, template: str) -> str:
        return preprocessor_slot_description_to_value(template)

    def model_output_to_bool(self, output: str) -> bool:
        if output == "yes":
            return True
        elif output == "no":
            return False
        else:
            print(f"flan model returned '{output}' as a response")
            return False
