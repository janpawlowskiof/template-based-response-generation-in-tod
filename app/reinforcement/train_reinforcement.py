import pickle
import random

import torch
import wandb
from tqdm import tqdm
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed

from app.datasets.paraphrase_data_module import ParaphraseDataModule
from app.model.nlg_lightning_module import NLGLightningModule
from app.rankers.accuracy_ranker import calculate_slots_accuracies


sample_model_inputs = [
    "Znanazłem taki apartament: W obiekcie są 4 sypialnie. Miesięczny czynsz wynosi 4800 USD dolarów. Nieruchomość znajduje się pod adresem 1580 Clayton Road # 1. Obiekt posiada 3 łazienki. Obiekt nazywa się Casa Pino Condos."
]


def train_reinforcement(nlg_module: NLGLightningModule, ref_nlg_module: NLGLightningModule, data_module: ParaphraseDataModule):
    nlg_module = nlg_module.train()
    tokenizer = nlg_module.tokenizer
    dataloader = data_module.train_dataloader()
    print(f"\tUsing batch_size={dataloader.batch_size}")

    # TODO: batch_size must be set properly here or else???
    config = PPOConfig(
        model_name="test_nlg_model", 
        learning_rate=5e-6,
        batch_size=dataloader.batch_size, 
        log_with="wandb",
        ppo_epochs=4,
        horizon=2000,
        mini_batch_size=1,
        init_kl_coef=0.2,
        # vf_coef=0.01
    )
    score_fn = calculate_slots_accuracies
    # score_fn = score_sentiment

    set_seed(config.seed)
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(nlg_module.model)
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(ref_nlg_module.model)

    generation_kwargs = {
        "top_k": 0.0, "top_p": 1.0, "do_sample": True, "max_new_tokens": 256
        # "top_k": 0.0, "top_p": 1.0, "do_sample": True, "eos_token_id": -1, "max_new_tokens": 16
        # "do_sample": False, "num_beams": 4, "eos_token_id": 1, "max_new_tokens": 256
    }

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    ppo_trainer.dataloader = dataloader

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    try:
        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            input_tensors = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)

            # with torch.no_grad():
                # bs = input_tensors.shape[0]
                # num_return_sequences = 8
                # context_response_tensors = ref_model.generate(
                #     input_ids=input_tensors, 
                #     attention_mask=attention_masks, 
                #     num_return_sequences=num_return_sequences, 
                #     **generation_kwargs
                # ).detach().reshape([bs, num_return_sequences, -1])
                
                # context_response_texts = [
                #     tokenizer.batch_decode(sequences=generated_sequences_for_one_input, skip_special_tokens=True)
                #     for generated_sequences_for_one_input in context_response_tensors
                # ]
                # context_scores: List[Dict[str, torch.Tensor]] = []
                # for generated_sequences_for_one_input, entry_metadata in zip(context_response_texts, batch["metadata"], strict=True):
                #     context_scores_for_one_input = {
                #         text: score_fn(text, used_slots=entry_metadata["used_slots"], device=device)
                #         for text in generated_sequences_for_one_input
                #     }
                #     context_scores.append(context_scores_for_one_input)

            input_texts = tokenizer.batch_decode(sequences=input_tensors, skip_special_tokens=True)
            
            # ppo_trainer.model = ppo_trainer.model.eval()
            # ppo_trainer.model.eval()
            
            response_tensors = [
                ppo_trainer.generate(
                    query, attention_mask=attention_mask.unsqueeze(0), **generation_kwargs
                ).squeeze().detach()
                for query, attention_mask in zip(input_tensors, attention_masks, strict=True)
            ]
            # response_texts = tokenizer.batch_decode(sequences=response_tensors, skip_special_tokens=True)
            response_texts = [tokenizer.decode(r[1:].squeeze(), skip_special_tokens=True) for r in response_tensors]
            print(response_texts[:8])

            sample_model_input_ids = [
                tokenizer(
                    x, max_length=128, padding="max_length", truncation=True, return_tensors="pt"
                )["input_ids"].squeeze(0).to(device)
                for x in sample_model_inputs
            ]
            
            sample_model_outputs = [
                ppo_trainer.generate(query, **generation_kwargs).squeeze().detach()
                for query in sample_model_input_ids
            ]
            decoded_sample_model_outputs = tokenizer.batch_decode(sequences=sample_model_outputs, skip_special_tokens=True)
            wandb.log({
                "decoded_text": wandb.Table(data=[[i, o] for i, o in zip(input_texts, response_texts)], columns=["input", "decoded_texts"]), 
                "sample_decoded_text": wandb.Table(data=[[i, o] for i, o in zip(sample_model_inputs, decoded_sample_model_outputs)], columns=["input", "sample_decoded_texts"])
            })

            scores = [
                score_fn(text, used_slots=entry_metadata["used_slots"], device=device)
                for text, entry_metadata in zip(response_texts, batch["metadata"], strict=True)
            ]
            # rewards = []
            # for score, context_scores_for_one_input in zip(scores, context_scores, strict=True):
            #     reward = sum(score >= context_score for context_score in context_scores_for_one_input.values())
            #     reward = reward / len(context_scores_for_one_input)
            #     reward = torch.tensor(reward, device=device)
            #     # reward = torch.tensor(1.0, device=device)
            #     # reward = reward * 2 - 1
            #     rewards.append(reward)
            # NOTE: temporaty hack for testing
            # rewards = scores
            rewards = [torch.tensor(1.0, device=device) for score in scores]

            trimmed_input_tensors = [
                input_tensor[:attention_mask.nonzero()[-1] + 1]
                for input_tensor, attention_mask 
                in zip(input_tensors, attention_masks, strict=True)
            ]

            trimmed_response_tensors = [
                trim_tensor_to_eos_token(response_tensor, eos_token_id=ppo_trainer.model.config.eos_token_id)
                for response_tensor 
                in response_tensors
            ]

            # ppo_trainer.model = ppo_trainer.model.train()
            stats = ppo_trainer.step(trimmed_input_tensors, trimmed_response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            # ppo_trainer.model = ppo_trainer.model.eval()
            
    finally:
        filename = f"checkpoint{random.randint(0, 100000)}.pickle"
        print(f"dumping filename {filename}")
        with open(filename, "wb") as output_file:
            pickle.dump(model, output_file)


def trim_tensor_to_eos_token(tensor, eos_token_id):
    eos_tokens_positions = (tensor == eos_token_id).nonzero()
    if len(eos_tokens_positions) == 0:
        print("No eos token in generated response. This should not happen often.")
        return tensor
    print("EOS token in generated response.")    
    return tensor[:eos_tokens_positions[-1] + 1]
