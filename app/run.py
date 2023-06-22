from copy import deepcopy
import json
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any, Dict

import pandas as pd
import pytorch_lightning as pl
import typer
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from transformers import PreTrainedTokenizer

from app.datasets.paraphrase_data_module import ParaphraseDataModule
from app.model.nlg_lightning_module import NLGLightningModule
from app.model.nlg_lightning_module_config import NLGLightningModuleConfig
from app.reinforcement.train_reinforcement import train_reinforcement
from app.utils import CONFIGS_DIR_PATH, WANDB_CACHE_PATH, load_config, merge_dicts
import app.rankers.accuracy_ranker


typer_app = typer.Typer()


@typer_app.command()
def train_fresh(
    name: str = typer.Option(..., "--name", "-n"),
    trainer_config_path: Path = typer.Option(CONFIGS_DIR_PATH / "trainer_config.yaml", "--trainer-config", "-t"),
):
    app.rankers.accuracy_ranker.SACC_MODELS = {}
    model_config_dict: Dict[str, Any] = load_config(CONFIGS_DIR_PATH / name / "model_config.yaml")
    data_module_config_dict: Dict[str, Any] = load_config(CONFIGS_DIR_PATH / name / "data_module_config.yaml")
    trainer_config_dict: Dict[str, Any] = load_config(trainer_config_path)

    model_config: NLGLightningModuleConfig = NLGLightningModuleConfig.from_dict(model_config_dict)
    model: NLGLightningModule = NLGLightningModule(model_config)
    combined_configs = merge_dicts([model_config_dict, trainer_config_dict, data_module_config_dict])
    wandb_run = WandbLogger(project="nlg_baseline", log_model=True, group="train", config=combined_configs)
    data_module = get_data_module(data_module_config_dict, wandb_run=wandb_run, tokenizer=model.tokenizer)

    trainer = get_trainer(wandb_run, trainer_config_dict)
    trainer.fit(model, datamodule=data_module)


@typer_app.command()
def train_fresh_and_predict(
    name: str = typer.Option(..., "--name", "-n"),
    trainer_config_path: Path = typer.Option(CONFIGS_DIR_PATH / "trainer_config.yaml", "--trainer-config", "-t"),
    output_path: Path = typer.Option("/mnt/data/jpawlowski/predictions/", "--output-path", "-o"),
    split: str = typer.Option("test", "--split"),
    language: str = typer.Option(..., "--language"),
    run_name: str = typer.Option(None, "--run-name"),
):
    run_name = run_name or name
    app.rankers.accuracy_ranker.SACC_MODELS = {}
    model_config_dict: Dict[str, Any] = load_config(CONFIGS_DIR_PATH / name / "model_config.yaml")
    data_module_config_dict: Dict[str, Any] = load_config(CONFIGS_DIR_PATH / name / "data_module_config.yaml")
    trainer_config_dict: Dict[str, Any] = load_config(trainer_config_path)

    model_config: NLGLightningModuleConfig = NLGLightningModuleConfig.from_dict(model_config_dict)
    model: NLGLightningModule = NLGLightningModule(model_config)
    combined_configs = merge_dicts([model_config_dict, trainer_config_dict, data_module_config_dict])
    wandb_run = WandbLogger(project="nlg_baseline", log_model=True, group="train_and_predict", config=combined_configs, name=run_name)
    data_module = get_data_module(data_module_config_dict, wandb_run=wandb_run, tokenizer=model.tokenizer)

    trainer = get_trainer(wandb_run, trainer_config_dict)
    trainer.fit(model, datamodule=data_module)

    print(f"Making prediction for artifact from run {run_name}")
    # needs to be done manaully as for training it is always "right"
    if model.is_decoder_only:
        model.tokenizer.padding_side = "left"

    sgd_data_config_path = Path(f"{split}_sgd_config_{language}.yaml")
    sgd_data_module_config_dict: Dict[str, Any] = load_config(CONFIGS_DIR_PATH / sgd_data_config_path)

    sgd_data_module_config_dict["is_decoder_only"] = model.is_decoder_only
    sgd_data_module = get_data_module(sgd_data_module_config_dict, wandb_run=wandb_run, tokenizer=model.tokenizer)

    output_path = (Path(output_path) / run_name).with_suffix(".json")
    predict_with_model(model, wandb_run=wandb_run, trainer=trainer, data_module=sgd_data_module, output_path=output_path, language=language, artifact_name=run_name, artifact_type="dataset")


@typer_app.command()
def predict_sgd(
    checkpoint_name: str = typer.Option(..., "--checkpoint", "-c"),
    trainer_config_path: Path = typer.Option(CONFIGS_DIR_PATH / "trainer_config.yaml", "--trainer_config", "-t"),
    output_path: Path = typer.Option("/mnt/data/jpawlowski/predictions/", "--output-path", "-o"),
    split: str = typer.Option("test", "--split"),
    language: str = typer.Option(..., "--language"),
):
    data_config_path = Path(f"{split}_sgd_config_{language}.yaml")
    trainer_config_dict: Dict[str, Any] = load_config(trainer_config_path)
    data_module_config_dict: Dict[str, Any] = load_config(CONFIGS_DIR_PATH / data_config_path)

    run_name: str = wandb.Api().artifact(checkpoint_name).logged_by().name
    run_name = f"{run_name}-{split}-sgd-predict"
    print(f"Making prediction for artifact from run {run_name}")

    wandb_run: WandbLogger = WandbLogger(project="nlg_baseline", log_model=True, group="predict", name=run_name)
    artifact: wandb.Artifact = wandb_run.use_artifact(checkpoint_name)
    
    artifact_dir = artifact.download(root=WANDB_CACHE_PATH)
    model_path = Path(artifact_dir) / "model.ckpt"
    model = NLGLightningModule.load_from_checkpoint(model_path)

    # needs to be done manaully as for training it is always "right"
    if model.is_decoder_only:
        model.tokenizer.padding_side = "left"

    data_module_config_dict["is_decoder_only"] = model.is_decoder_only
    data_module = get_data_module(data_module_config_dict, wandb_run=wandb_run, tokenizer=model.tokenizer)

    trainer: pl.Trainer = get_trainer(wandb_run, trainer_config_dict=trainer_config_dict)

    output_path = (Path(output_path) / run_name).with_suffix(".json")
    predict_with_model(model, wandb_run=wandb_run, trainer=trainer, data_module=data_module, output_path=output_path, language=language, artifact_name=run_name, artifact_type="dataset")


def predict_with_model(model: NLGLightningModule, wandb_run: WandbLogger, trainer, data_module, output_path, language: str, artifact_name="predictions", artifact_type="predictions"):
    if language == "en":
        sacc_name = "sacc_flan_t5_large"
    elif language == "pl":
        sacc_name = "sacc_flan_t5_large_pl"
    app.rankers.accuracy_ranker.SACC_MODELS = app.rankers.accuracy_ranker.load_sacc_models(names=[sacc_name], device="cuda:0")
    model.config.sacc_name_for_correction_loop = sacc_name

    batches_predictions = trainer.predict(model, datamodule=data_module)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = defaultdict(list)
    for batch_predictions in batches_predictions:
        for dl_name, values in batch_predictions.items():
            predictions[dl_name].extend(values)

    with output_path.open("w", encoding='utf8') as output_file:
        json.dump(predictions, output_file, ensure_ascii=False, indent=4)
    # wandb_run._experiment.save(str(output_path), policy="now")

    artifact = wandb.Artifact(artifact_name, type=artifact_type, incremental=False)
    artifact.add_file(str(output_path), name="predictions.json")
    for dl_name, dl_predictions in predictions.items():
        dl_predictions = [
            {k: str(v) for k, v in entry.items() if k != "metadata"}
            for entry in dl_predictions
        ]
        df = pd.DataFrame(dl_predictions)
        table = wandb.Table(dataframe=df, dtype=str)
        artifact.add(table, name=dl_name)
    wandb_run._experiment.log_artifact(artifact)


@typer_app.command()
def interactive(
    checkpoint_name: str = typer.Option(..., "--checkpoint", "-c"),
):
    artifact_dir = wandb.Api().artifact(checkpoint_name).download(root=WANDB_CACHE_PATH)
    model_path = Path(artifact_dir) / "model.ckpt"
    model = NLGLightningModule.load_from_checkpoint(model_path)
    model.config.select_best_response_method = "first"

    while True:
        sentence = input("Input your sentence: ")
        if model.is_decoder_only:
            sentence = model.tokenizer.bos_token + sentence + model.tokenizer.sep_token
        tokenized = model.tokenizer(
            sentence,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        output = model.generate(input_ids, attention_mask, metadata=[])
        decoded_output = model.batch_decode(output, skip_special_tokens=False, remove_input=False)
        pprint(decoded_output)


def get_trainer(wandb_run, trainer_config_dict: Dict[str, Any]) -> pl.Trainer:
    wandb_experiment = wandb_run.experiment
    checkpoint_path = Path(
        trainer_config_dict["checkpoint_path"]) / f"id={wandb_experiment.id}-name={wandb_experiment.name}"
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor="dev_sgd_loss", mode="min", save_top_k=1,
                                          save_last=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=trainer_config_dict["devices"],
        max_epochs=trainer_config_dict["max_epochs"],
        callbacks=[checkpoint_callback, TQDMProgressBar()],
        logger=wandb_run,
        max_steps=trainer_config_dict["max_steps"]
    )
    return trainer


def get_data_module(data_module_config_dict: Dict[str, Any], wandb_run: WandbLogger, tokenizer: PreTrainedTokenizer):
    data_module = ParaphraseDataModule(
        **data_module_config_dict,
        wandb_run=wandb_run,
        tokenizer=tokenizer
    )
    return data_module


if __name__ == "__main__":
    typer_app()
