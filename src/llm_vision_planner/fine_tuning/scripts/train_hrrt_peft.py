#!/usr/bin/env python3
"""Train a preference-conditioned HRRT waypoint planner with LoRA or QLoRA."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DISTILLATION_MODES = ("dss", "dss_scott")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_rows(rows, path: Path, distillation_mode: str):
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    modes = {row.get("distillation_mode", "dss") for row in rows}
    if modes != {distillation_mode}:
        raise ValueError(
            f"{path} contains distillation modes {sorted(modes)}, expected only {distillation_mode!r}"
        )


def tokenize_rows(rows, tokenizer, max_length):
    from datasets import Dataset

    encoded_rows = []
    for row in rows:
        prompt = row["prompt"]
        completion = row["completion"]
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        completion_ids = tokenizer(
            completion + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]
        input_ids = (prompt_ids + completion_ids)[:max_length]
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        if not any(label != -100 for label in labels):
            raise ValueError(f"{row.get('sample_id', 'sample')}: completion was truncated entirely")
        encoded_rows.append(
            {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
        )
    return Dataset.from_list(encoded_rows)


def compatible_training_arguments(**kwargs):
    from transformers import TrainingArguments

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    kwargs[evaluation_key] = "steps"
    return TrainingArguments(**kwargs)


def save_history(history, output_dir: Path):
    rows = [entry for entry in history if "loss" in entry or "eval_loss" in entry]
    with (output_dir / "loss_history.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("fine_tuning/datasets/hrrt_sft_train.csv"))
    parser.add_argument("--validation", type=Path, default=Path("fine_tuning/datasets/hrrt_sft_validation.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("fine_tuning/outputs/llama31_8b_hrrt_lora"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--distillation-mode", choices=DISTILLATION_MODES, default="dss")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qlora", action="store_true", help="Load the base model in 4-bit NF4.")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
    )

    train_rows = read_rows(args.train)
    validation_rows = read_rows(args.validation)
    validate_rows(train_rows, args.train, args.distillation_mode)
    validate_rows(validation_rows, args.validation, args.distillation_mode)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = tokenize_rows(train_rows, tokenizer, args.max_seq_length)
    validation_data = tokenize_rows(validation_rows, tokenizer, args.max_seq_length)

    model_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8),
        args=compatible_training_arguments(
            output_dir=str(args.output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            warmup_ratio=0.05,
            bf16=True,
            fp16=False,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            seed=args.seed,
            report_to="none",
        ),
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_history(trainer.state.log_history, args.output_dir)
    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            stream,
            indent=2,
        )
    print(f"Saved HRRT adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
