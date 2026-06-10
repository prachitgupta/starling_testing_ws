#!/usr/bin/env python3
import argparse
import csv
import inspect
from pathlib import Path

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel


SCRIPT_DIR = Path(__file__).resolve().parent
FINE_TUNING_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET = FINE_TUNING_DIR / "datasets" / "rrt_expert_dataset.csv"
DEFAULT_OUTPUT = FINE_TUNING_DIR / "outputs" / "llama31_8b_rrt_lora"
DEFAULT_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def format_examples(examples, tokenizer):
    texts = []
    for messages_json in examples["messages"]:
        import json

        messages = json.loads(messages_json)
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return {"text": texts}


def training_arguments(**kwargs):
    strategy_key = "eval_strategy"
    if strategy_key not in inspect.signature(TrainingArguments.__init__).parameters:
        strategy_key = "evaluation_strategy"
    kwargs[strategy_key] = "steps"
    return TrainingArguments(**kwargs)


def sft_trainer(**kwargs):
    signature = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in signature and "tokenizer" in kwargs:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    return SFTTrainer(**{key: value for key, value in kwargs.items() if key in signature})


def save_loss_artifacts(log_history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "loss_history.csv"
    png_path = output_dir / "loss_curve.png"

    rows = [
        {
            "step": entry.get("step"),
            "epoch": entry.get("epoch"),
            "train_loss": entry.get("loss"),
            "eval_loss": entry.get("eval_loss"),
            "learning_rate": entry.get("learning_rate"),
        }
        for entry in log_history
        if "loss" in entry or "eval_loss" in entry
    ]
    if not rows:
        print("No loss logs were found; skipping loss plot.")
        return

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "epoch", "train_loss", "eval_loss", "learning_rate"])
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Wrote loss history to {csv_path}; install matplotlib to create {png_path}.")
        return

    train_points = [(row["step"], row["train_loss"]) for row in rows if row["train_loss"] is not None]
    eval_points = [(row["step"], row["eval_loss"]) for row in rows if row["eval_loss"] is not None]
    plt.figure(figsize=(8, 5))
    if train_points:
        plt.plot([point[0] for point in train_points], [point[1] for point in train_points], label="train loss")
    if eval_points:
        plt.plot([point[0] for point in eval_points], [point[1] for point in eval_points], marker="o", label="eval loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Fine-tuning Loss Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()
    print(f"Wrote loss history to {csv_path} and plot to {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.1-8B-Instruct on RRT expert paths with Unsloth LoRA.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--val-split-ratio", type=float, default=0.10)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true", help="Use QLoRA if GPU memory is tight.")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_dataset("csv", data_files=str(args.dataset), split="train")
    dataset = dataset.map(lambda examples: format_examples(examples, tokenizer), batched=True)
    split = dataset.train_test_split(test_size=args.val_split_ratio, seed=args.seed)

    trainer = sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_arguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=str(args.output_dir),
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    save_loss_artifacts(trainer.state.log_history, Path(args.output_dir))
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
