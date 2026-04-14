# Fine-Tuning Pipeline

## What Fine-Tuning Does
Takes a pretrained model and continues training on domain-specific data.
The model learns your style, terminology, and desired behavior.

## When to Fine-Tune vs Prompt Engineer

| Situation | Use |
|---|---|
| Need consistent style/tone | Fine-tuning |
| Need domain-specific knowledge | Fine-tuning |
| Few examples available | Prompt engineering |
| Task changes frequently | Prompt engineering |
| Budget is limited | Prompt engineering first |
| Response format must be exact | Fine-tuning |

## Data Format (JSONL)
Each line is a JSON object with a messages array:
- system: defines the assistant's role
- user: the question or instruction
- assistant: the expected response

## LoRA — Why It Makes Fine-Tuning Affordable
LoRA (Low-Rank Adaptation) freezes most model weights and trains only
two small matrices added to attention layers. This reduces trainable
parameters from billions to millions — making fine-tuning possible
on consumer hardware.

## Files
- data_prep.py — generates and cleans Q&A pairs
- format_dataset.py — formats and saves JSONL datasets
- train_config.json — training hyperparameters
- data/train.jsonl — training split (auto-generated)
- data/val.jsonl — validation split (auto-generated)

## Running the Pipeline
python -m finetuning.format_dataset              # seeds only (fast)
python -m finetuning.format_dataset --build      # full pipeline with generation
python -m finetuning.format_dataset --validate finetuning/data/train.jsonl
python -m finetuning.format_dataset --sample finetuning/data/train.jsonl