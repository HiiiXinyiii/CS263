import json
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from process_data import *


def train():
    # Get conversation data
    conversation = combine(mode="train", json_path="data/train_spc_dataset.json", save_path="savings/processed_conversations.txt")

    # Load model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load data
    train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path="savings/processed_conversations.txt",
            block_size=512
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Params
    training_args = TrainingArguments(
        output_dir="savings",
        overwrite_output_dir=True,
        num_train_epochs=300,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
    )

    # Create trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # train model
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained('savings/finetuned_gpt2')
    tokenizer.save_pretrained('savings/finetuned_gpt2')


if __name__ == "__main__":
    train()
