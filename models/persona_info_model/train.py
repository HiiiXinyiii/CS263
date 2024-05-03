from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from process_data import *
from model import Persona


# Persona Information
persona_info = "Elaine is a female student at UCLA from China. "
# Base model
base_model = GPT2LMHeadModel.from_pretrained('gpt2')
# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Data
encoded_dialogues = [incorporate_persona_info_into_dialogue(dialogue, tokenizer)
                     for dialogue in read_data()["dialogue"]]
dataset = Dataset.from_dict({"dialogues": encoded_dialogues})
# Define model
params = {
    "persona_info": persona_info,
    "tokenizer": tokenizer,
    "base_model": base_model
}
model = Persona(params)


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # The output directory for the model predictions and checkpoints
    num_train_epochs=3,              # Total number of training epochs
    per_device_train_batch_size=4,   # Batch size for training
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
)

# Initialize the Trainer
trainer = Trainer(
    model=model.base_model,
    args=training_args,
    train_dataset=dataset['dialogues']
)


# Train the model
trainer.train()


# Save the model and the tokenizer
model.base_model.save_pretrained('./savings/elaine_persona_model')
model.tokenizer.save_pretrained('./savings/elaine_persona_model')
