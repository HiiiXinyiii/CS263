from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
from torch import nn
import math
import numpy as np
from torch.utils.data import IterableDataset
from transformers import GPT2Tokenizer, GPT2Model, GPT2Tokenizer
import json

@dataclass
class MiniGPTConfig:
    path_to_data: Path = Path("train_fake_dataset.json")
    batch_size: int = 32
    num_layers: int = 8
    vocab_size: int = 50257
    embed_dim: int = (
        192 
    )
    feedforward_size: Optional[int] = (
        None  # hidden layer in the feedforward network, None sets it to 4*embed_dim
    )
    context_length: int = 192
    num_heads: int = 16
    weight_tie: bool = (
        True 
    )
    feedforward_dropout: float = 0.55
    attention_dropout: float = 0.55
    out_dropout: float = 0.55
    embed_dropout: float = 0.55
    learning_rate: float = 3e-4
    log_interval: int = 50
    save_path: Path = Path("models/minigpt/")
    save_iterations: int = 50
    to_log: bool = True
    max_iter: int = 500000
    to_clip_grad: bool = False
    gradient_clip: float = 0.99
    scheduler: bool = False

class SingleHeadAttention(nn.Module):
    """
    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)
    """

    def __init__(
        self,
        input_dim,
        output_key_query_dim=None,
        output_value_dim=None,
        dropout=0.1,
        max_len=256,
    ):
        super().__init__()

        self.input_dim = input_dim
        if output_key_query_dim:
            self.output_key_query_dim = output_key_query_dim
        else:
            self.output_key_query_dim = input_dim

        if output_value_dim:
            self.output_value_dim = output_value_dim
        else:
            self.output_value_dim = input_dim

        self.key = nn.Linear(in_features=self.input_dim, out_features=self.output_key_query_dim, bias=False)
        self.query = nn.Linear(in_features=self.input_dim, out_features=self.output_key_query_dim, bias=False)
        self.value = nn.Linear(in_features=self.input_dim, out_features=self.output_value_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1)
        self.register_buffer(
            "causal_mask", causal_mask
        )

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        causal_mask = self.causal_mask[:num_tokens, :num_tokens]

        x_key = self.key(x)
        x_query = self.query(x)
        x_value = self.value(x)
        score = x_query @ x_key.transpose(-2, -1) / math.sqrt(self.output_key_query_dim)
        score = score.masked_fill(causal_mask== 1, float('-inf'))
        attention = nn.functional.softmax(score, dim=-1)
        attention = self.dropout(attention)
        out = attention @ x_value

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        for i in range(num_heads):
            setattr(self, f'head_{i}', SingleHeadAttention(
                input_dim=self.input_dim,
                output_key_query_dim=self.head_dim,
                output_value_dim=self.head_dim,
                dropout=dropout,
                max_len=256
            ))
        self.out = nn.Linear(input_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = []

        for i in range(self.num_heads):
            head = getattr(self, f'head_{i}')
            head_outputs.append(head(x))

        concat = torch.cat(head_outputs, dim=-1)
        out = self.out(concat)
        out = self.dropout(out)
        return out

class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, feedforward_dim=None, dropout=0.1):
        super().__init__()
        if feedforward_dim is None:
            feedforward_dim = input_dim * 4

        self.fc1 = nn.Linear(input_dim, feedforward_dim, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(feedforward_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x_out0 = self.fc1(x)
        x_out1 = self.activation(x_out0)
        x_out2 = self.fc2(x_out1)
        out = self.dropout(x_out2)
        return out

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True) -> None:
        super().__init__()

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(tuple(self.normalized_shape)))
            self.beta = nn.Parameter(torch.zeros(tuple(self.normalized_shape)))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.var(input, dim=-1, keepdim=True, unbiased=False)
        out = (input - mu) / torch.sqrt(sigma + self.eps) * self.gamma + self.beta
        return out

class TransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feedforward_dim=None):
        super().__init__()
        self.norm1 = LayerNorm(input_dim)
        self.attention = MultiHeadAttention(input_dim=input_dim, num_heads=num_heads)
        self.norm2 = LayerNorm(input_dim)
        self.feedforward = FeedForwardLayer(input_dim=input_dim, feedforward_dim=feedforward_dim)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn = self.attention(x_norm)
        x_out1 = x + attn
        x_norm2 = self.norm2(x_out1)
        x_fwd = self.feedforward(x_norm2)
        x_out2 = x_out1 + x_fwd
        return x_out2

class PersonaEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained('gpt2').to(self.device)
        self.linear = nn.Linear(self.model.config.hidden_size, embed_dim)

    def forward(self, persona_text):
        # print(type(persona_text))
        inputs = self.tokenizer(persona_text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden_state = outputs.last_hidden_state.mean(dim=1)
        # return outputs.last_hidden_state.mean(dim=1)
        return self.linear(hidden_state)

class MiniGPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_embedding = nn.Embedding(config.vocab_size, config.embed_dim).to(self.device)
        self.positional_embedding = nn.Embedding(config.context_length, config.embed_dim).to(self.device)
        # self.persona_embedding = nn.Embedding(config.num_personas, config.embed_dim)
        self.embed_dropout = nn.Dropout(config.embed_dropout).to(self.device)
        self.persona_embedding_module = PersonaEmbedding(config.embed_dim).to(self.device)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    config.embed_dim, config.num_heads, config.feedforward_size
                )
                for _ in range(config.num_layers)
            ]
        ).to(self.device)
        self.prehead_norm = LayerNorm(config.embed_dim).to(self.device)
        self.head = nn.Linear(config.embed_dim, config.vocab_size).to(self.device)

        if config.weight_tie:
            self.head.weight = self.vocab_embedding.weight
        pos = torch.arange(0, config.context_length, dtype=torch.long).to(self.device)
        self.register_buffer("pos", pos, persistent=False)
        self.apply(self._init_weights)

    def forward(self, x, persona_text):
        x = x.to(self.device)
        persona_embed = self.persona_embedding_module(persona_text)
        # print(persona_embed.shape)
        batch_size, seq_len = x.shape
        # print(x.shape)

        x_embed = self.vocab_embedding(x) +self.positional_embedding(self.pos)[:seq_len,:].unsqueeze(0).repeat(batch_size,1,1)
        x_embed += persona_embed.unsqueeze(1).expand(-1, seq_len, -1)
        x_embed = self.embed_dropout(x_embed)
        for transformer_layer in self.transformer_layers:
            x_embed = transformer_layer(x_embed)
        x_norm = self.prehead_norm(x_embed)
        out = self.head(x_norm)
        return out

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module._get_name() == "fc2":
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, p, max_new_tokens=100):
        context = torch.tensor(context).unsqueeze(0).to(self.device)
        output_token = []
        for i in range(max_new_tokens):
            in_token = context[:, 48:] if context.size(1) >= 60 else context
            logits = self.forward(in_token,p)[:,-1,:]
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).to(self.device)
            output_token.append(next_token.item())
            context = torch.cat([context, next_token], dim=1)
        return context


class PersonaDataset(IterableDataset):
    def __init__(self, data_folder: Path, tokenizer: GPT2Tokenizer, context_length: int = 2):

        self.data_path = data_folder
        self.context_length = context_length
        self.data = self.load_data()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_data(self):
        with open(self.data_path, 'r') as file:
            data = json.load(file)

        dialogue_batches = []
        for entry in data:
            user2_persona = entry.get("persona", "No user2 persona")
            context = entry.get("conversation", [])

            dialogue_batches.append((user2_persona, context))

        return dialogue_batches

    def __iter__(self):
        for user2_persona, context in self.data:

            for i in range(len(context)):
                input_text = ""
                if i == 0:
                    pass
                elif i == 1:
                    for speaker, utterance in context[i-1].items():
                        input_text += f"{speaker}: {utterance} "
                else:
                    for speaker, utterance in context[i-2].items():
                        input_text += f"{speaker}: {utterance} "
                    for speaker, utterance in context[i-1].items():
                        input_text += f"{speaker}: {utterance} "

                target_text = input_text
                for speaker, utterance in context[i].items():
                    target_text += f"{speaker}: {utterance} "
                    break

            # for i in range(len(context) - (self.context_length + 1)):
            #     current_context = context[i:i + self.context_length]
            #     additional_user_speech = context[i + self.context_length].get('user', 'No question')
            #     target_text = context[i + self.context_length].get('bot', 'No response')
            #     target_text = f"Bot: {target_text}"
            #     context_str = " ".join(f"User: {conv.get('user', 'No question')} Bot: {conv.get('bot', 'No response')}" for conv in current_context)
            #     input_text = f"{context_str} User: {additional_user_speech}"
            #     target_text = input_text+target_text
                # Tokenize the input and target text with padding and attention mask
                encoding = self.tokenizer(
                    input_text,
                    truncation=True,
                    max_length=80,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(torch.long)
                padding_length = 80 - input_ids.shape[1]
                padded_input_ids = torch.cat([torch.full((1, padding_length), tokenizer.pad_token_id), input_ids], dim=1).squeeze(0)

                target_encoding = self.tokenizer(
                    target_text,
                    padding='max_length',
                    truncation=True,
                    max_length=80,
                    return_tensors="pt",
                )

                # input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0).to(torch.long)
                target_ids = target_encoding['input_ids'].squeeze(0).to(torch.long)

                yield padded_input_ids, target_ids, user2_persona


    def __len__(self):
        return len(self.data) - self.context_length


model = MiniGPT(MiniGPTConfig).to("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('model9_250.pth'))
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
# from einops import rearrange
import wandb

config = MiniGPTConfig
model = MiniGPT(config)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if config.to_log:
    wandb.init(project="dl2_proj8")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_dataset = PersonaDataset(
    config.path_to_data,
    tokenizer = tokenizer,
    context_length=2,
)
eval_dataset = PersonaDataset(
    "val_fake_dataset.json", tokenizer = tokenizer, context_length=2
)

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, pin_memory=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))


if not Path.exists(config.save_path):
    Path.mkdir(MiniGPTConfig.save_path, parents=True, exist_ok=True)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (x, y, z) in enumerate(dataloader):
            if i == 1000:
              break
            x, y, z = x.to(device), y.to(device), z
            outputs = model(x, z)
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / 1000

# Ensure dataset and dataloaders are correctly set up
print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Eval Dataset Size: {len(eval_dataset)}")

model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

best_val_loss = float('inf')

for epoch in range(10):
    model.train()
    running_loss = 0
    print(len(train_dataloader))
    for i, (x, y, z) in enumerate(train_dataloader):
        x, y, z= x.to(device), y.to(device), z
        # persona_embed = model.persona_embedding_module(z)
        outputs = model(x, z)
        loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % config.log_interval == 0:
            print(f"Epoch: {epoch+1}, Batch: {i}, Loss: {loss.item()}")
            if config.to_log:
                wandb.log({"train_loss": loss.item()})

        if i % 50 == 0:
            val_loss = evaluate(model, eval_dataloader)
            print(f"Validation Loss: {val_loss}")
            if config.to_log:
                wandb.log({"val_loss": val_loss})



        if i % config.save_iterations == 0:
            torch.save(model.state_dict(), f'{config.save_path}/model{epoch}_{i}.pth')
            print(f"Model saved at iter {i}")

avg_loss = running_loss / len(train_dataloader)
print(f"Average Loss after Epoch {epoch+1}: {avg_loss}")
