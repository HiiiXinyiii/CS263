import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from dataloader import PersonaDataset
from model import PersonaChatbot


def train():
    torch.cuda.empty_cache()
    device = torch.device("cpu")        # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = PersonaChatbot(base_model, tokenizer, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()

    with open("data\spc_data.json", 'r', encoding="utf-8") as file:
        train_data = json.load(file)
    len_train_data = len(train_data)
    for i_idx in range(len_train_data):
        train_dataset = PersonaDataset(train_data[i_idx]['conversations'], tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        print(f"Training on {i_idx + 1}/{len_train_data} conversations")
        model.train_model(train_dataloader, optimizer, criterion)

        if i_idx % 100 == 0:
            model.save_model()

    model.save_model()


if __name__ == "__main__":
    train()


