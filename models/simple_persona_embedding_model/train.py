import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from dataloader import PersonaDataset
from model import PersonaChatbot


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    base_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open("data\spc_data.json", 'r', encoding="utf-8") as file:
        data = json.load(file)
    dataset = PersonaDataset(data[0]['conversations'], tokenizer)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = PersonaChatbot(base_model, tokenizer, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()

    model.train_model(data_loader, optimizer, criterion)

    model.save_model()


