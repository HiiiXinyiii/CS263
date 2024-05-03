import json
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader

from dataloader import PersonaDataset


class PersonaChatbot(nn.Module):
    def __init__(self, base_model, tokenizer, device):
        super(PersonaChatbot, self).__init__()

        self.device = device

        self.base_model = base_model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = 512
        self.persona_num_embeddings = 1
        self.persona_embedding_dim = self.base_model.config.hidden_size

        # The whole embeddings just represent one persona
        self.persona_embedding_layer = torch.nn.Embedding(num_embeddings=self.persona_num_embeddings, embedding_dim=self.persona_embedding_dim)
        self.fc = nn.Linear(self.base_model.config.hidden_size + self.persona_embedding_dim, self.max_length)
        # self.fc = nn.Linear(self.base_model.config.hidden_size, self.max_length)

    def forward(self, user_text):
        text_token = self.tokenizer(user_text,
                                    max_length=self.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt')

        input_ids = text_token['input_ids'].to(self.device)
        attention_mask = text_token['attention_mask'].to(self.device)
        # token_type_ids = text_token['token_type_ids'].squeeze(0)

        # Get persona embeddings
        persona_indices = torch.randint(0, self.persona_num_embeddings, (input_ids.size(0),), device=self.device)
        persona_embeddings = self.persona_embedding_layer(persona_indices)
        # Get outputs from the base BERT model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Combine BERT outputs with persona embeddings
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat((last_hidden_state, persona_embeddings), dim=1)
        # combined = last_hidden_state + persona_embeddings

        # Final feed-forward layer
        logits = self.fc(combined)

        return logits

    def train_model(self, data_loader, optimizer, criterion, epochs=1):
        self.train()

        for epoch in range(epochs):
            for idx, (user_text, bot_text) in enumerate(data_loader):
                logits = self.forward(user_text)

                labels = self.tokenizer(bot_text,
                                         max_length=self.max_length,
                                         padding='max_length',
                                         truncation=True,
                                         return_tensors='pt')['input_ids']

                loss = criterion(logits, labels.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Text {idx}, Criterion: {loss.item()}")

    def save_model(self, file_path="savings/model.cpt"):
        # torch.save(self.state_dict(), file_path)
        torch.save(self, file_path)



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

    model.train_model(data_loader, optimizer, criterion, epochs=10)

    model.save_model()
