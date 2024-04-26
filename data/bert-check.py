import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch


class PersonaDialogueDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.personas = dataframe['Persona'].tolist()
        self.bots = dataframe['bot_response'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.personas)

    def __getitem__(self, idx):
        persona = str(self.personas[idx])
        bot_response = str(self.bots[idx])
        text_pair = persona + " [SEP] " + bot_response
        encoding = self.tokenizer.encode_plus(
            text_pair,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

df = pd.read_csv('personality.csv')
df['bot_response'] = df['chat'].apply(
    lambda x: x.split('\n')[1] if len(x.split('\n')) > 1 else '')

dataset = PersonaDialogueDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=32)

predictions = []
model.eval()
with torch.no_grad():
    for batch in dataloader:
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        predictions = torch.argmax(outputs.logits, dim=1)
        print(predictions)


# New-Persona-New-Conversations.csv bert check
df = pd.read_csv('New-Persona-New-Conversations.csv')
df['bot_response'] = df['Best Generated Conversation'].apply(
    lambda x: x.split('\n')[1] if len(x.split('\n')) > 1 else '')

dataset = PersonaDialogueDataset(df, tokenizer)
dataset.personas = df['user 2 personas'].tolist()
dataloader = DataLoader(dataset, batch_size=32)

predictions = []
model.eval()
with torch.no_grad():
    for batch in dataloader:
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        predictions = torch.argmax(outputs.logits, dim=1)
        print(predictions)