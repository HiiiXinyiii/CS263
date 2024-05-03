import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


# Assume the dataset follow the spc format
class PersonaDataset(Dataset):
    def __init__(self, conversations, tokenizer):
        # Cancel a warning
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

        self.conversations = conversations
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.conversations)

    # Retrieve the conversation
    def __getitem__(self, idx):
        return self.conversations[idx]['user'], self.conversations[idx]['bot'],

        """
        # Encode the conversation using the tokenizer
        encoded_input = self.tokenizer.encode_plus(
            self.conversations[idx]['user'],
            self.conversations[idx]['bot'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Extract and return the relevant fields
        input_ids = encoded_input['input_ids'].squeeze(0)
        attention_mask = encoded_input['attention_mask'].squeeze(0)
        token_type_ids = encoded_input['token_type_ids'].squeeze(0)

        return input_ids, attention_mask, token_type_ids
        """


if __name__ == "__main__":
    with open('data\spc_data.json', 'r', encoding="utf-8") as file:
        data = json.load(file)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = PersonaDataset(data[0]['conversations'], tokenizer, max_length=512)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


