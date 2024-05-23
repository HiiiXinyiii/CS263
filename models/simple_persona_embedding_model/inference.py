import torch
from model import PersonaChatbot
from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    model = torch.load("savings\model.cpt")
    tokenizer = model.tokenizer  # BertTokenizer.from_pretrained('bert-base-uncased')

    model.eval()

    user_text = "Hi How are you"
    output = model(user_text)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)


    inputs = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True)
    output = model.base_model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )
    output_text = tokenizer.decode(output, skip_special_tokens=True)

    print(output_text)
