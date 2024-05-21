import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from dataloader import PersonaDataset
from model import PersonaChatbot


def generate_response(model, tokenizer, user_text):
    with torch.no_grad():
        logits = model(user_text)
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return predicted_text


def test(checkpoint="savings/model.cpt", save_result=None):
    torch.cuda.empty_cache()

    # Save the result of generation
    """
    The form:
    [
        {
            "conversations": [
                {
                    "user": "I hope your blog post goes well! What are you writing about?",
                    "bot": "Thank you! I'm writing about my favorite fashion trends from the 90s. I can't wait to share it with everyone."
                }
        }
    ]
    """
    res = []        

    # Load the model
    model = torch.load(checkpoint)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    with open("data/val_spc_dataset.json", 'r', encoding="utf-8") as file:
        val_data = json.load(file)

    for i_idx_conversation in range(len(train_data)):
        current_conversation = val_data[i_idx_conversation]['conversations']
        tmp_res = {"user1_persona": val_data[i_idx_conversation]["user1_persona"], 
                    "user2_persona": val_data[i_idx_conversation]["user2_persona"],
                    "conversations": []
                }
        # Iterate all the sentences in the conversation
        for j_idx_sentence in range(len(current_conversation))
            output_text = generate_response(model, tokenizer, current_conversation[j_idx_sentence]["user"])
            tmp_res["conversations"].append({"user": current_conversation[j_idx_sentence]["user"], "bot": output_text})

        res.append(tmp_res)

    # Save the result
    if save_result is not None:
        with open("save_result", "w") as json_file:
            json.dump(res, json_file, indent=4)

    return res


if __name__ == "__main__":
    val_result = test(checkpoint="savings/model.cpt", save_result="data/val_result.json")

