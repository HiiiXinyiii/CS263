import json
from datasets import Dataset
from transformers import GPT2Tokenizer


def read_data(path="data/test_data.json"):
    with open(path, "r") as file:
        data = json.load(file)

    return data


# Preprocess the dialogue data
def incorporate_persona_info_into_dialogue(dialogue, tokenizer):
    # Convert persona info to a string
    persona_info = " ".join([f"{key}: {value}" for key, value in dialogue["persona"].items()])
    # Concatenate the persona info with the input and the response
    concatenated_dialogue = persona_info + " " + dialogue["input"] + tokenizer.eos_token + dialogue["response"]

    return tokenizer(concatenated_dialogue).data


# 拼接对话上下文和回复
def combine(mode="train", json_path="data/test_data.json", save_path="savings/processed_conversations.txt"):
    """
    :param mode: "train" is get both user and bot; "test" is get user
    """
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    conversations = []
    for item in data:
        persona = item["user1_persona"] + " " + item["user2_persona"]
        for conv in item["conversations"]:
            if mode.lower() == "train":
                conversations.append(persona + " " + conv.get("user", "") + " " + conv.get("bot", ""))
            else:
                conversations.append(persona + " " + conv.get("user", ""))

    # Save path
    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(conv + '\n')

    return conversations
    

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    encoded_dialogues = [incorporate_persona_info_into_dialogue(dialogue, tokenizer)
                         for dialogue in read_data()["dialogue"]]
    dataset = Dataset.from_dict({"dialogues": encoded_dialogues})

    print(dataset)
