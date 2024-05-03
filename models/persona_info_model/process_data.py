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


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    encoded_dialogues = [incorporate_persona_info_into_dialogue(dialogue, tokenizer)
                         for dialogue in read_data()["dialogue"]]
    dataset = Dataset.from_dict({"dialogues": encoded_dialogues})

    print(dataset)
