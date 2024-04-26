import json
from transformers import GPTNeoXTokenizerFast

tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20b')


def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def tokenize_pc(data, tokenizer):
    tokenized_data = []
    for item in data:
        history = "<History>"
        persona = f"<Persona> {item['persona']}"
        for convo in item['conversations']:
            user_text = f"<User> {convo['user']}"
            bot_text = f"<Bot> {convo['bot']}"
            conversation = f"{persona} {history} {user_text} {bot_text}"
            conversation_sequence = tokenizer.encode(conversation, truncation=True, max_length=2048)
            tokenized_data.append(conversation_sequence)
            history += f" {convo['user']} {convo['bot']} "
    return tokenized_data


def tokenize_spc(data, tokenizer):
    tokenized_data = []
    for item in data:
        history = "<History>"
        user1_persona = f"<Persona 1> {item['user1_persona']}"
        user2_persona = f"<Persona 2> {item['user2_persona']}"
        for convo in item['conversations']:
            user_text = f"<User> {convo['user']}"
            bot_text = f"<Bot> {convo['bot']}"
            conversation = f"{user1_persona} {user2_persona} {history} {user_text} {bot_text}"
            conversation_sequence = tokenizer.encode(conversation, truncation=True, max_length=2048)
            tokenized_data.append(conversation_sequence)
            history += f" {convo['user']} {convo['bot']} "
    return tokenized_data


# data = load_data('pc_data.json')
# tokenized_data = tokenize_pc(data, tokenizer)
#
# for encoded_input in tokenized_data[:5]:
#     print(encoded_input)
#     decoded_text = tokenizer.decode(encoded_input)
#     print(decoded_text)

data1 = load_data('spc_data.json')
tokenized_data1 = tokenize_spc(data1, tokenizer)

for encoded_input in tokenized_data1[:5]:
    print(encoded_input)
    decoded_text1 = tokenizer.decode(encoded_input)
    print(decoded_text1)
#

# total_numbers = sum(len(sublist) for sublist in tokenized_data1)
# print(total_numbers)