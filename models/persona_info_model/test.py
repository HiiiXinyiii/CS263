import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from process_data import *



def generate_response(input_text, model, tokenizer, max_length=1000):
    encoded_input = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(encoded_input, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def test():
    conversations = combine(mode="test", json_path="data/val_spc_dataset.json", save_path=None)

    tokenizer = GPT2Tokenizer.from_pretrained('savings/finetuned_gpt2')
    model = GPT2LMHeadModel.from_pretrained('savings/finetuned_gpt2')

    # Generate response
    responses = []
    len_conversations = len(conversations)
    for i_idx, i_conversation in enumerate(conversations):
        print(f"Testing on {i_idx + 1}/{len_conversations}")
        ans = generate_response(i_conversation, model, tokenizer)
        responses.append(ans)
    
    # Write into file
    with open("savings/test_result.txt", 'w', encoding='utf-8') as file:
        for i_response in responses:
            file.write(i_response + "\n")


if __name__ == "__main__":
    test()


