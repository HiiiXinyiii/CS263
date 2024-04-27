from model import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer


if __name__ == "__main__":
    persona_info = "Elaine is a female student at UCLA from China. "
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    base_model = GPT2LMHeadModel.from_pretrained('gpt2')

    params = {
        "persona_info": persona_info,
        "tokenizer": tokenizer,
        "base_model": base_model
    }

    persona = Persona(params)

    input_text = "I love discussing computer science and AI."

    response = persona.inference(input_text)

    print("***", response, "****")






