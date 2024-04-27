import torch


class Persona:
    def __init__(self, params):
        self.persona_info = params["persona_info"]
        self.base_model = params["base_model"]
        self.tokenizer = params["tokenizer"]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def inference(self, input_text):
        encoded_input = self.tokenizer.encode(self.persona_info + input_text, return_tensors='pt')
        attention_mask = torch.ones(encoded_input.shape, dtype=torch.long)
        attention_mask[encoded_input == self.tokenizer.pad_token_id] = 0
        output = self.base_model.generate(
            encoded_input,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=100
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response









