from typing import List, Dict

import numpy as np
import torch

def _get_ppl(
    model,
    tokenizer,
    history: str,
    response: str,
):
    device = model.device
    history_tokens = tokenizer.tokenize(history)
    response_tokens = tokenizer.tokenize(response)
    
    input = tokenizer.convert_tokens_to_ids(history_tokens + response_tokens)
    input = torch.tensor(input, device=device).unsqueeze(0)
    with torch.no_grad():
        output = model(input)
    
    len_history = len(history_tokens)
    all_logits = output.logits[0, len_history - 1 : -1]
    labels = input[0, len_history: ]
    dist = torch.distributions.Categorical(logits=all_logits)
    logprobs = dist.log_prob(labels)
    logprobs = list(logprobs.cpu().numpy())
    
    return np.exp(-np.mean(logprobs))

def get_ppl(
    model,
    tokenizer,
    conversations: List[Dict],
):
    history = ""
    ppls = []
    for round in conversations:
        history += "\"" + round["user"] + "\"\n"
        ppls.append(_get_ppl(model, tokenizer, history, "\"" + round["bot"] + "\""))
        history += "\"" + round["bot"] + "\"\n"
    return np.mean(ppls)


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    
    good_conversations = [
        {
            "user": "Hi! How are you?",
            "bot": "Good, thanks for asking! How about yourself?"
        },
    ]
    bad_conversations = [
        {
            "user": "Hi! How are you?",
            "bot": "Hello world!"
        },
    ]
    print(get_ppl(model, tokenizer, good_conversations))
    print(get_ppl(model, tokenizer, bad_conversations))
