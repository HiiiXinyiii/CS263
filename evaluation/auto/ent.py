from typing import List, Dict

import numpy as np
import torch

label2score = {
    0: 1.0,
    1: 0.0,
    2: -1.0,
}

def _get_ent(
    model, 
    tokenizer,
    premise: str,
    hypothesis: str,
):
    device = model.device
    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to(device)) 
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label = np.argmax(prediction)
    return label2score[label]
    
def get_ent(
    model,
    tokenizer,
    persona: List[str],
    utterance: str,
):
    scores = [_get_ent(model, tokenizer, utterance, p) for p in persona]
    return np.nanmean(scores)


if __name__ == "__main__":
    import nltk
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    tokenizer = AutoTokenizer.from_pretrained("zayn1111/deberta-v3-dnli", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained("zayn1111/deberta-v3-dnli")
    
    persona = "I'm moving to a new city to pursue my culinary dreams. I am a marathon runner. I like to sing broadway show tunes. I knit myself a sweater that's so warm and comfy, and it also helps me fight my sweet tooth because I'm too busy knitting to eat gummy worms! My sister is a teacher, and my niece and nephew are still in school."
    persona = nltk.sent_tokenize(persona)
    utterance = "I'm doing pretty well. I'm excited to be moving to a new city soon!"
    
    print(get_ent(model, tokenizer, persona, utterance))
    