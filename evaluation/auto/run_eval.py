import json
from tqdm import tqdm
from copy import copy
import nltk
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dist import get_dist
from ent import get_ent
from ppl import get_ppl

def eval(file="data/openai_test_result.json"):
    data = json.load(open(file))
    
    # ppl
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    for dp in tqdm(data):
        conversations = copy(dp["history"])
        conversations[-1]["bot"] = dp["generated_response"]
        dp["ppl"] = get_ppl(model, tokenizer, conversations)
    
    # ent
    model = AutoModelForSequenceClassification.from_pretrained("zayn1111/deberta-v3-dnli")
    tokenizer = AutoTokenizer.from_pretrained("zayn1111/deberta-v3-dnli", use_fast=False)
    for dp in tqdm(data):
        persona = dp["bot_persona"]
        dp["ent"] = get_ent(model, tokenizer, nltk.sent_tokenize(persona), dp["generated_response"])
    
    json.dump(data, open(file, "w"))
    
    print("PPL for file '{}' is {:.2f}".format(
        file, np.mean([d["ppl"] for d in data])
    ))
    print("C.Score for file '{}' is {:.2f}".format(
        file, np.mean([d["ent"] for d in data])
    ))
    print("Dist-1/2 for file '{}' is {:.2f}".format(
        file, get_dist(" ".join([d["generated_response"] for d in data]))
    ))
    
if __name__ == "__main__":
    eval()