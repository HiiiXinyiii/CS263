from typing import Dict, List
from dotenv import load_dotenv
load_dotenv("../../.env")

import os
import json
import openai
from tenacity import retry, stop_after_attempt, wait_fixed

from tqdm import tqdm
from copy import copy

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def chat(
    bot_persona: str,
    history: List[Dict[str, str]],
):
    prompt = "Your task is to generate the next utterance based on dialogue history when talking to the user.\nYour speaking style should fully imitate the personality role assigned to you! Please do not expose that you are an artificial intelligence model or a language model, you must always remember that you are only assigned one personality role. Don't be verbose or too formal or polite when speaking. The response should be one to two sentences. Note that you should mimic the style provided in history so that you will continue to act as the 'bot'. Now generate the next utterance given this history: {}".format(history)
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot that plays well upon a given persona. Your persona setting is: {} You will try your best to provide good conversation that is coherent, fluent and also express your persona during chit chat.".format(bot_persona)},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        temperature=1.0,
    ).choices[0].message.content
    
    return response

def test(save_path="savings/test_result.json"):
    
    test_conversations = json.load(open("./data/val_spc_dataset.json"))[:50]
    test_data = []
    
    for dp in test_conversations:
        bot_persona = dp["user2_persona"]
        for idx in range(len(dp["conversations"])):
            history = copy(dp["conversations"][:idx])
            history.append({
                "user": dp["conversations"][idx]["user"],
            })
            test_data.append({
                "bot_persona": bot_persona,
                "history": history,
                "gt_response": dp["conversations"][idx]["bot"]
            })

    print("length of test data: ", len(test_data))
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, 'w') as json_file:
            json.dump(test_data, json_file, indent=4)
    
    for entry in tqdm(test_data):
        entry["generated_response"] = chat(
            bot_persona=entry["bot_persona"],
            history=entry["history"]
        )
    
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, 'w') as json_file:
            json.dump(test_data, json_file, indent=4)
    

if __name__ == "__main__":
    test()