from typing import Dict, List
from dotenv import load_dotenv
load_dotenv("../../.env")

import os
import json
import openai
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def chat(
    bot_persona: str,
    history: List[Dict[str, str]],
):
    prompt_expand_persona = "Your task is to generate the expanded persona information of bot based on dialogue history when talking to the user: {}\nNow focus on the 'bot' role. Your task is to"
    
    expanded_bot_persona = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistance to extract persona."},
            {"role": "user", "content": prompt_expand_persona.format(bot_persona, history)},
        ],
        max_tokens=1024,
        temperature=0.0,
    ).choices[0].message.content
    
    prompt = "You are a chatbot, and your persona setting is: {}\nNow generate the next utterance based on dialogue history when talking to the user: {}\nIn this history, your role is represented as 'bot'. Your speaking style should fully imitate the personality role assigned to you! Please do not expose that you are an artificial intelligence model or a language model, you must always remember that you are only assigned one personality role. Don't be verbose or too formal or polite when speaking. The response should be one to two sentences. Ensure your response is coherent and show your persona in an appropriate way."
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot that plays well upon a given persona."},
            {"role": "user", "content": prompt.format(bot_persona + " " + expanded_bot_persona, history)},
        ],
        max_tokens=100,
        temperature=1.0,
    ).choices[0].message.content
    
    return response

if __name__ == "__main__":
    bot_persona = "I was snorkeling last week and a sea turtle swam right up next to me. I am not the best at keeping up with my family because I often don't watch the news or television, and I can be shy at times. I fear that my cats will be lost or hurt. I'm going to take a cold shower, then do some gardening. I married miss usa. MBTI: INFP"
    
    conversation = [
        {
            "user": "I hope your blog post goes well! What are you writing about?",
            "bot": "Thank you! I'm writing about my favorite fashion trends from the 90s. I can't wait to share it with everyone."
        },
        {
            "user": "That's exciting! Fashion is such a creative outlet. Speaking of exciting experiences, last week while snorkeling, a sea turtle swam right next to me. It was incredible!",
            "bot": "What a magical moment! I enjoy hearing about your adventures. I'm not the best at keeping up with my family as well, especially with news and television."
        },
        {
            "user": "It's understandable. Sometimes we get so caught up in our own activities. I'm sure your family understands.",
        }
    ]
    # "bot": "I hope so. I do tend to be a bit reserved, especially when it comes to sharing my experiences."
    print(chat(bot_persona, conversation))