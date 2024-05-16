import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import string

nltk.download('punkt')

def get_dist(text: str):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = word_tokenize(text)
    
    unigrams = Counter(words)
    dist_1 = len(unigrams) / len(words) if words else 0
    
    bigrams = Counter(ngrams(words, 2))
    dist_2 = len(bigrams) / len(words[:-1]) if len(words) > 1 else 0
    
    return dist_1, dist_2

if __name__ == "__main__":
    
    conversations = [
        {
            "user": "Hi! How are you?",
            "bot": "Good, thanks for asking! How about yourself?"
        },
        {
            "user": "I'm doing pretty well. I'm excited to be moving to a new city soon!",
            "bot": "Oh that's awesome! What city are you moving to?"
        },
        {
            "user": "I'm moving to Portland, Oregon. I'm going to be pursuing my culinary dreams there.",
            "bot": "That sounds amazing! I love Portland. I'm originally from there."
        },
        {
            "user": "Really? That's so cool! Do you still live there?",
            "bot": "No, I moved away a few years ago. But I still visit my family there often."
        },
        {
            "user": "That's great. Do you've any favorite places to visit in Portland?",
            "bot": "Yes, I love going to Powell's Books. It's the largest independent bookstore in the world."
        },
        {
            "user": "That sounds amazing! I've never been to Powell's Books before. I'm definitely going to have to check it out.",
            "bot": "You definitely should! It's a really cool place."
        },
        {
            "user": "Thanks for the recommendation! I'm sure I'll love it.",
            "bot": "No problem! I'm glad I could help."
        }
    ]
    text = " ".join([c["bot"] for c in conversations])
    print(get_dist(text))