import praw
import json
import time

start = time.time()

reddit = praw.Reddit(
    client_id='v2OnR3UAhw7wV03nbMudGA',
    client_secret='7jhZqrZ_5BJisX8VTdiAskV7GM27Eg',
    password='kangjun020321',
    user_agent='MyApp by u/Plastic_Wrangler642',
    username='Plastic_Wrangler642'
)


def fetch_conversations(subreddit_name, limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    conversations = []

    for submission in subreddit.hot(limit=limit):
        if not submission.selftext:
            continue

        topic_body = submission.title + "\n" + submission.selftext

        submission.comments.replace_more(limit=0)
        for comment in submission.comments:
            if comment.body:  # 确保评论不为空
                conversation = {
                    "source": topic_body,
                    "reply": comment.body
                }
                conversations.append(conversation)

    return conversations


subreddit_name = 'CasualConversation'
conversation_data = fetch_conversations(subreddit_name, limit=50)

with open('reddit_dialogues.json', 'w', encoding='utf-8') as f:
    json.dump(conversation_data, f, ensure_ascii=False, indent=4)

print("Saved as reddit_dialogues.json")




