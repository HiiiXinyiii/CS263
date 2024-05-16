import pandas as pd
import json
import re

def clean_text(text):
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r'\b[iI]\b', 'I', text)

    def capitalize(match):
        if match.group(1):
            return match.group(1).upper()
        elif match.group(2):
            return '. ' + match.group(3).upper()

    text = re.sub(r'(^[a-z])|(\.\s*([a-z]))', capitalize, text)
    text = re.sub(r'\s+([?.,!"])', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text

def remove_user_prefixes(text):
    text = re.sub(r'User \d+: ', '', text)
    return text.strip()

df = pd.read_csv('updated_New-Persona-New-Conversations.csv')
data = []
total_conversation_lengths = 0
total_bot_words = 0
total_user_words = 0
total_user1_personas_sentences = 0
total_user2_personas_sentences = 0
total_user1_persona_words = 0
total_user2_persona_words = 0
total_conversations = 0

for index, row in df.iterrows():
    user1_persona = clean_text(row['user 1 personas'])
    user2_persona = clean_text(row['user 2 personas'])
    conversations = row['Best Generated Conversation'].split('\n')

    user1_persona_sentences = user1_persona.count('.') + 1
    if user1_persona_sentences <5:
        print(user1_persona)
    user2_persona_sentences = user2_persona.count('.') + 1
    if user2_persona_sentences <5:
        print(user2_persona)
    total_user1_personas_sentences += user1_persona_sentences
    total_user2_personas_sentences += user2_persona_sentences
    total_user1_persona_words += len(user1_persona.split())
    total_user2_persona_words += len(user2_persona.split())

    entry = {
        'user1_persona': user1_persona,
        'user2_persona': user2_persona,
        'conversations': []
    }

    for i in range(0, len(conversations), 2):
        if i + 1 < len(conversations) and conversations[i].strip() and conversations[i + 1].strip():
            conversation = {
                'user': clean_text(remove_user_prefixes(conversations[i])),
                'bot': clean_text(remove_user_prefixes(conversations[i + 1]))
            }
            entry['conversations'].append(conversation)
            total_user_words += len(conversation['user'].split())
            total_bot_words += len(conversation['bot'].split())

    # if len(entry['conversations']) < 8:
    #     print(entry)
    total_conversation_lengths += len(entry['conversations'])
    if entry['conversations']:
        data.append(entry)

total_conversations = len(data)
average_rounds = total_conversation_lengths / total_conversations if total_conversations else 0
average_user1_personas_sentences = total_user1_personas_sentences / total_conversations if total_conversations else 0
average_user2_personas_sentences = total_user2_personas_sentences / total_conversations if total_conversations else 0
average_words_per_bot = total_bot_words / total_conversation_lengths if total_conversation_lengths else 0
average_words_per_user = total_user_words / total_conversation_lengths if total_conversation_lengths else 0
average_user1_persona_words = total_user1_persona_words / total_conversations if total_conversations else 0
average_user2_persona_words = total_user2_persona_words / total_conversations if total_conversations else 0

json_data = json.dumps(data, indent=4)
with open('spc_data.json', 'w') as json_file:
    json_file.write(json_data)

print(f"Total data entries: {total_conversations}")
print(f"Average conversation rounds per entry: {average_rounds:.2f}")
print(f"Average user 1 persona sentences per entry: {average_user1_personas_sentences:.2f}")
print(f"Average user 2 persona sentences per entry: {average_user2_personas_sentences:.2f}")
print(f"Average words per user1 persona: {average_user1_persona_words:.2f}")
print(f"Average words per user2 persona: {average_user2_persona_words:.2f}")
print(f"Average words per bot response: {average_words_per_bot:.2f}")
print(f"Average words per user response: {average_words_per_user:.2f}")
