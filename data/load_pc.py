import pandas as pd
import json
import re


def clean_text(text):
    text = text.strip()
    text = re.sub(r'\b[iI]\b', 'I', text)

    def capitalize(match):
        if match.group(1):
            return match.group(1).upper()
        elif match.group(2):
            return '. ' + match.group(3).upper()

    text = re.sub(r'(^[a-z])|(\.\s*([a-z]))', capitalize, text)
    text = re.sub(r'\s+([?.,!"])', r'\1', text)

    return text


df = pd.read_csv('updated_personality.csv')
data = []
total_conversation_lengths = 0
total_bot_words = 0
total_user_words = 0
total_personas_sentences = 0
total_persona_words = 0
total_conversations = 0

for index, row in df.iterrows():
    conversations = row['chat'].split('\n')
    cleaned_persona = clean_text(row['Persona'])
    persona_sentences = cleaned_persona.count('.') + 1
    total_personas_sentences += persona_sentences
    total_persona_words += len(cleaned_persona.split())

    entry = {
        'persona': cleaned_persona,
        'conversations': []
    }

    for i in range(0, len(conversations), 2):
        if i + 1 < len(conversations) and conversations[i].strip() and conversations[i + 1].strip():
            conversation = {
                'user': clean_text(conversations[i]),
                'bot': clean_text(conversations[i + 1])
            }
            entry['conversations'].append(conversation)
            total_user_words += len(conversations[i].split())
            total_bot_words += len(conversations[i + 1].split())

    total_conversation_lengths += len(entry['conversations'])
    if entry['conversations']:
        data.append(entry)

total_conversations = len(data)
average_rounds = total_conversation_lengths / total_conversations if total_conversations else 0
average_personas_sentences = total_personas_sentences / total_conversations if total_conversations else 0
average_persona_words = total_persona_words / total_conversations if total_conversations else 0
average_words_per_bot = total_bot_words / total_conversation_lengths if total_conversation_lengths else 0
average_words_per_user = total_user_words / total_conversation_lengths if total_conversation_lengths else 0

json_data = json.dumps(data, indent=4)
with open('pc_data.json', 'w') as json_file:
    json_file.write(json_data)

print(f"Total data entries: {total_conversations}")
print(f"Average conversation rounds per entry: {average_rounds:.2f}")
print(f"Average persona sentences per entry: {average_personas_sentences:.2f}")
print(f"Average words per persona: {average_persona_words:.2f}")
print(f"Average words per bot response: {average_words_per_bot:.2f}")
print(f"Average words per user response: {average_words_per_user:.2f}")


