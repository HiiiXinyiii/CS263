import pandas as pd

contractions = {
    " i am ": " i'm ",
    " you are ": " you're ",
    " he is ": " he's ",
    " she is ": " she's ",
    " it is ": " it's ",
    "that is ": "that's ",
    "what is ": "what's ",
    "where is ": "where's ",
    "there is ": "there's ",
    "who is ": "who's ",
    "how is ": "how's ",
    " we are ": " we're ",
    " they are ": " they're ",
    "have not ": "haven't ",
    "has not ": "hasn't ",
    "had not ": "hadn't ",
    "will not ": "won't ",
    "would not ": "wouldn't ",
    "do not ": "don't ",
    "does not ": "doesn't ",
    "did not ": "didn't ",
    "cannot ": "can't ",
    "could not ": "couldn't ",
    "should not ": "shouldn't ",
    "might not ": "mightn't ",
    "must not ": "mustn't "
}

additional_contractions = {
    "let us ": "let's ",
    "are not ": "aren't ",
    "is not ": "isn't ",
    "was not ": "wasn't ",
    "were not ": "weren't ",
    " I have ": " I've ",
    " you have ": " you've ",
    " we have ": " we've ",
    " they have ": " they've ",
    "would have ": "would've ",
    "should have ": "should've ",
    "could have ": "could've ",
    "he has ": "he's ",
    "she has ": "she's ",
    "it has ": "it's ",
    "that has ": "that's ",
    "what has ": "what's ",
    "where has ": "where's ",
    "there has ": "there's ",
    "who has ": " who's ",
    "how has ": "how's ",
    " he will ": " he'll ",
    " she will ": " she'll ",
    " it will ": " it'll ",
    "there will ": "there'll ",
    "we will ": "we'll ",
    "they will ": "they'll ",
    " I would ": " I'd ",
    " you would ": " you'd ",
    " he would ": " he'd ",
    " she would ": " she'd ",
    " it would ": " it'd ",
    " we would ": " we'd ",
    " they would ": " they'd ",
    " I had ": " I'd ",
    " you had ": " you'd ",
    " he had ": " he'd ",
    " she had ": " she'd ",
    " it had ": " it'd ",
    " we had ": " we'd ",
    " they had ": " they'd "
}

contractions.update(additional_contractions)


def replace_contractions(text):
    for key, value in contractions.items():
        text = text.replace(key, value)
    return text


df = pd.read_csv('personality.csv')
df['Persona'] = df['Persona'].apply(replace_contractions)
df['chat'] = df['chat'].apply(replace_contractions)
df.to_csv('updated_personality.csv', index=False)

df1 = pd.read_csv('New-Persona-New-Conversations.csv')
df1['user 1 personas'] = df1['user 1 personas'].apply(replace_contractions)
df1['user 2 personas'] = df1['user 2 personas'].apply(replace_contractions)
df1['Best Generated Conversation'] = df1['Best Generated Conversation'].apply(replace_contractions)
df1.to_csv('updated_New-Persona-New-Conversations.csv', index=False)