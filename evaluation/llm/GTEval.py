import json

from openai import OpenAI


def load_json_data(filepath):
    with open(filepath, "r") as file:
        return json.load(file)


def format_data(data):
    formatted_data = []
    system_instruction = (
        "You are an AI assistant who helps humans perform the Turing test more easily. "
        "You will be provided with two sets of conversations. Each set can contain AI-generated utterances. "
        "You need to read both conversations as they incrementally grow and judge if AI is involved in either of them. "
        "Your response should use the following format: '0\n' or '1\n' or '2\n' or '3\n', "
        "where 0 means neither, 1 means Conversation 1, 2 means Conversation 2, and 3 means both."
    )

    for entry in data:
        system_message = {"role": "system", "content": system_instruction}
        gt_conversations = entry["gt_conversations"]
        generated_conversations = entry["generated_conversations"]

        max_length = max(len(gt_conversations), len(generated_conversations))
        for i in range(max_length):
            formatted_conversation1 = ""
            formatted_conversation2 = ""

            for j in range(i + 1):
                if j < len(gt_conversations):
                    formatted_conversation1 += f'user: "{gt_conversations[j]["user"]}", bot: "{gt_conversations[j].get("bot", "No response")}", '
                if j < len(generated_conversations):
                    formatted_conversation2 += f'user: "{generated_conversations[j]["user"]}", bot: "{generated_conversations[j]["bot"]}", '

            formatted_conversation1 = formatted_conversation1.strip(", ")
            formatted_conversation2 = formatted_conversation2.strip(", ")

            user_message = {
                "role": "user",
                "content": f'Conversation 1: "{formatted_conversation1}"; Conversation 2: "{formatted_conversation2}"',
            }

            formatted_data.append([system_message, user_message])

    return formatted_data


def format_openai_data(data):
    formatted_data = []
    system_instruction = (
        "You are an AI assistant who helps humans perform the Turing test more easily. "
        "You will be provided with two sets of conversations. Each set can contain AI-generated utterances. "
        "You need to read both conversations as they incrementally grow and judge if AI is involved in either of them. "
        "Your response should use the following format: '0\n' or '1\n' or '2\n' or '3\n', "
        "where 0 means neither, 1 means Conversation 1, 2 means Conversation 2, and 3 means both."
    )

    for entry in data:
        system_message = {"role": "system", "content": system_instruction}
        bot_persona = entry["bot_persona"].replace('"', '\\"')
        history = entry["history"]
        gt_response = entry["gt_response"].replace('"', '\\"')
        generated_response = entry["generated_response"].replace('"', '\\"')

        formatted_conversation1 = 'persona: "{}"; conversation: "'.format(bot_persona)
        for h in history:
            if "user" in h:
                formatted_conversation1 += 'user: "{}", '.format(
                    h["user"].replace('"', '\\"')
                )
            if "bot" in h:
                formatted_conversation1 += 'bot: "{}", '.format(
                    h["bot"].replace('"', '\\"')
                )

        formatted_conversation1 += 'user: "Latest", bot: "{}"'.format(gt_response)
        formatted_conversation1 = formatted_conversation1.strip(", ")

        formatted_conversation2 = (
            formatted_conversation1[: -len(gt_response) - 1] + generated_response + '"'
        )

        user_message = {
            "role": "user",
            "content": f'Conversation 1: "{formatted_conversation1}"; Conversation 2: "{formatted_conversation2}"',
        }

        formatted_data.append([system_message, user_message])

    return formatted_data


def format_friend_data(data):
    formatted_data = []
    system_instruction = (
        "You are an AI assistant who helps humans perform the Turing test more easily. "
        "You will be provided with two sets of conversations. Each set can contain AI-generated utterances. "
        "You need to read both conversations as they incrementally grow and judge if AI is involved in either of them. "
        "Your response should use the following format: '0', '1', '2', or '3', "
        "where 0 means neither, 1 means Conversation 1, 2 means Conversation 2, and 3 means both."
    )

    for entry in data:
        system_message = {"role": "system", "content": system_instruction}
        gt_conversations = entry["gt_conversations"]
        generated_conversations = entry["generated_conversations"]
        persona = entry["persona"].replace('"', '\\"')

        max_length = max(len(gt_conversations), len(generated_conversations))
        for i in range(max_length):
            formatted_conversation1 = ""
            formatted_conversation2 = ""

            for j in range(i + 1):
                if j < len(gt_conversations) and j < len(generated_conversations):
                    for person, line in gt_conversations[j].items():
                        formatted_conversation1 += f'{person}: "{line}", '
                    for person, line in generated_conversations[j].items():
                        formatted_conversation2 += f'{person}: "{line}", '

            formatted_conversation1 = formatted_conversation1.strip(", ")
            formatted_conversation2 = formatted_conversation2.strip(", ")

            user_message = {
                "role": "user",
                "content": f'Persona: "{persona}"; Conversation 1: "{formatted_conversation1}"; Conversation 2: "{formatted_conversation2}"',
            }

            formatted_data.append([system_message, user_message])

    return formatted_data


def send_request_to_api(data):
    client = OpenAI()
    responses = []
    for messages in data:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages
        )
        responses.append(response)
    return responses


def save_responses(responses, output_filepath):
    with open(output_filepath, "w") as file:
        json.dump([response.choices[0].message.content for response in responses], file)


def main():
    # json_filepath = "simple.json"
    # json_filepath = "New document 1.json"
    # json_filepath = "openai_test_result.json"
    json_filepath = "friends_results.json"
    output_filepath = "score_gteval.json"

    json_data = load_json_data(json_filepath)
    # formatted_data = format_data(json_data)
    # formatted_data = format_openai_data(json_data)
    formatted_data = format_friend_data(json_data)
    responses = send_request_to_api(formatted_data)
    save_responses(responses, output_filepath)


if __name__ == "__main__":
    main()
