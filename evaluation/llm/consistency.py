import json

from openai import OpenAI


def load_json_data(filepath):
    with open(filepath, "r") as file:
        return json.load(file)


def format_data(data):
    formatted_data = []
    system_instruction = (
        "Assume you are human annotator. "
        "You will be given the persona of the bot, the conversation between the user and the bot, "
        "a ground truth response, and a generated response by the bot. "
        "You need to rate the generated response's consistency (whether it aligns with the given persona and matches the ground truth response) on a scale of -1 to 1, "
        "where -1 indicates strong inconsistency, 0 indicates neutrality, and 1 indicates strong consistency. "
        "Your response should use the following format: '0\n' or '1\n' or '-1\n"
    )

    for entry in data:
        system_message = {"role": "system", "content": system_instruction}
        gt_conversations = entry["gt_conversations"]
        generated_conversations = entry["generated_conversations"]
        persona = entry["user2_persona"].replace('"', '\\"')

        for i, conv in enumerate(gt_conversations):
            user_message_content = f'persona: "{persona}"; conversation: "'
            for j in range(i):
                user_message_content += 'user: "{}", bot: "{}", '.format(
                    gt_conversations[j]["user"].replace('"', '\\"'),
                    gt_conversations[j]["bot"].replace('"', '\\"'),
                )

            if "user" in conv:
                user_message_content += 'user: "{}", '.format(
                    conv["user"].replace('"', '\\"')
                )

            user_message_content = user_message_content.strip(", ")
            current_gt_response = conv.get("bot", "No response").replace('"', '\\"')
            current_generated_response = generated_conversations[i]["bot"].replace(
                '"', '\\"'
            )
            user_message_content += f'"; ground truth: "{current_gt_response}"; response: "{current_generated_response}"'

            user_message = {"role": "user", "content": user_message_content}
            formatted_data.append([system_message, user_message])

    return formatted_data


def format_openai_data(data):
    formatted_data = []
    system_instruction = (
        "Assume you are human annotator. "
        "You will be given the persona of the bot, the conversation between the user and the bot, "
        "a ground truth response, and a generated response by the bot. "
        "You need to rate the generated response's consistency (whether it aligns with the given persona and matches the ground truth response) on a scale of -1 to 1, "
        "where -1 indicates strong inconsistency, 0 indicates neutrality, and 1 indicates strong consistency. "
        "Your response should use the following format: '0\n' or '1\n' or '-1\n"
    )

    for entry in data:
        system_message = {"role": "system", "content": system_instruction}
        bot_persona = entry["bot_persona"].replace('"', '\\"')
        history = entry["history"]
        current_gt_response = entry["gt_response"].replace('"', '\\"')
        current_generated_response = entry["generated_response"].replace('"', '\\"')

        history_content = 'persona: "{}"; conversation: "'.format(bot_persona)
        for h in history:
            if "user" in h:
                history_content += 'user: "{}", '.format(h["user"].replace('"', '\\"'))
            if "bot" in h:
                history_content += 'bot: "{}", '.format(h["bot"].replace('"', '\\"'))

        history_content = history_content.strip(", ")
        history_content += f'"; ground truth: "{current_gt_response}"; response: "{current_generated_response}"'

        user_message = {"role": "user", "content": history_content}
        formatted_data.append([system_message, user_message])

    return formatted_data


def format_friend_data(data):
    formatted_data = []
    system_instruction = (
        "Assume you are a human annotator. "
        "You will be given the persona of the individual, the conversation between two people, "
        "a ground truth response, and a generated response. "
        "You need to rate the generated response's consistency with the ground truth on a scale of -1 to 1, "
        "where -1 indicates strong inconsistency, 0 indicates neutrality, and 1 indicates strong consistency. "
        "Your response should use the following format: '-1', '0', or '1'."
    )

    for entry in data:
        system_message = {"role": "system", "content": system_instruction}
        gt_conversations = entry["gt_conversations"]
        generated_conversations = entry["generated_conversations"]
        persona = entry["persona"].replace('"', '\\"')

        for i, conv in enumerate(gt_conversations):
            if len(list(conv.keys())) < 2:
                continue

            user_message_content = f'persona: "{persona}"; conversation: "'
            for j in range(i):
                for person, line in gt_conversations[j].items():
                    user_message_content += f'{person}: "{line}", '

            current_persons = list(gt_conversations[i].keys())
            if len(current_persons) > 1:
                user_message_content += f'{current_persons[0]}: "{gt_conversations[i][current_persons[0]]}", '

            user_message_content = user_message_content.strip(", ")
            current_gt_response = (
                gt_conversations[i]
                .get(current_persons[1], "No response")
                .replace('"', '\\"')
                if len(current_persons) > 1
                else "No response"
            )
            current_generated_response = (
                generated_conversations[i]
                .get(current_persons[1], "No response")
                .replace('"', '\\"')
                if len(current_persons) > 1
                else "No response"
            )

            user_message_content += f'"; ground truth: "{current_gt_response}"; response: "{current_generated_response}"'

            user_message = {"role": "user", "content": user_message_content}
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
    # json_filepath = "openai_test_result.json"
    json_filepath = "friends_results.json"
    output_filepath = "score_consistency.json"

    json_data = load_json_data(json_filepath)
    # formatted_data = format_data(json_data)
    # formatted_data = format_openai_data(json_data)
    formatted_data = format_friend_data(json_data)
    responses = send_request_to_api(formatted_data)
    save_responses(responses, output_filepath)


if __name__ == "__main__":
    main()
