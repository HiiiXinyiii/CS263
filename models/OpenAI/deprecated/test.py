import os
import json
import openai


def get_chatgpt_response(prompt, bot_info, user_info=None):
    openai.api_key = ""  
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": bot_info},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message['content']


def test(save_path="savings/test_result.json"):
    # Save the responses
    res = []

    with open("data/val_spc_dataset.json", "r", encoding='utf-8') as f:
        data = json.load(f)

    len_data = len(data)
    for i_idx, i_item in enumerate(data[0:2]):
        print(f"Testing on conversation {i_idx + 1}/{len_data}")

        tmp_res = {"conversations": []}
        for j_conv in i_item.get("conversations", []):
            tmp_res["conversations"].append(get_chatgpt_response(prompt=j_conv.get("bot", ""), bot_info=i_item.get("user2_persona", "")))
        res.append(tmp_res)


    # Save result
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, 'w') as json_file:
            json.dump(res, json_file, indent=4)


if __name__ == "__main__":
    test()

