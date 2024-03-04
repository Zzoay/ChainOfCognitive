
import os

import ast
import json
import random
import re
import requests
import time
from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset, load_from_disk

import openai
openai_config = json.load(open('config/openai.json'))
openai.api_base = openai_config['openai_api_base']
openai.api_key = openai_config ['openai_api_key']


system_prompt = """You are an assistant with empathy. Your task is to build the human cognitive chain in order to reframe cognition and thus make the thoughts of the user more positive. 
I will ask you to output the results in a step-by-step manner, you don't need to complete all the requirements at once. 
"""

group_prompt = """
Please list 100 common groups and 20 situations they may face each (not necessarily bad), which can lead to negative cognitions and thus induce negative thoughts. 
Firstly, please output 100 different groups, in python list format such as "[..., ...]" (output only contain the list).

Note:
1. Please narrate in the first person, both the situation and negative thoughts should be complete sentences.
2. 20 situations for a group
3. 5 negative thoughts for a situation
"""

situation_prompt = """Please generate 20 situations (not necessarily bad or negative) that {group_name} may face, in the first person (for example, I...). Ensure JSON format (without extra information), where the keys are "group" and "situation" (ensure double quotes), like {{r"group": ..., r"situation": [{{r"situation_id": <situation_id>, r"situation": ...}}]}}. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

negative_thought_prompt = """Correspondingly, what negative thoughts might arise (due to cognitive traps) in response to these situations? It is in the first person (for example, I...). Keep the json format, as {"group": "...", "negative_thoughts": [{"situation_id": <situation_id>, "thoughts": [...,...]}, ...]} Don't output "situation" for clarity. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

positive_thought_prompt = """These negative thoughts may be due to cognitive traps. Please step out of these traps and turn them into positives (strict one-to-one). It is in the first person (for example, I...). Keep the json format, as {"situation_id": <situation_id>, "thoughts": [...,...]} Don't output "situation" for clarity. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

negative_expression_prompt = """These 5 negative thoughts may lead to 5 corresponding negative expressions and actions. Please infer them based on each thought, in the first person (...I...), strict one-to-one. Please put them in the tone of an emotional seeker. Expressions and actions should be consistent. Each expression and action should encompass the situation the seeker faced and be affected by each negative thought. Each expression is only the utterance. Each action only contain specific behaviour. They're both complete sentences. Please Keep the json format, as {"situation_id": <situation_id>, "expressions_and_actions": [[<expression_1>, <action_1>], [<expression_2>, <action_2>], ...]} \n Don't output "situation" for clarity. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

positive_response_prompt = """To counteract these 5 negative thoughts, please output 5 (one-to-one) positive corresponding reponse strategies (in the third person (...him/her...)) and responses (in the second person (...you...)), based on the reframed positive thoughts, strict one-to-one. Each positive strategy and response should be sympathetic, correspond to each negative and positive thought. Responses and strategies should be consistent. Each strategy is to expect to change or improve the person's thought and behaviour to be positive. Each response is with a style of an emotional companion, only the utterance. Keep the json format, as {"situation_id": <situation_id>, "responses_and_strategies": [[<strategy_1>, <response_1>], [<strategy_2>, <response_2>], ...]} \n Don't output "situation" for clarity. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""


def get_response(history_messages, model="gpt-3.5-turbo-0613"):
    if True:
        return get_response_ust(history_messages, {"model": model})
    while True:
        try:
            completion = openai.ChatCompletion.create(model=model, temperature=1, messages=history_messages)
            # print(completion.usage)
        except Exception as e:
            print(e)
            time.sleep(3)
            continue
        break
    return completion.choices[0].message.content.strip()

def get_response_ust(messages, parameters=None):
    url = openai_config['openai_api_base']
    headers = {
        "Content-Type": "application/json",
        "Authorization": openai_config['openai_api_key']
        }
    data = {
        "messages": messages,
        }
    if parameters is not None:
        data.update(parameters)
    max_tries, cnt = 3, 0
    while True:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data)).json()
            usage = response['usage']['total_tokens']
            # print(response)
        except Exception as e:
            print(e)
            time.sleep(3)
            cnt += 1
            if cnt >= max_tries:
                break
            continue
        break
    return response['choices'][0]['message']['content'].strip()

def get_groups(history_messages, load_from_file=False):
    history_messages.append({
        "role": "user",
        "content": group_prompt,
    })
    if load_from_file:
        with open('data/groups.txt', 'r') as f:
            for line in f.readlines():
                ret = ast.literal_eval(line)
        return history_messages, ret
    r = get_response(history_messages, model="gpt-4")
    # process to list
    list_string = r[r.find('['): r.rfind(']')+1]
    ret = ast.literal_eval(list_string)
    ret = json.loads(json.dumps(ret))
    with open('data/groups.txt', 'a') as f:
        json.dump(ret, f)
    return history_messages, ret

def get_situations(history_messages, group_name, load_from_file=False):
    messages = history_messages.copy()
    messages.append({
        "role": "user",
        "content": situation_prompt.format(group_name=group_name)
    })
    if load_from_file:
        with open('data/situations.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                ret = ast.literal_eval(line)
                if ret['group'] == group_name:
                    return messages, ret
    r = get_response(messages) 
    list_string = r[r.find('{'): r.rfind('}')+1]
    ret = ast.literal_eval(list_string)
    # save to file
    with open(f'data/situations.txt', 'a') as f:
        f.write('\n')
        json.dump(ret, f)
    # history_messages.append({
    #     "role": "assistant",
    #     "content": str(ret),
    # })
    return messages, ret

def get_negative_thoughts(history_messages, situations, load_from_file=False):
    messages = history_messages.copy()
    messages.append({
        "role": "assistant",
        "content": str(situations),
    })
    messages.append({
        "role": "user",
        "content": negative_thought_prompt,
    })
    if load_from_file:
        with open('data/negative_thoughts.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                ret = ast.literal_eval(line)
                if ret['group'] == situations['group']:
                    return messages, ret['negative_thoughts']
    r = get_response(messages)
    ret = ast.literal_eval(r)
    # save to file
    with open(f'data/negative_thoughts.txt', 'a') as f:
        f.write('\n')
        json.dump(ret, f)
    # history_messages.append({
    #     "role": "assistant",
    #     "content": str(ret),
    # })
    return messages, []

def get_positive_thoughts(history_messages, situations, negative_thoughts, load_from_file=False):
    ret = {"group": situations['group'], "thoughts": []}
    for idx, negative_thought in enumerate(negative_thoughts):
        situation = situations['situations'][idx]
        messages = history_messages.copy()
        messages[-2] = {
            "role": "assistant",
            "content": "One of situations: " + str(situation),
        }
        messages.append({
            "role": "assistant",
            "content": str(negative_thought),
        })
        messages.append({
            "role": "user",
            "content": positive_thought_prompt,
        })
        if load_from_file:
            with open('data/positive_thoughts.txt', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    ret = ast.literal_eval(line)
                    if ret['group'] == situations['group']:
                        return messages, ret['positive_thoughts']
        while True:
            try:
                r = get_response(messages)
                one_thought = ast.literal_eval(r)
                break
            except SyntaxError as e:
                print(e)
                continue
        ret["thoughts"].append(one_thought)
        messages = []
    
    with open(f'data/positive_thoughts.txt', 'a') as f:
        f.write('\n')
        f.write(str(ret))
    return messages, ret

def get_negative_expressions_actions(history_messages, situations, negative_thoughts, load_from_file=False):
    ret = {"group": situations['group'], "expressions_and_actions": []}
    for idx, negative_thought in tqdm(enumerate(negative_thoughts)):
        situation = situations['situations'][idx]
        messages = history_messages.copy()
        messages.pop(0)
        messages[-2] = {
            "role": "assistant",
            "content": "One of situations: " + str(situation),
        }
        messages.append({
            "role": "assistant",
            "content": str(negative_thought),
        })
        messages.append({
            "role": "user",
            "content": negative_expression_prompt,
        })

        if load_from_file:
            with open('data/negative_expressions_actions.txt', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    ret = ast.literal_eval(line)
                    if ret['group'] == situations['group']:
                        return messages, ret['expressions_and_actions']

        while True:
            try:
                r = get_response(messages, model="gpt-3.5-turbo-0125")
                one_expression = ast.literal_eval(r)
                break
            except SyntaxError as e:
                print(e)
                continue
        ret["expressions_and_actions"].append(one_expression)
        messages = []
    
    with open(f'data/negative_expressions_actions.txt', 'a') as f:
        f.write('\n')
        f.write(str(ret))
    return messages, ret

def get_positive_responses_targets(history_messages, situations, positive_thoughts, load_from_file=False):
    ret = {"group": situations['group'], "strategies_and_responses": []}
    for idx, positive_thought in tqdm(enumerate(positive_thoughts), total=len(positive_thoughts)):
        situation = situations['situations'][idx]
        messages = history_messages.copy()
        messages.pop(0)
        messages.pop(0)
        messages.pop(1)
        messages.append({
            "role": "assistant",
            "content": str(positive_thought),
        })
        messages.append({
            "role": "user",
            "content": positive_response_prompt,
        })

        if load_from_file:
            with open('data/positive_strategies_responses.txt', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    ret = ast.literal_eval(line)
                    if ret['group'] == situations['group']:
                        return messages, ret['strategies_and_responses']

        while True:
            try:
                r = get_response(messages, model="gpt-3.5-turbo")
                one_response = ast.literal_eval(r)
                break
            except SyntaxError as e:
                print(e)
                continue
        ret["strategies_and_responses"].append(one_response)
        messages = []
    
    with open('data/positive_strategies_responses.txt', 'a') as f:
        f.write('\n')
        f.write(str(ret))
    
    return messages, ret

def chain_init_to_positive():
    history_messages = [{
        "role": "system",
        "content": system_prompt,
    }]
    history_messages, groups = get_groups(history_messages, load_from_file=True)
    for i, group_name in tqdm(enumerate(groups), total=len(groups)):
        # if not os.path.exists(f'data/{group_name}'):
        #     os.mkdir(f'data/{group_name}')
        # if i < 10:
        #     continue

        situation_msg, situations = get_situations(history_messages, group_name, load_from_file=True)
        situation_msg.pop(1)

        neg_msg, negative_thoughts = get_negative_thoughts(situation_msg, situations, load_from_file=True)
        pos_msg, positive_thoughts = get_positive_thoughts(neg_msg, situations, negative_thoughts, load_from_file=True)

        neg_r_msg, negative_expressions_actions = get_negative_expressions_actions(neg_msg, situations, negative_thoughts, load_from_file=True)
        pos_r_msg, positive_responses_targets = get_positive_responses_targets(pos_msg, situations, positive_thoughts, load_from_file=False)

        pass

if __name__ == "__main__":
    chain_init_to_positive()