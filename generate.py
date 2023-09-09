
import os

import ast
import re
import json
import time
import random
from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset, load_from_disk

import openai
openai_config = json.load(open('config/openai.json'))
openai.api_base = openai_config['openai_api_base']
openai.api_key = openai_config ['api_key']


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

situation_prompt = """Please generate 20 situations (not necessarily bad or negative) that {group_name} may face, in the first person (for example, I...). Ensure JSON format (without extra information), where the keys are "group" and "situation" (ensure double quotes), like {{r"group": ..., r"situation": [{{r"situation_id": 1, r"situation": ...}}]}}. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

negative_thought_prompt = """Correspondingly, what negative thoughts might arise (due to cognitive traps) in response to these situations? It is in the first person (for example, I...). Keep the json format, as {"group": "...", "negative_thoughts": [{"situation_id": 1, "thoughts": [...,...]}, ...]} Don't output "situation" for clarity. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

positive_thought_prompt = """These negative thoughts may be due to cognitive traps. Please step out of these traps and turn them into positives (strict one-to-one). It is in the first person (for example, I...). Keep the json format, as {"situation_id": 1, "thoughts": [...,...]} Don't output "situation" for clarity. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

negative_expression_prompt = """These negative thoughts may lead to some negative expressions. Please infer the possible expressions in the first person (for example, I...), strict one-to-one. Please put them in the tone of an emotional seeker. The expressions should contain the situation the seeker faced and be affected by his negative thoughts. Please Keep the json format, as {"situation_id": 1, "expressions": [...,...]} Don't output "situation" for clarity. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

negative_action_prompt = """These negative thoughts may lead to some negative actions. Please infer the possible actions in the first person (for example, I...), strict one-to-one. Please put them in the tone of an emotional seeker. The actions should contain the situation the seeker faced and be affected by his negative thoughts. Please Keep the json format, as {"situation_id": 1, "actions": [...,...]} Don't output "situation" for clarity. Remember that single quotes can be problematic to import, so wrap the content in double quotes).
"""

positive_response_prompt = """
"""

positive_action_prompt = """
"""


def get_response(history_messages, model="gpt-3.5-turbo-0613"):
    while True:
        try:
            completion = openai.ChatCompletion.create(model=model, temperature=1, messages=history_messages)
            print(completion.usage)
        except Exception as e:
            print(e)
            time.sleep(3)
            continue
        break
    return completion.choices[0].message.content.strip()

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

def get_negative_expressions(history_messages, situations, negative_thoughts, load_from_file=False):
    ret = {"group": situations['group'], "expressions": []}
    for idx, negative_thought in enumerate(negative_thoughts):
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
            with open('data/negative_expressions.txt', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    ret = ast.literal_eval(line)
                    if ret['group'] == situations['group']:
                        return messages, ret['negative_expressions']

        while True:
            try:
                r = get_response(messages)
                one_expression = ast.literal_eval(r)
                break
            except SyntaxError as e:
                print(e)
                continue
        ret["expressions"].append(one_expression)
        messages = []
    
    with open(f'data/negative_expressions.txt', 'a') as f:
        f.write('\n')
        f.write(str(ret))
    return messages, ret

def chain_init_to_positive():
    history_messages = [{
        "role": "system",
        "content": system_prompt,
    }]
    history_messages, groups = get_groups(history_messages, load_from_file=True)
    for i, group_name in enumerate(groups):
        # if not os.path.exists(f'data/{group_name}'):
        #     os.mkdir(f'data/{group_name}')
        # if i < 38:
        #     continue

        situation_msg, situations = get_situations(history_messages, group_name, load_from_file=True)
        situation_msg.pop(1)

        neg_msg, negative_thoughts = get_negative_thoughts(situation_msg, situations, load_from_file=True)
        pos_msg, positive_thoughts = get_positive_thoughts(neg_msg, situations, negative_thoughts, load_from_file=True)

        neg_r_msg, negative_expressions = get_negative_expressions(neg_msg, situations, negative_thoughts, load_from_file=False)

        pass

if __name__ == "__main__":
    chain_init_to_positive()