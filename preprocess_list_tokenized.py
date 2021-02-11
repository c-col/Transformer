import numpy as np
import json
import re
from Utils import *

np.random.seed(4)


def output_process(example):
    state = e['state'][-1]
    if type(state) == str:
        return state
    else:
        return ' '.join(state)


def polish_notation(steps):
    step_mapping = {}
    for ix, s in enumerate(steps):
        references = re.findall('@@\d+@@', s)
        if len(references):
            indices = [int(x.replace('@@','')) - 1 for x in references]
            if len(references) == 1:
                s = '(' + s.replace(' ' + references[0], '') + ', ' + step_mapping[indices[0]] + ')'
                step_mapping[ix] = s
            else:
                first_index, final_index = s.index(references[0]) - 1, s.index(references[-1]) + len(references[-1])
                s = '(' + s[:first_index] + s[final_index:] + ', '
                for jx in indices:
                    s += step_mapping[jx] + ', '
                s = s[:-2] + ')'
                step_mapping[ix] = s
        else:
            step_mapping[ix] = s

    return step_mapping[len(steps) - 1][1:-1]


def subgraphs_from_polish(polish_):
    if polish_.count('(') == 0 and polish_.count(','):
        return [x.strip() for x in polish_.split(',')][1:]
    result_holder = []
    while True:
        try:
            first_paren = polish_.index('(')
        except ValueError:
            break
        open_paren = 1
        for ix, char in enumerate(polish_[first_paren+1:]):
            if char == '(':
                open_paren += 1
            elif char == ')':
                open_paren -= 1
            if open_paren == 0:
                result_holder.append(polish_[first_paren+1:first_paren + ix + 1])
                polish_ = polish_[first_paren + ix:]
                # print('new polish:', polish_)
                break
    while '' in result_holder:
        result_holder.remove('')
    intermed_results = [subgraphs_from_polish(x) for x in result_holder]
    if type(intermed_results[0]) == list:
        intermed_results = [item for sublist in intermed_results for item in sublist]
    return result_holder + intermed_results


def remove_duplicates(data_list):
    data_set = []
    sorted_data = sorted(data_list, key=lambda x: ' '.join(x['nlg']))
    for ix in range(len(sorted_data) - 1):
        e = sorted_data[ix]
        if e['nlg'] != sorted_data[ix + 1]['nlg']:
            data_set.append(e)

    data_set.append(sorted_data[-1])
    return data_set


def is_valid_dag(nlg):
    steps_n = len(nlg)
    references = re.findall('@@\d+@@', ' '.join(nlg))
    return len(list(set(references))) + 1 == steps_n


def get_valid_subgraphs(example):
    states, instructions, tokenized_states = example['state'], example['nlg'], example['tokenized_state']
    subgraphs = []
    steps_n = len(states)
    for steps_index in range(steps_n):
        if is_valid_dag(instructions[:steps_index + 1]):
            subgraphs.append((instructions[:steps_index + 1], tokenized_states[steps_index]))
        else:
            new_instructions = prune_and_reference(instructions[:steps_index + 1])
            subgraphs.append((new_instructions, tokenized_states[steps_index]))

    return subgraphs


def prune_and_reference(instructions):
    queue = [instructions[-1]]
    required_indices = [len(instructions) - 1]
    while len(queue):
        step = queue.pop(0)
        references = re.findall(r'@@\d+@@', step)
        indices = [int(x.replace('@@', '')) - 1 for x in references]

        required_indices += indices
        queue += [instructions[index] for index in indices]

    prior_removals = 0
    pruned_instructions = []
    for index, instruction in enumerate(instructions):
        if index not in required_indices:
            prior_removals += 1

        else:
            if prior_removals > 0:
                for ref_index, referencer in enumerate(instructions[index + 1:]):
                    if '@@' + str(index + 1) + '@@' in referencer:
                        instructions[index + ref_index + 1] = instructions[index + ref_index + 1].replace(
                            '@@' + str(index + 1) + '@@', '@@' + str(index + 1 - prior_removals) + '@@'
                        )

            pruned_instructions.append(instruction)

    return pruned_instructions


def tokenize_string(example_state, example_vocab):
    return_step = ''
    temp_state = example_state[:].lower()
    first_tok = True
    while len(temp_state):
        if temp_state[:3] in example_vocab or temp_state[:3][::-1] in example_vocab:
            if first_tok:
                return_step += temp_state[:3] + ' '
                first_tok = False
            else:
                return_step += '%' + temp_state[:3] + ' '
            temp_state = temp_state[3:]
        elif temp_state[:2] in example_vocab or temp_state[:2][::-1] in example_vocab:
            if first_tok:
                return_step += temp_state[:2] + ' '
                first_tok = False
            else:
                return_step += '%' + temp_state[:2] + ' '
            temp_state = temp_state[2:]
        elif temp_state[0] in example_vocab:
            if first_tok:
                return_step += temp_state[0] + ' '
                first_tok = False
            else:
                return_step += '%' + temp_state[0] + ' '
            temp_state = temp_state[1:]
        else:
            return None
    return return_step


with open('list_task_v2.json', 'r', encoding="utf-8") as input_file:
    data = json.loads(input_file.read())
    data = remove_duplicates(data)
    n = len(data)
    np.random.shuffle(data)

    vocab = []
    for e in data:
        e_vocab, tokenized_state = [], []
        nlg, state = e['nlg'], e['state']
        add_bool = True

        for ix, step in enumerate(nlg):
            tokenized_step = ''
            # if terminal node ...
            if step.startswith('the string '):
                new_string = step.split("'")[1]
                tokenized_state.append(state[ix].lower().strip())
                e_vocab.append(new_string.lower())

            # if state is a string
            elif type(state[ix]) == str:
                # if it's a reversal
                if state[ix][::-1].lower() in e_vocab:
                    tokenized_state.append(state[ix].lower().strip())
                else:
                    tokenized_step = tokenize_string(state[ix], e_vocab)
                    if tokenized_step is not None:
                        tokenized_state.append(tokenized_step.strip())
                    else:
                        add_bool = False
                        break

            # if state[ix] is a list
            else:
                for list_element in state[ix]:
                    temp_tok = tokenize_string(list_element, e_vocab)
                    if temp_tok is None:
                        add_bool = False
                        break
                    else:
                        tokenized_step += ' ' + temp_tok

                if add_bool:
                    tokenize_step = remove_whitespace(tokenized_step).strip()
                    tokenized_state.append(tokenized_step)
                else:
                    break

        if add_bool:
            e['tokenized_state'] = tokenized_state

        vocab += e_vocab + ['%' + x for x in e_vocab] + [x[::-1] for x in e_vocab] + ['%' + x[::-1] for x in e_vocab]

    vocab = list(set(vocab))
    # with open('string_piece_vocabulary.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(vocab))

    filtered_data = []
    for e in data:
        if 'tokenized_state' in e.keys():
            filtered_data.append(e)

    train = filtered_data[:int(n*0.8)]
    val = filtered_data[int(n*0.8):int(n*0.9)]
    test = filtered_data[int(n*0.9):]

    train_in, train_out = '', ''
    for jx, e in enumerate(train):
        if jx % 5000 == 0:
            print(round(float(jx / len(train) * 100), 2), '% complete')
        subgraphs = get_valid_subgraphs(e)
        for subgraph in subgraphs:
            train_input = remove_whitespace(' @@SEP@@ '.join(subgraph[0]).lower().strip())
            train_in += train_input + '\n'
            if type(subgraph[1]) == list:
                train_out += ' '.join(subgraph[1]) + '\n'
            else:
                train_out += remove_whitespace(subgraph[1].strip()) + '\n'

        # train_in += ' @@SEP@@ '.join(e['nlg']).lower() + '\n'
        # train_out += e['tokenized_state'][-1].strip() + '\n'

    val_in, val_out = '', ''
    for e in val:
        val_input = ' @@SEP@@ '.join(e['nlg']).lower()
        val_in += val_input + '\n'
        val_out += e['tokenized_state'][-1].strip() + '\n'

    test_in, test_out = '', ''
    for e in test:
        test_input = ' @@SEP@@ '.join(e['nlg']).lower()
        test_in += test_input + '\n'
        test_out += e['tokenized_state'][-1].strip() + '\n'

    base_path = './dag_baseline_2a/'
    with open(base_path + 'train_in.txt', 'w', encoding='utf-8') as f:
        f.write(train_in)

    with open(base_path + 'train_out.txt', 'w', encoding='utf-8') as f:
        f.write(train_out)

    with open(base_path + 'val_in.txt', 'w', encoding='utf-8') as f:
        f.write(val_in)

    with open(base_path + 'val_out.txt', 'w', encoding='utf-8') as f:
        f.write(val_out)

    with open(base_path + 'test_in.txt', 'w', encoding='utf-8') as f:
        f.write(test_in)

    with open(base_path + 'test_out.txt', 'w', encoding='utf-8') as f:
        f.write(test_out)