import numpy as np
import json
import re

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
    final_list = []
    for example in data_list:
        if example not in final_list:
            final_list.append(example)

    return final_list


def is_valid_dag(nlg):
    steps_n = len(nlg)
    references = re.findall('@@\d+@@', ' '.join(nlg))
    return len(list(set(references))) + 1 == steps_n


def get_valid_subgraphs(example):
    states, instructions = example['state'], example['nlg']
    subgraphs = []
    steps_n = len(states)
    for steps_index in range(steps_n):
        if is_valid_dag(instructions[:steps_index + 1]):
            subgraphs.append((instructions[:steps_index + 1], states[steps_index]))

    return subgraphs


chars = 'abcdefghijklmnopqrstuvwxyz'

with open('list_task_v2.json', 'r', encoding="utf-8") as input_file:
    data = json.loads(input_file.read())
    # data = remove_duplicates(data)
    n = len(data)
    np.random.shuffle(data)

    train = data[:int(n*0.8)]
    val = data[int(n*0.8):int(n*0.9)]
    test = data[int(n*0.9):]

    train_in, train_out = '', ''
    # train_examples = []
    # val_skip, test_skip = 0, 0
    for jx, e in enumerate(train):
        if jx % 5000 == 0:
            print(float(jx / len(train) * 100), 'percent complete')
        subgraphs = get_valid_subgraphs(e)
        for subgraph in subgraphs:
            train_input = ' @@SEP@@ '.join(subgraph[0])
            # train_examples.append(train_input)
            train_in += train_input + '\n'
            if type(subgraph[1]) == list:
                train_out += ' '.join(subgraph[1]) + '\n'
            else:
                train_out += subgraph[1] + '\n'
        # train_in += ' @@SEP@@ '.join(e['nlg']) + '\n'

        # pol = polish_notation(e['nlg'])
        # pol_subgraphs = subgraphs_from_polish(pol)
        # pol_uq_subgraphs = list(set(pol_subgraphs))

        # train_in += polish_notation(e['nlg']) + '\n'
        # train_out += output_process(e) + '\n'
        # train_out += polish_notation(e['nlg']) + '\n'

    val_in, val_out = '', ''
    for e in val:
        val_input = ' @@SEP@@ '.join(e['nlg'])
        # if val_input in train_examples:
        #     val_skip += 1
        #     continue
        val_in += val_input + '\n'
        # val_in += polish_notation(e['nlg']) + '\n'
        val_out += output_process(e) + '\n' + '\n'
        # val_out += polish_notation(e['nlg']) + '\n'

    test_in, test_out = '', ''
    for e in test:
        test_input = ' @@SEP@@ '.join(e['nlg'])
        # if test_input in train_examples:
        #     test_skip += 1
        #     continue
        test_in += test_input + '\n'
        # test_in += polish_notation(e['nlg']) + '\n'
        test_out += output_process(e) + '\n' + '\n'
        # test_out += polish_notation(e['nlg']) + '\n'

    with open('./list_data_allen_v3/train_in.txt', 'w', encoding='utf-8') as f:
        f.write(train_in)

    with open('./list_data_allen_v3/train_out.txt', 'w', encoding='utf-8') as f:
        f.write(train_out)

    with open('./list_data_allen_v3/val_in.txt', 'w', encoding='utf-8') as f:
        f.write(val_in)

    with open('./list_data_allen_v3/val_out.txt', 'w', encoding='utf-8') as f:
        f.write(val_out)

    with open('./list_data_allen_v3/test_in.txt', 'w', encoding='utf-8') as f:
        f.write(test_in)

    with open('./list_data_allen_v3/test_out.txt', 'w', encoding='utf-8') as f:
        f.write(test_out)

    # with open('./allen_to_polish/train.source', 'w', encoding='utf-8') as f:
    #     f.write(train_in)
    #
    # with open('./allen_to_polish/train.target', 'w', encoding='utf-8') as f:
    #     f.write(train_out)
    #
    # with open('./allen_to_polish/val.source', 'w', encoding='utf-8') as f:
    #     f.write(val_in)
    #
    # with open('./allen_to_polish/val.target', 'w', encoding='utf-8') as f:
    #     f.write(val_out)
    #
    # with open('./allen_to_polish/test.source', 'w', encoding='utf-8') as f:
    #     f.write(test_in)
    #
    # with open('./allen_to_polish/test.target', 'w', encoding='utf-8') as f:
    #     f.write(test_out)

print(val_skip, test_skip)