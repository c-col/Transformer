import re

def remove_whitespace(string_):
    string_ = str(string_)
    while '  ' in string_:
        string_ = string_.replace('  ', ' ')
    return string_


def subgraphs_from_polish(polish_):
    # get all subgraphs contained within a string of polish notation

    # base recursion case - commas and no opening parentheses
    if polish_.count('(') == 0 and polish_.count(','):
        return [x.strip() for x in polish_.split(',')][1:]


    result_holder = []
    while True:
        # terminate on end of string / no new subgraphs
        try:
            first_paren = polish_.index('(')
        except ValueError:
            break

        # get a single subgraph and add it to results
        open_paren = 1
        for ix, char in enumerate(polish_[first_paren+1:]):
            if char == '(':
                open_paren += 1
            elif char == ')':
                open_paren -= 1
            if open_paren == 0:
                result_holder.append(polish_[first_paren+1:first_paren + ix + 1])

                # focus on later part of the string and check if there are any more subgraphs at a given level
                polish_ = polish_[first_paren + ix:]
                break
    while '' in result_holder:
        result_holder.remove('')

    # recur on all results - check for deeper subgraphs
    intermed_results = [subgraphs_from_polish(x) for x in result_holder]

    # flatten recursive results
    if type(intermed_results[0]) == list:
        intermed_results = [item for sublist in intermed_results for item in sublist]

    # add recursive results and current-level results, return result
    return result_holder + intermed_results


def polish_n_steps(polish_):
    # calculate the number of "traditional QDMR" steps for an input represented in polish notation
    subgraphs = subgraphs_from_polish(polish_)
    unique_subgraphs = list(set(subgraphs))
    return len(unique_subgraphs) + 1


def zip_io_data(path_prefix):
    path_in, path_out = path_prefix + '_in.txt', path_prefix + '_out.txt'
    f_in, f_out = open(path_in, 'r', encoding='utf-8'), open(path_out, 'r', encoding='utf-8')
    in_ = [x.replace('\n', '') for x in f_in.readlines()]
    out_ = [x.replace('\n', '') for x in f_out.readlines()]
    return list(zip(in_, out_))

# compositional eval tools

def eval_re_reference():
    pass


def eval_split_input(nlg):
    composition_controller = []

    steps = nlg.lower().split(' @@sep@@ ')
    for step_index in range(len(steps)):
        step_instructions = []
        references = re.findall('@@\d+@@', steps[step_index])
        if len(references):
            step_copy = steps[step_index][:]
            for reference_index, reference in enumerate(references):
                index_form = int(reference.replace('@@', '')) - 1
                step_instructions.append(index_form)
                step_copy = step_copy.replace(reference, '@@' + str(reference_index + 1) + 'R@@')
            for reference_index in range(len(references)):
                ref_string = str(reference_index + 1)
                step_copy = step_copy.replace('@@' + ref_string + 'R@@', '@@' + ref_string + '@@')
            step_instructions.append(step_copy)
        else:
            # simple terminal node
            step_instructions.append(steps[step_index])

        composition_controller.append(step_instructions)
    return composition_controller

# def is_valid_dag(nlg):
#     steps_n = len(nlg)
#     references = re.findall('@@\d+@@', ' '.join(nlg))
#     return len(list(set(references))) + 1 == steps_n
#
#
# def get_valid_subgraphs(example):
#     states, instructions, tokenized_states = example['state'], example['nlg'], example['tokenized_state']
#     subgraphs = []
#     steps_n = len(states)
#     for steps_index in range(steps_n):
#         if is_valid_dag(instructions[:steps_index + 1]):
#             subgraphs.append((instructions[:steps_index + 1], tokenized_states[steps_index]))
#         else:
#             new_instructions = prune_and_reference(instructions[:steps_index + 1])
#             subgraphs.append((new_instructions, tokenized_states[steps_index]))
#
#     return subgraphs
#
#
# def prune_and_reference(instructions):
#     queue = [instructions[-1]]
#     required_indices = [len(instructions) - 1]
#     while len(queue):
#         step = queue.pop(0)
#         references = re.findall(r'@@\d+@@', step)
#         indices = [int(x.replace('@@', '')) - 1 for x in references]
#
#         required_indices += indices
#         queue += [instructions[index] for index in indices]
#
#     prior_removals = 0
#     pruned_instructions = []
#     for index, instruction in enumerate(instructions):
#         if index not in required_indices:
#             prior_removals += 1
#
#         else:
#             if prior_removals > 0:
#                 for ref_index, referencer in enumerate(instructions[index + 1:]):
#                     if '@@' + str(index + 1) + '@@' in referencer:
#                         instructions[index + ref_index + 1] = instructions[index + ref_index + 1].replace(
#                             '@@' + str(index + 1) + '@@', '@@' + str(index + 1 - prior_removals) + '@@'
#                         )
#
#             pruned_instructions.append(instruction)
#
#     return pruned_instructions