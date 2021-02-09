
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