import numpy as np
import json
# from nltk.corpus import words
# words.words()
#
# strings, booleans, integer base
# slice := (start, stop) integer pair
# TaskList := list of strings
# mask := list of booleans
# transformation := 2-d array of non-negative integers

np.random.seed(4)
CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  # ' 1234567890'
CHAR_LIST = [x for x in CHARS]


def random_choice_from_list(list_):
    return list_[np.random.randint(len(list_))]


def reference_dag_form(index):
    return '@@' + str(index + 1) + '@@'


def nlg_multiple_references(index_list):
    if len(index_list) == 0:
        raise ValueError("don't pass empty index lists")
    elif len(index_list) == 1:
        return reference_dag_form(index_list[0])
    elif len(index_list) == 2:
        return reference_dag_form(index_list[0]) + ' and ' + reference_dag_form(index_list[1])
    else:
        return_string = ''
        for i in index_list[:-1]:
            return_string += reference_dag_form(i) + ', '
        return return_string + 'and ' + reference_dag_form(index_list[-1])


class Operation:
    def __init__(self):
        # some notion of operands or operand indices on initialization
        self.target_index = []

    def nlg(self):
        # nlg given operation and operands, in stochastic way
        # needs to be fed some notion of index to refer to operands
        raise NotImplementedError('superclass method for nlg should always be overridden')

    def produce(self):
        # generate actual sub-result using operands - may need to pass in generator state
        raise NotImplementedError('superclass method for produce should always be overridden')


class StringSelect(Operation):
    def __init__(self):
        super().__init__()
        # init_string = str(np.random.choice(CHAR_LIST))  # this is the old single char implementation
        choice_count = np.random.randint(1, 4)
        init_string = ''
        for _ in range(choice_count):
            init_string += str(np.random.choice(CHAR_LIST))

        self.value = init_string

    def nlg(self):
        return "the string '" + self.value.strip() + "'"

    def produce(self):
        return self.value

    def signature(self):
        return 'StringSelect-' + self.value


class RepeatStringAsString(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.duplicate_count = np.random.randint(2, 4)
        self.value = self.target_value*self.duplicate_count

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        targets = [(i, x) for i, x in enumerate(g.state) if type(x) == str]
        choice = np.random.choice(len(targets))
        return targets[choice]

    def nlg(self):
        index_nlg = ' ' + reference_dag_form(self.target_index) + ' '
        duplicate_nlg = str(self.duplicate_count) + ' '
        component_1_nlg = [
            "repeat",
            "duplicate",
            "can you repeat",
            "please duplicate"
        ]
        component_2_nlg = [
            'times',
            'times as a string',
            'times in a row',
        ]
        selected_component_1 = random_choice_from_list(component_1_nlg)
        selected_component_2 = random_choice_from_list(component_2_nlg)
        return selected_component_1 + index_nlg + duplicate_nlg + selected_component_2

    def produce(self):
        return self.value

    def signature(self):
        return 'RepeatStringAsString-' + str(self.target_index)


class ListFromString(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.value = [self.target_value]

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        targets = [(i, x) for i, x in enumerate(g.state) if type(x) == str]
        choice = np.random.choice(len(targets))
        return targets[choice]

    def nlg(self):
        index_nlg = ' ' + reference_dag_form(self.target_index)
        component_1_nlg = [
            "make",
            "create",
            "return",
            "instantiate",
            "generate",
            "form",
            "can i get",
            "please make"
        ]
        component_2_nlg = [
            ' a list',
            ' a list',
            ' a list',
            ' a sequence of strings',
            ' an array',
            ' an array',
        ]
        component_3_nlg = [
            ' from',
            ' using',
            ' with'
        ]
        selected_component_1 = random_choice_from_list(component_1_nlg)
        selected_component_2 = random_choice_from_list(component_2_nlg)
        selected_component_3 = random_choice_from_list(component_3_nlg)
        return selected_component_1 + selected_component_2 + selected_component_3 + index_nlg

    def produce(self):
        return self.value

    def signature(self):
        return 'ListFromString-' + str(self.target_index)


class RepeatStringAsList(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.duplicate_count = np.random.randint(2, 4)
        self.value = [self.target_value]*self.duplicate_count

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        targets = [(i, x) for i, x in enumerate(g.state) if type(x) == str]
        choice = np.random.choice(len(targets))
        return targets[choice]

    def nlg(self):
        index_nlg = ' ' + reference_dag_form(self.target_index) + ' '
        duplicate_nlg = str(self.duplicate_count) + ' '
        component_1_nlg = [
            "repeat",
            "duplicate",
            "can you repeat",
            "please duplicate"
        ]
        component_2_nlg = [
            'times as ',
            'times into ',
            'times in a row as ',
            'times in a row into '
        ]
        component_3_nlg = [
            'a list',
            'a list',
            'a list',
            'a sequence of strings',
            'an array',
            'an array'
        ]
        selected_component_1 = random_choice_from_list(component_1_nlg)
        selected_component_2 = random_choice_from_list(component_2_nlg)
        selected_component_3 = random_choice_from_list(component_3_nlg)
        return selected_component_1 + index_nlg + duplicate_nlg + selected_component_2 + selected_component_3

    def produce(self):
        return self.value

    def signature(self):
        return 'RepeatStringAsList-' + str(self.target_index) + '-' + str(self.duplicate_count)


class ConcatenateStringsToString(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.value = ''.join(self.target_value)

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        candidates = [(i, x) for i, x in enumerate(g.state) if type(x) == str]
        targets = []

        # keep going until at least 2 targets selected
        # TODO: there is probably a np.random function to do all of this
        while len(targets) < 2:
            for c in candidates:
                # ignore targets already selected
                if c in targets:
                    continue

                # select with some random probability
                if np.random.rand() > 0.5:
                    targets.append(c)

        np.random.shuffle(targets)

        target_indices, target_values = [], []
        for t in targets:
            target_indices.append(t[0])
            target_values.append(t[1])

        return target_indices, target_values

    def nlg(self):
        index_nlg = ' ' + nlg_multiple_references(self.target_index) + ' '
        component_1_nlg = [
            "concatenate",
            "add",
            "fuse",
            "add together",
            "fuse together",
            "put together",
            "merge"
        ]
        component_2_nlg = [
            'as a single string',
            'as one string',
            'as a string'
        ]

        selected_component_1 = random_choice_from_list(component_1_nlg)
        selected_component_2 = random_choice_from_list(component_2_nlg)

        return selected_component_1 + index_nlg + selected_component_2

    def produce(self):
        return self.value

    def signature(self):
        return 'ConcatenateStringsToString-' + str(self.target_index)


class ConcatenateStringsToList(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.value = self.target_value

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        candidates = [(i, x) for i, x in enumerate(g.state) if type(x) == str]
        targets = []

        # keep going until at least 2 targets selected
        # TODO: there is probably a np.random function to do all of this
        while len(targets) < 2:
            for c in candidates:
                # ignore targets already selected
                if c in targets:
                    continue

                # select with some random probability
                if np.random.rand() > 0.5:
                    targets.append(c)

        np.random.shuffle(targets)

        target_indices, target_values = [], []
        for t in targets:
            target_indices.append(t[0])
            target_values.append(t[1])

        return target_indices, target_values

    def nlg(self):
        index_nlg = ' ' + nlg_multiple_references(self.target_index) + ' '
        component_1_nlg = [
            "concatenate",
            "add",
            "fuse",
            "add together",
            "fuse together",
            "put together",
            "merge"
        ]
        component_2_nlg = [
            'as a list',
            'as a single list',
            'as one list',
            'as a sequence of strings',
            'as an array',
            'as a single array',
            'as one array',
        ]

        selected_component_1 = random_choice_from_list(component_1_nlg)
        selected_component_2 = random_choice_from_list(component_2_nlg)

        return selected_component_1 + index_nlg + selected_component_2

    def produce(self):
        return self.value

    def signature(self):
        return 'ConcatenateStringsToList-' + str(self.target_index)


class ListReversal(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.value = self.target_value[::-1]

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        targets = [(i, x) for i, x in enumerate(g.state) if type(x) == list and len(x) > 1 and len(list(set(x))) > 1]
        choice = np.random.choice(len(targets))
        return targets[choice]

    def nlg(self):
        index_nlg = ' ' + reference_dag_form(self.target_index)
        component_1_nlg = [
            "invert the order of",
            "reverse the order of",
            "reverse",
            "return a reversed",
            "can i get a reversed",
            "please reverse"
        ]
        selected_component_1 = random_choice_from_list(component_1_nlg)
        return selected_component_1 + index_nlg

    def produce(self):
        return self.value

    def signature(self):
        return 'ListReversal-' + str(self.target_index)


class StringReversal(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.value = self.target_value[::-1]

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        targets = [(i, x) for i, x in enumerate(g.state) if type(x) == str and len(x) > 1 and len(list(set(x))) > 1]
        choice = np.random.choice(len(targets))
        return targets[choice]

    def nlg(self):
        index_nlg = ' ' + reference_dag_form(self.target_index)
        component_1_nlg = [
            "invert the order of",
            "reverse the order of",
            "reverse",
            "return a reversed",
            "can i get a reversed",
            "please reverse"
        ]
        selected_component_1 = random_choice_from_list(component_1_nlg)
        return selected_component_1 + index_nlg

    def produce(self):
        return self.value

    def signature(self):
        return 'StringReversal-' + str(self.target_index)


class ListSwapIndices(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.index_1 = np.random.randint(len(self.target_value))
        self.index_2 = self.index_1
        while self.index_2 == self.index_1:
            self.index_2 = np.random.randint(len(self.target_value))

        element_temp = self.target_value[self.index_1]
        self.value = self.target_value[:]
        self.value[self.index_1] = self.value[self.index_2]
        self.value[self.index_2] = element_temp

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        targets = [(i, x) for i, x in enumerate(g.state) if type(x) == list and len(x) > 1 and len(list(set(x))) > 1]
        choice = np.random.choice(len(targets))
        return targets[choice]

    def nlg(self):
        index_nlg = ' ' + reference_dag_form(self.target_index)
        swap_nlg = ' ' + str(self.index_1) + ' and ' + str(self.index_2)
        component_1_nlg = [
            "swap indices",
            "swap elements",
            "change elements at indices",
            "switch out indices",
            "switch out elements"
        ]
        component_2_nlg = [
            " for",
            " of",
            " for the list",
            " of the array",
            " for the array",
            " of the list"
        ]
        selected_component_1 = random_choice_from_list(component_1_nlg)
        selected_component_2 = random_choice_from_list(component_2_nlg)
        return selected_component_1 + swap_nlg + selected_component_2 + index_nlg

    def produce(self):
        return self.value

    def signature(self):
        return 'ListSwapIndices-' + str(self.target_index) + '-' + str(self.index_1) + '-' + str(self.index_2)


class ListDeleteIndex(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.del_index = np.random.randint(len(self.target_value))

        self.value = self.target_value[:]
        del self.value[self.del_index]

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        targets = [(i, x) for i, x in enumerate(g.state) if type(x) == list and len(x) > 1 and len(list(set(x))) > 1]
        choice = np.random.choice(len(targets))
        return targets[choice]

    def nlg(self):
        index_nlg = ' ' + reference_dag_form(self.target_index)
        del_nlg = ' ' + str(self.del_index)
        component_1_nlg = [
            "delete index",
            "delete element",
            "get rid of index",
            "please get rid of element",
            "deletee element",
            "destroy index"
        ]
        component_2_nlg = [
            " for",
            " of",
            " for the list",
            " of the array",
            " for the array",
            " of the list"
        ]
        selected_component_1 = random_choice_from_list(component_1_nlg)
        selected_component_2 = random_choice_from_list(component_2_nlg)
        return selected_component_1 + del_nlg + selected_component_2 + index_nlg

    def produce(self):
        return self.value

    def signature(self):
        return 'ListDeleteIndex-' + str(self.target_index) + '-' + str(self.del_index)


class ListSliceDeletion(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.index_1 = np.random.randint(len(self.target_value))
        self.index_2 = self.index_1
        while self.index_2 == self.index_1:
            self.index_2 = np.random.randint(len(self.target_value))

        if self.index_1 < self.index_2:
            self.value = self.target_value[:self.index_1] + self.target_value[self.index_2:]
        else:
            self.value = self.target_value[:self.index_2] + self.target_value[self.index_1:]

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        targets = [(i, x) for i, x in enumerate(g.state) if type(x) == list and len(x) > 2 and len(list(set(x))) > 1]
        choice = np.random.choice(len(targets))
        return targets[choice]

    def nlg(self):
        index_nlg = ' ' + reference_dag_form(self.target_index)
        if self.index_1 < self.index_2:
            del_nlg = ' ' + str(self.index_1) + ' to ' + str(self.index_2)
        else:
            del_nlg = ' ' + str(self.index_2) + ' to ' + str(self.index_1)

        component_1_nlg = [
            "delete the range of",
            "delete span represented by",
            "deletee the span represented by",
            "delete the slice of",
            "delete the slice from",
            "delete the span of",
            "delete the range from indices"
        ]
        component_2_nlg = [
            " for",
            " of",
            " for the list",
            " of the array",
            " for the array",
            " of the list"
        ]
        selected_component_1 = random_choice_from_list(component_1_nlg)
        selected_component_2 = random_choice_from_list(component_2_nlg)
        return selected_component_1 + del_nlg + selected_component_2 + index_nlg

    def produce(self):
        return self.value

    def signature(self):
        return 'ListSliceDeletion-' + str(self.target_index) + '-' + str(self.index_1) + '-' + str(self.index_2)


class ListSliceReversal(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.index_1 = np.random.randint(len(self.target_value))
        self.index_2 = self.index_1
        while self.index_2 == self.index_1:
            self.index_2 = np.random.randint(len(self.target_value))

        if self.index_1 < self.index_2:
            self.value = self.target_value[:self.index_1] + self.target_value[self.index_1:self.index_2:-1] + self.target_value[self.index_2:]
        else:
            self.value = self.target_value[:self.index_2] + self.target_value[self.index_2:self.index_1:-1] + self.target_value[self.index_1:]

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        targets = [(i, x) for i, x in enumerate(g.state) if type(x) == list and len(x) > 2 and len(list(set(x))) > 1]
        choice = np.random.choice(len(targets))
        return targets[choice]

    def nlg(self):
        index_nlg = ' ' + reference_dag_form(self.target_index)
        if self.index_1 < self.index_2:
            del_nlg = ' ' + str(self.index_1) + ' to ' + str(self.index_2)
        else:
            del_nlg = ' ' + str(self.index_2) + ' to ' + str(self.index_1)
        component_1_nlg = [
            "invert the order of",
            "reverse the order of",
            "reverse",
            "please reverse"
        ]
        component_2_nlg = [
            " the range of",
            " the span represented by",
            " the slice of",
            " the slice from",
            " the span of",
            " the range from"
        ]
        component_3_nlg = [
            " for",
            " of",
            " for the list",
            " of the array",
            " for the array",
            " of the list"
        ]
        selected_component_1 = random_choice_from_list(component_1_nlg)
        selected_component_2 = random_choice_from_list(component_2_nlg)
        selected_component_3 = random_choice_from_list(component_3_nlg)
        return selected_component_1 + selected_component_2 + del_nlg + selected_component_3 + index_nlg

    def produce(self):
        return self.value

    def signature(self):
        return 'ListSliceReversal-' + str(self.target_index) + '-' + str(self.index_1) + '-' + str(self.index_2)


class ListFlatten(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.value = ''.join(self.target_value)

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        targets = [(i, x) for i, x in enumerate(g.state) if type(x) == list and len(x) > 1 and len(list(set(x))) > 1]
        choice = np.random.choice(len(targets))
        return targets[choice]

    def nlg(self):
        index_nlg = ' ' + reference_dag_form(self.target_index)
        component_1_nlg = [
            "flatten",
            "flatten the list",
            "return a flattened",
            "can i get a flattened",
            "please flatten"
        ]
        selected_component_1 = random_choice_from_list(component_1_nlg)
        return selected_component_1 + index_nlg

    def produce(self):
        return self.value

    def signature(self):
        return 'ListFlatten-' + str(self.target_index)


class ConcatenateListsToList(Operation):
    def __init__(self, generator_object):
        super().__init__()
        self.target_index, self.target_value = self.get_target(generator_object)
        self.value = []
        for list_ in self.target_value:
            self.value += list_

    def get_target(self, g):
        # takes in a generator, finds suitable targets for the operation, and returns the target at random
        candidates = [(i, x) for i, x in enumerate(g.state) if type(x) == list]
        targets = []

        # keep going until at least 2 targets selected
        # TODO: there is probably a np.random function to do all of this
        while len(targets) < 2:
            for c in candidates:
                # ignore targets already selected
                if c in targets:
                    continue

                # select with some random probability
                if np.random.rand() > 0.5:
                    targets.append(c)

        np.random.shuffle(targets)

        target_indices, target_values = [], []
        for t in targets:
            target_indices.append(t[0])
            target_values.append(t[1])

        return target_indices, target_values

    def nlg(self):
        index_nlg = ' ' + nlg_multiple_references(self.target_index) + ' '
        component_1_nlg = [
            "concatenate",
            "add",
            "fuse",
            "add together",
            "fuse together",
            "put together",
            "merge"
        ]
        component_2_nlg = [
            'as a single list',
            'as one list',
            'as a list',
            'as a single array',
            'as one array',
            'as an array'
        ]

        selected_component_1 = random_choice_from_list(component_1_nlg)
        selected_component_2 = random_choice_from_list(component_2_nlg)

        return selected_component_1 + index_nlg + selected_component_2

    def produce(self):
        return self.value

    def signature(self):
        return 'ConcatenateListsToList-' + str(self.target_index)


class Generator:
    def __init__(self):
        initial_operation = StringSelect()
        self.operations = [initial_operation]
        self.state = [initial_operation.produce()]
        self.referenced_steps = []
        self.last_dag = None
        self.generate()

    def generate(self):
        while self.step():
            pass

    def remove_redundant_references(self):
        self.referenced_steps = list(set(self.referenced_steps))

    def step(self):
        # terminate generation OR produce another operation
        if self.terminate_on_length():
            return False
        elif self.terminate_on_DAG():
            self.last_dag = self.export()

        # calculate available actions
        action = self.choose_action()
        operation = eval(action)
        self.operations.append(operation)
        self.state.append(operation.produce())
        if type(operation.target_index) == list:
            self.referenced_steps += operation.target_index
        else:
            # type is int
            self.referenced_steps.append(operation.target_index)
        self.remove_redundant_references()

        return True

    def choose_action(self):
        actions = ['StringSelect()', 'RepeatStringAsString(self)', 'ListFromString(self)', 'RepeatStringAsList(self)']
        # TODO: clean below
        # conditional actions go here
        string_count = 0
        string_reversal, list_reversal = False, False
        list_2_bool = False
        list_count = 0

        for s in self.state:
            if type(s) == str:
                string_count += 1
                if len(s) > 1 and len(list(set(s))) > 1:
                    string_reversal = True
            elif type(s) == list:
                list_count += 1
                if len(s) > 1 and len(list(set(s))) > 1:
                    list_reversal = True
                if len(s) > 2 and len(list(set(s))) > 1:
                    list_2_bool = True

        if string_count > 1:
            actions += ['ConcatenateStringsToString(self)', 'ConcatenateStringsToList(self)']
            actions += ['ConcatenateStringsToString(self)']*2
        if list_count > 1:
            actions += ['ConcatenateListsToList(self)']
            actions += ['ConcatenateListsToList(self)']*2
        if string_reversal:
            actions += ['StringReversal(self)']
        if list_reversal:
            actions += ['ListReversal(self)', 'ListSwapIndices(self)', 'ListDeleteIndex(self)', 'ListFlatten(self)']
        if list_2_bool:
            actions += ['ListSliceDeletion(self)', 'ListSliceReversal(self)']

        choice = random_choice_from_list(actions)
        return choice

    def terminate_on_length(self):
        # termination conditions for when to stop generating
        if len(self.state) > 25:
            return True

        # prevent too long of strings from being generated
        for s in self.state:
            if type(s) == str and len(s) > 10:
                return True

        signatures = [op.signature() for op in self.operations]
        if len(list(set(signatures))) != len(signatures):
            return True

        return False

    def terminate_on_DAG(self):
        # termination condition to accept a state
        # check that DAG condition is satisfied and there is at least one list in state

        if len(self.state) == len(self.referenced_steps) + 1:
            for s in self.state:
                if type(s) == list:
                    return True
            # if type(self.state[-1]) == list:
            #     return True
        return False

    def export(self):
        # called on DAG confirmation - no need to save referenced steps
        return {
            # 'operations': list(self.operations),
            'state': list(self.state),
            'nlg': [op.nlg() for op in self.operations]
        }

# distribution info:
output_avg_token_len = []
output_avg_len = []
output_avg_uq_tokens = []
output_total_uq_tokens = []

input_avg_step_len = []
input_total_uq_tokens = []
input_total_uq_steps = []


attempts = 0
# max_len, max_len_dag = 0, None
distribution = {k: 0 for k in range(1, 35)}
data = []
# n_generations = 100000
# while attempts < n_generations:
#     gen_example = Generator()
#     if gen_example.last_dag is not None:
#         # distribution info:
#         output_ = gen_example.last_dag['state'][-1]
#         input_ = gen_example.last_dag['nlg']
#
#         output_avg_token_len.append(len(''.join(output_))/len(output_))
#         output_avg_len.append(len(output_))
#         output_avg_uq_tokens.append(len(list(set(output_))))
#         output_total_uq_tokens += output_
#
#         step_lens = [len(x.split()) for x in input_]
#         input_avg_step_len += step_lens
#         input_total_uq_tokens += (' '.join(input_)).split()
#         input_total_uq_steps += input_
#
#         distribution[len(gen_example.last_dag['state'])] += 1
#
#         # add to data structure
#         data.append(gen_example.last_dag)
#
#         # if len(gen_example.last_dag['state']) > max_len:
#         #     max_len = len(gen_example.last_dag['state'])
#         #     max_len_dag = gen_example.last_dag
#
#         attempts += 1

qdmr_distribution = {
    1: 0, 2: 718, 3: 1750, 4: 2635, 5: 2027, 6: 1361, 7: 836, 8: 387, 9: 150, 10: 57, 11: 26,
    12: 13, 13: 4, 14: 1, 15: 1, 16: 1, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0,
    25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0
}

expanded_qdmr_distribution = {
    1: 0, 2: 718, 3: 3150, 4: 4035, 5: 3527, 6: 2461, 7: 1536, 8: 787, 9: 450, 10: 157, 11: 76,
    12: 43, 13: 14, 14: 5, 15: 5, 16: 5, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0,
    25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0
}
qdmr_distribution = {k: expanded_qdmr_distribution[k]*20 for k in expanded_qdmr_distribution.keys()}
print(qdmr_distribution)

def distribution_satisfied(dist):
    for i in range(2, 17):
        if dist[i] != qdmr_distribution[i]:
            return False
    return True


def distribution_add(example, dist):
    steps = len(example)
    if steps > 16:
        return False
    return dist[steps] < qdmr_distribution[steps]


while not distribution_satisfied(distribution):
    gen_example = Generator()
    if gen_example.last_dag is not None:
        # distribution info:
        output_ = gen_example.last_dag['state'][-1]
        input_ = gen_example.last_dag['nlg']

        if distribution_add(input_, distribution):
            output_avg_token_len.append(len(''.join(output_))/len(output_))
            output_avg_len.append(len(output_))
            output_avg_uq_tokens.append(len(list(set(output_))))
            output_total_uq_tokens += output_

            step_lens = [len(x.split()) for x in input_]
            input_avg_step_len += step_lens
            input_total_uq_tokens += (' '.join(input_)).split()
            input_total_uq_steps += input_

            distribution[len(input_)] += 1

            # add to data structure
            data.append(gen_example.last_dag)

print('\n new dist', distribution)

total = sum(distribution.values())
print(total)
for k in distribution.keys():
    distribution[k] = round(distribution[k]/total*100, 2)

print(distribution)

print('output_avg_token_len', sum(output_avg_token_len)/len(output_avg_token_len))
print('output_avg_len', sum(output_avg_len)/len(output_avg_len))
print('output_avg_uq_tokens', sum(output_avg_uq_tokens)/len(output_avg_uq_tokens))
print('output_total_uq_tokens', len(list(set(output_total_uq_tokens))))
print('input_avg_step_len', sum(input_avg_step_len)/len(input_avg_step_len))
print('input_total_uq_tokens', len(list(set(input_total_uq_tokens))))
print('input_total_uq_steps', len(list(set(input_total_uq_steps))))

with open('list_task_v2.json', 'w') as f:
    json.dump(data, f)