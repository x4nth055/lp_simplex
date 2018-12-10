import re
import numpy as np
from utils import sorter1, sorter2

class _LinearParsing:

    OBJECTIVE_FUNCTION_SPLITTER = re.compile(r"(?P<op>\+|\-)")
    LINEAR_PATTERN = re.compile(r"^\s*(?P<op>[\-\+])?\s*(?P<a>\d*(\.\d*)?)\s*(?P<var>\w+\d*)\s*$")

    @staticmethod
    def parse_linear(items: list, varnames: list, z: list): 
        for item in items:
            match = _LinearParsing.LINEAR_PATTERN.fullmatch(item)
            if not match:
                raise ObjectiveFunction.NotValidObjectiveFunctionError(f"Right part of the objective function does not match: '{item}'")
            groupdict = match.groupdict()
            var = groupdict['var']
            operator = groupdict['op']
            number = groupdict['a']

            varnames.append(var)
            # set default operator to + ( easy and quick math )
            if not operator:
                operator = "+"

            # set default number to 1 when there isn't
            if not number:
                number = '1'

            number = float(operator + number)
            z.append(number)

    @staticmethod
    def fill_items(string: str, items: list):
        split_result = _LinearParsing.OBJECTIVE_FUNCTION_SPLITTER.split(string)
        if not split_result[0]:
            # remove '' at first.
            split_result.pop(0)
        if split_result[0] not in '-+':
            # not an operator, so we add '+' at first
            split_result.insert(0, '+')
        for operator_index, number_index in zip(range(0, len(split_result), 2), range(1, len(split_result), 2)):
            operator, number = split_result[operator_index].strip(), split_result[number_index].strip()
            items.append(operator + number)

    @staticmethod
    def remove_redundant_vars(varnames: list, z: list):

        def get_redundant_vars():
            # Thanks to Idriss.
            items = [[], []]
            for item in varnames:
                if item in items[1]:
                    continue
                occurrences = [i for i, x in enumerate(varnames) if x == item]
                if len(occurrences) > 1:
                    # print(f"Redundancy detected for {item} in {occurrences}")
                    items[0].append(tuple(occurrences))
                items[1].append(item)
            return items[0]

        redundant_vars = get_redundant_vars()

        indexes_to_be_removed = set()

        # print("Redundant vars: ", redundant_vars)
        for index1, index2 in redundant_vars:
            result = z[index1] + z[index2]
            indexes_to_be_removed.add(index2)
            z[index1] = result

        new_z = [ z for i, z in enumerate(z) if i not in indexes_to_be_removed ]
        new_vars = [ var for i, var in enumerate(varnames) if i not in indexes_to_be_removed ]

        varnames.clear()
        z.clear()

        z.extend(new_z)
        varnames.extend(new_vars)

    @staticmethod
    def _str_linear(z, varnames):
        returned = ""
        for c, varname in zip(z, varnames):
            if c > 0:
                returned += f" +{c} {varname}"
            elif c == 1:
                returned += f" {varname}"
            else:
                returned += f" {c} {varname}"
        return returned


class ObjectiveFunction:

    LEFT_PART_PATTERN  = re.compile(r"^\s*(?P<opt>min|max)\s+(?P<fname>\w+)\s*")

    def __init__(self, string):
        self.string = string

        self._left_part = None

        self._items = []

        self.z = []
        self.varnames = []

        self.optimize = None
        self.fname = None

        self.parse()

    def __str__(self):
        returned = f"{self.optimize} {self.fname} ="
        return returned + _LinearParsing._str_linear(self.z, self.varnames)
    
    def parse(self):
        self._split_and_fill_items()
        self._parse_left()
        self._parse_right()
        self._remove_redundant_vars()

    def _split_and_fill_items(self):
        """Splits `self.string` and fill `self._items`"""
        try:
            left, right = self.string.split("=")
        except ValueError:
            raise ObjectiveFunction.NotValidObjectiveFunctionError("An OR objective function must constain one and only one equality `=` character")

        self._left_part = left

        right = right.strip()
        if '+' not in right and '-' not in right:
            self._items.append(right)
            return
        
        _LinearParsing.fill_items(right, self._items)       
            
    def _parse_left(self):
        match = ObjectiveFunction.LEFT_PART_PATTERN.fullmatch(self._left_part)
        if not match:
            raise ObjectiveFunction.NotValidObjectiveFunctionError("Left part of the objective function does not match.")
        groupdict = match.groupdict()
        self.optimize = groupdict['opt']
        self.fname = groupdict['fname']

    def _parse_right(self):
        _LinearParsing.parse_linear(self._items, self.varnames, self.z)
    
    def _remove_redundant_vars(self):
        _LinearParsing.remove_redundant_vars(self.varnames, self.z)

    class NotValidObjectiveFunctionError(Exception):
        def __init__(self, message):
            self.message = message
    

class Constraint:

    ecarts_counter = 0
    artificial_counter = 0

    class NotValidConstraintError(Exception):
        def __init__(self, message):
            self.message = message

    SPLITTER_PATTERN = re.compile(r"(?P<op>=|<=|>=)")

    def __init__(self, string=None):
        self.string = string

        self._items = []

        self.a = []
        self.ecarts, self.artificial = None, None
        self.varnames = []

        self.op = None
        self.b = None

        self._right_part = None
        self._left_part  = None

        self.parse()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        returned = _LinearParsing._str_linear(self.a, self.varnames)
        returned += f" {self.op} {self.b}"
        return returned.strip()

    def parse(self):
        self._split_and_fill_items()
        self._parse_left()
        self._remove_redundant_vars()
        self._finalize_a_vars()

    def _finalize_a_vars(self):
        if self.b == 0:
            return
        if self.op == '=':
            return
        elif self.op == '<=':
            if self.b > 0:
                # add e1
                self.a.extend([1.0])
                self.varnames.extend([Constraint._gen_ecarts()])
                self.ecarts = True
            else:
                # multiply the RHS and LHS by -1 and change the operator
                self.a = [ item * -1 for item in self.a ]
                self.b *= -1
                # substract e1 and add a1
                self.a.extend([-1.0, 1.0])
                self.ecarts, self.artificial = True, len(self.a) - 1
                self.varnames.extend([Constraint._gen_ecarts(), Constraint._gen_artifical()])
        elif self.op == '>=':
            if self.b > 0:
                # substract e1 and add a1
                self.a.extend([-1.0, 1.0])
                self.ecarts, self.artificial = True, len(self.a) - 1
                self.varnames.extend([Constraint._gen_ecarts(), Constraint._gen_artifical()])
            else:
                self.a = [ item * -1 for item in self.a ]
                self.b *= -1
                # add e1
                self.ecarts = True
                self.a.extend([1.0])
                self.varnames.extend([Constraint._gen_ecarts()])
        self.op = "="
        # convert to numpy array
        self.a = np.array(self.a)

    def _split_and_fill_items(self):
        try:
            left, self.op, right = Constraint.SPLITTER_PATTERN.split(self.string)
        except ValueError:
            raise Constraint.NotValidConstraintError("An LP Constraint must contain one and only one of the following characters: ('=', '<=', '>=')")
        
        self._left_part = left
        right = right.strip()
        # must be a float.
        try:
            self.b = float(right)
        except ValueError:
            raise Constraint.NotValidConstraintError("Right part of a constraint isn't valid: " + right)

        _LinearParsing.fill_items(self._left_part, self._items)

    def _parse_left(self):
        _LinearParsing.parse_linear(self._items, self.varnames, self.a)

    def _remove_redundant_vars(self):
        _LinearParsing.remove_redundant_vars(self.varnames, self.a)

    def __add__(self, other):

        # define some parameters when its either `Constraint` or `Constraints` instance.
        if isinstance(other, Constraint):
            new_b = [self.b] + [other.b]
            other_numcols = 1
            other_constraints = [other]
        elif isinstance(other, Constraints):
            new_b = [self.b] + other.b
            other_numcols = len(other.a.T[0])
            other_constraints = other.constraints
        else:
            _type = type(other).__name__
            raise TypeError(f"Add (+) operation is not supported for `LP Constraint` and {_type}")

        varnames = other.varnames
        b = other.b
        a = other.a
        global_order = sorted(set(self.varnames) | set(varnames), key=sorter1)
        self_var_val = { k: v for k, v in zip(self.varnames, self.a.T)}
        other_var_val = { k: v for k, v in zip(varnames, a.T)}
        
        self_numcols = 1

        target = np.zeros((self_numcols + other_numcols, len(global_order)))

        for i, var in enumerate(global_order):
            if var in self_var_val:
                target.T[i, :self_numcols] = self_var_val[var]
            else:
                target.T[i, :self_numcols] = 0
            
            if var in other_var_val:
                target.T[i, self_numcols:] = other_var_val[var]
            else:
                target.T[i, self_numcols:] = 0

        new_constraints = [self] + other_constraints

        return Constraints(global_order, new_constraints, target, new_b)


    @staticmethod
    def _gen_ecarts():
        Constraint.ecarts_counter += 1
        return f"e{Constraint.ecarts_counter}"

    @staticmethod
    def _gen_artifical():
        Constraint.artificial_counter += 1
        return f"a{Constraint.artificial_counter}"


class Constraints:

    def __init__(self, vars, constraints, a, b):
        self.varnames = vars
        self.constraints = constraints
        self.a = a
        self.b = b

        # detect artificial variables
        self.artificials = [ i for i, var in enumerate(self.varnames) if var.startswith('a') ]
        # detect ecart variables
        self.ecarts = [ i for i, var in enumerate(self.varnames) if var.startswith('e') ]

    def __str__(self):
        returned = ""
        for constraint in self.constraints:
            returned += f"{constraint}\n"
        return returned.strip()

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):

        # define some parameters when its either `Constraint` or `Constraints` instance.
        if isinstance(other, Constraint):
            new_b = self.b + [other.b]
            other_numcols = 1
            other_constraints = [other]
        elif isinstance(other, Constraints):
            new_b = self.b + other.b
            other_numcols = len(other.a.T[0])
            other_constraints = other.constraints
        else:
            _type = type(other).__name__
            raise TypeError(f"Add (+) operation is not supported for `LP Constraint` and {_type}")

        varnames = other.varnames
        b = other.b
        a = other.a
        try:
            a.T
        except AttributeError:
            a = np.array(a)
            
        global_order = sorted(set(self.varnames) | set(varnames), key=sorter1)
        self_var_val = { k: v for k, v in zip(self.varnames, self.a.T)}
        other_var_val = { k: v for k, v in zip(varnames, a.T)}
        
        self_numcols = len(self.a.T[0])

        target = np.zeros((self_numcols + other_numcols, len(global_order)))

        for i, var in enumerate(global_order):
            if var in self_var_val:
                target.T[i, :self_numcols] = self_var_val[var]
            else:
                target.T[i, :self_numcols] = 0
            
            if var in other_var_val:
                target.T[i, self_numcols:] = other_var_val[var]
            else:
                target.T[i, self_numcols:] = 0

        return Constraints(global_order, self.constraints + other_constraints, target, new_b)


if __name__ == "__main__":

    c1 = Constraint("4x1 + 5x2 + x3 <= 4")
    c2 = Constraint("14x3 + 12x2 - x4 <= 3")
    c3 = Constraint("3x2 + 2 x3 + x4 >= 5")
    c4 = Constraint("-x2 + 5x3 <= -1")

    c = c1 + c2 + c3 + c4
    print(c.a)
    print("="*60)
    print(c.artificials)




