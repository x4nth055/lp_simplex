import numpy as np
import re
from colorama import Fore, init
from utils import get_all_occ, sorter1
from linparse import ObjectiveFunction, Constraint, Constraints


"""
Calculate a formed linear programming problem using the simplex method

## TODO
    - Upper-bound inequality '>=' not yet implemented ( even though `linparse.Constraint` gives info about artificial variables -- where they are -- )
    - Refers to the first, the two-phase method isn't completed.
    - Equality constraints
    - Initial dual and final dual
    - Sensitivity analysis
"""


init()

WHITE = Fore.WHITE
PIVOT_COLUMN_COLOR = Fore.CYAN
PIVOT_ROW_COLOR = Fore.YELLOW
PIVOT_COLOR = Fore.RED
Z_COLOR = Fore.GREEN

class NotSolutionError(Exception):
    def __init__(self, message):
        self.message = message


class LinearProgramming:

    def __init__(self, objfunc, constraints):

        self.constraints = constraints
        self.objfunc = objfunc

        self.A = self.constraints.a
        self.b = self.constraints.b

        self.varnames = None
        
        self.z = 0

        if self.objfunc.optimize == 'max':
            self.is_not_optimized = self.is_not_maximized
            self.opt = max
        else:
            self.is_not_optimized = self.is_not_minimized
            self.opt = min

        # Basic variables
        if self.is_2phase_method():
            self.vbs = [[self.constraints.varnames[i] for i in self.constraints.artificials], [-1 for i in self.constraints.artificials]]
            self.cj = [self.vbs[0].copy(), self.vbs[1].copy()]
        else:
            self.vbs = [[self.constraints.varnames[i] for i in self.constraints.ecarts], [0 for i in self.constraints.ecarts]]
            self.cj = [self.objfunc.varnames, self.objfunc.z]

        # Transposed A
        self.A_T = None

        # Cj - Zj
        self.cj_zj = None

        # Ratio column (θ)
        self._ratio_column = None

        # pivot infos
        self._pivot_column_index = None
        self._pivot_row_index    = None
        self._pivot_column       = None
        self._pivot_row          = None
        self._pivot              = None

        # number of iterations
        self.iterations = 0

        self._get_vars()

    def _get_vars(self):
        """Generates all basic and non-basic variables from both the objective function and constraints."""
        self.varnames = sorted(list(set(self.constraints.varnames) | set(self.objfunc.varnames)), key=sorter1)

    def is_2phase_method(self):
        """Checks if there is a two-phase method"""
        return self.constraints.artificials
        
    def init_mat(self):
        """Initialize start-up A matrix"""
        self.A = np.array(self.A)
        self.A_T = self.A.T
        self.cj[1] = np.hstack((self.cj[1], np.zeros((len(self.A[0]) - len(self.cj[1])))))
        self._get_pivot()
        self._calc_z()

    def _calc_z(self):
        total = 0
        for cj, b in zip(self.vbs[1], self.b):
            total += cj * b
        self.z = total
    
    def _cj_zj(self):
        """Calculates Cj-Zj"""
        final = []
        for icolumn, column in enumerate(self.A_T):
            total = 0
            for irow, cell in enumerate(column):
                total -= self.vbs[1][irow] * cell
            total += self.cj[1][icolumn]
            final.append(total)
        self.cj_zj = final

    def _get_pivot_column(self):
        self._cj_zj()
        # get all index occurences of optimum of cj-zj row
        optimum = self.opt(self.cj_zj)
        all_occ = get_all_occ(self.cj_zj, optimum)
        # if only one optimum, than just choose it
        if len(all_occ) == 1 or self.cj_zj[all_occ[0]] == 0:
            self._pivot_column_index = all_occ[0]
            self._pivot_column = self.A_T[self._pivot_column_index]
        else:
            # 2 cj-zj's have the same value
            # TODO
            raise Exception("Ambiguity choosing in get_pivot_column()")

    def _get_ratio_column(self):
        self._ratio_column = []
        for pivot_cell, b_cell in zip(self._pivot_column, self.b):
            if pivot_cell == 0 or (pivot_cell < 0 and b_cell >= 0):
                self._ratio_column.append(np.inf)
                continue
            target = b_cell / pivot_cell
            if target < 0:
                self._ratio_column.append(np.inf)
            else:
                self._ratio_column.append(target)

    def _get_pivot_row(self):
        self._get_ratio_column()
        # get all index occurences of minimum of ratio column
        minimum = min(self._ratio_column)
        all_occ = get_all_occ(self._ratio_column, minimum)
        if len(all_occ) == 1:
            self._pivot_row_index = all_occ[0]
        else:
            target_rows = [ self.A[index] for index in all_occ ]
            row_sums = [ sum(row) for row in target_rows ]
            min_row_sum = row_sums.index(min(row_sums))
            self._pivot_row_index = all_occ[min_row_sum]
        self._pivot_row = self.A[self._pivot_row_index]

    def _get_pivot(self):
        """Finding the pivot"""
        self._get_pivot_column()
        self._get_pivot_row()
        self._pivot = self.A[self._pivot_row_index, self._pivot_column_index]

    def _update_vbs(self):
        """Updates the basic variables after finding the pivot"""
        self.vbs[1][self._pivot_row_index] = self.cj[1][self._pivot_column_index]
        self.vbs[0][self._pivot_row_index] = self.cj[0][self._pivot_column_index]

    def show_current(self):
        """Shows current simplex tableau"""
        # cj
        print("-"*45 + f" Iteration {self.iterations:<3}" + "-"*45)
        print("   {}{:6}".format(WHITE, "Cj"), end='')
        print("    ", end='')
        for item in self.cj[1]:
            print(f"{WHITE}{item:7.2f}{WHITE}", end='  ')
        print("    " + 8*" " + " ")
        # -------------------------
        print(" "*9 + "VB", end='      ')
        for var in self.varnames:
            print(f"{var:6}", end='   ')

        print()
        # -------------------------
        for vb_vars, vb_values, row, b, ratio in zip(self.vbs[0], self.vbs[1], enumerate(self.A), self.b, self._ratio_column):
            irow, row = row
            print(f"{WHITE}{vb_values:7.2f}{WHITE}", end='  ')
            print(f"{WHITE}{vb_vars:2}{WHITE}", end='  ')
            for icolumn, cell in enumerate(row):
                color = PIVOT_ROW_COLOR if irow == self._pivot_row_index else WHITE
                if icolumn == self._pivot_column_index:
                    if color == PIVOT_ROW_COLOR:
                        color = PIVOT_COLOR
                    else:
                        color = PIVOT_COLUMN_COLOR
                print(f"{color}{cell:7.2f}{WHITE}", end='  ')
            print(f"{color}{b:7.2f}{WHITE}", end='  ')
            print(f"{ratio:7.2f}")
        # cj-zj
        print("   {}{:6}".format(WHITE, "Cj-Zj"), end='    ')
        for item in self.cj_zj:
            print(f"{item:7.2f}", end='  ')
        print(f"{Z_COLOR}{self.z:7.2f}{WHITE}")

    def _update_b(self):
        b = self.b[self._pivot_row_index]
        for irow, cell in enumerate(self.b):
            # x = x - (a*b) / pivot
            a = self.A[irow, self._pivot_column_index]
            self.b[irow] = self.b[irow] - a*b / self._pivot
        self.b[self._pivot_row_index] = b / self._pivot

    def next_iter(self):
        """Calculates the next simplex iterations"""
        self.iterations += 1

        # define new A
        next_A = self.A.copy()

        # pivot row calculation
        next_A[self._pivot_row_index] /= self._pivot
        # pivot column calculation
        next_A[:, self._pivot_column_index] = 0
        # set pivot to 1
        next_A[self._pivot_row_index, self._pivot_column_index] = 1

        for irow, row in enumerate(self.A):
            a = self.A[irow, self._pivot_column_index]
            for icolumn, column in enumerate(row):
                if irow == self._pivot_row_index and icolumn == self._pivot_column_index:
                    continue
                elif irow == self._pivot_row_index:
                    continue
                elif icolumn == self._pivot_column_index:
                    continue
                else:
                    # x = x - (a*b) / pivot
                    b = self.A[self._pivot_row_index, icolumn]
                    next_A[irow, icolumn] = self.A[irow, icolumn] - ( ( a * b ) / self._pivot )

        self._update_b()

        self.A = next_A
        self.A_T = self.A.T

        # search for next pivot
        self._get_pivot()

        # calculate z
        self._calc_z()

        # update basic variables
        self._update_vbs()

    def is_not_maximized(self):
        for item in self.cj_zj:
            if item > 0:
                return True

    def is_not_minimized(self):
        for item in self.cj_zj:
            if item <= 0:
                return True

    def print_result(self):
        """Prints the final result ( not very beautiful )"""
        if self.is_not_optimized():
            raise NotSolutionError("Solution not reached yet.")
        else:
            print("-"*25, "RESULT", "-"*25, sep="-")
            for vb, value in zip(self.vbs[0], self.b):
                if not vb.startswith("e") and not vb.startswith("a"):
                    print(" "*23, vb, "=", f"{value:.3f}", " "*23, end=' |\n')
            print(" "*23, "z", "=", f"{self.z:.3f}", " "*23, end=' |\n')

    def calc(self, verbose=True):
        # this is basically the main function
        self.init_mat()
        if verbose:
            self.show_current()
        while self.is_not_optimized():
            self.next_iter()
            if verbose:
                self.show_current()
        self.print_result()

    # def initial_dual(self):  # TODO
    #     A_dual = np.array(self.initial_A).T
    #     b_dual = self.initial_cj
    #     z_dual = self.initial_b

    #     return LinearProgramming(A_dual, b_dual, z_dual, maximize=self.opt == min)

    # def final_dual(self):  # TODO
    #     if self.is_not_maximized():
    #         raise NotSolutionError("Solution not reached yet.")
    #     else:
    #         # to be implemented
    #         pass


def test():
    objfunc = ObjectiveFunction("max z = 7t1 + 5t2 + 4t3")
    c1 = Constraint("t1 + t2 + t3 <= 25")
    c2 = Constraint("2t1 + t2 + t3 <= 40")
    c3 = Constraint("t1 + t2 <= 25")
    # c4 = Constraint("t2 + t3 = 15") this is invalid (equality not implemented yet)
    # c4 = Constraint("t2 + t3 >= 5") this is invalid (upper-bound uses artificial vars)
    c4 = Constraint("t2 + t3 <= 15")
    c = c1 + c2 + c3 + c4

    linprog = LinearProgramming(objfunc, c)
    linprog.calc()


if __name__ == "__main__":

    # objfunc = ObjectiveFunction("max z = 6x1 + 5x2")
    # c = Constraint("3x1 + 2x2 <= 12") + Constraint("x1 + x2 <= 5")

    test()
    


