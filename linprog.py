import numpy as np
import re
from colorama import Fore, init
from utils import get_all_occ, sorter1
from linparse import ObjectiveFunction, Constraint, Constraints


"""
Calculate a formed linear programming problem using the simplex method

## TODO
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
        self._get_vars()
        
        self.z = 0

        # minimize or maximize function
        self.opt = None
        self._opt = None

        self._set_optimize(max) if self.objfunc.optimize == "max" else self._set_optimize(min)

        # number of constraints
        self.conlen = len(self.A)

        # Basic variables
        self.vbs = [[0]*self.conlen, [0]*self.conlen]

        # constant Cj
        self.cj = [self.varnames, self.objfunc.z + [0] * (len(self.varnames) - len(self.objfunc.z))]

        # constant cjdict
        self.cjdict = {var: coef for var, coef in zip(self.cj[0], self.cj[1])}

        # working cjdict
        self._cjdict = None

        # working cj
        self._cj = None

        # Cj - Zj
        self.cj_zj = None

        # if there is artificial variables
        self.phase = 0

        # Ratio column (θ)
        self._ratio_column = None

        # pivot infos
        self._pivcol = None
        self._pivrow = None
        self._pivot  = None

        # number of iterations
        self.iterations = 0

    def _set_optimize(self, opt):
        self._opt = self.opt
        self.opt = opt
        self.is_not_optimized = self.is_not_maximized if opt is max else self.is_not_minimized

    def _detect_basic_vars(self):
        for i, column in enumerate(self.A.T):
            if 1 in column and np.count_nonzero(column) == 1:
                pos = np.where(column)[0][0]
                basicvar = self.varnames[i]
                self.vbs[0][pos] = basicvar
                self.vbs[1][pos] = self._cjdict[basicvar]

    def _get_vars(self):
        """Generates all basic and non-basic variables from both the objective function and constraints."""
        self.varnames = sorted(list(set(self.constraints.varnames) | set(self.objfunc.varnames)), key=sorter1)

    def is_2phase_method(self):
        """Checks if there is a two-phase method"""
        return self.constraints.artificials
        
    def init_mat(self):
        """Initialize start-up tableau"""

        # Basic variables
        if self.is_2phase_method():
            self.phase = 1
            # maximize -Ai
            self._opt = self.opt
            self._set_optimize(max)
            artificial_len = len(self.constraints.artificials)
            self._cj = [self.varnames, [0]*(len(self.varnames) - artificial_len) + [-1]*artificial_len]
        else:
            # make copy of original cj
            self._cj = [self.cj[0].copy(), self.cj[1].copy()]
        
        self._cjdict = {var: coef for var, coef in zip(self._cj[0], self._cj[1])}
        # detect initial basic variables and fill vbs
        self._detect_basic_vars()

        self.A = np.array(self.A)
        
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
        for icolumn, column in enumerate(self.A.T):
            total = 0
            for irow, cell in enumerate(column):
                total -= self.vbs[1][irow] * cell
            total += self._cj[1][icolumn]
            final.append(total)
        self.cj_zj = final

    def _get_pivot_column(self):
        self._cj_zj()
        # get all index occurences of optimum of cj-zj row
        optimum = self.opt(self.cj_zj)
        all_occ = get_all_occ(self.cj_zj, optimum)
        # if only one optimum, than just choose it
        if len(all_occ) == 1 or self.cj_zj[all_occ[0]] == 0:
            self._pivcol = all_occ[0]
            self._pivot_column = self.A.T[self._pivcol]
        else:
            # 2 or more cj-zj's have the same value
            all_cands = []
            for candidate in all_occ:
                self._pivot_column = self.A.T[candidate]
                all_cands.append((candidate, min(self._get_ratio_column())))

            self._pivcol = min(all_cands, key=lambda k: k[1])[0]
            self._pivot_column = self.A.T[self._pivcol]

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
        return self._ratio_column

    def _get_pivot_row(self):
        self._get_ratio_column()
        # get all index occurences of minimum of ratio column
        minimum = min(self._ratio_column)
        all_occ = get_all_occ(self._ratio_column, minimum)
        if len(all_occ) == 1:
            self._pivrow = all_occ[0]
        else:
            # 2 or more ratios have the same value
            target_rows = [ self.A[index] for index in all_occ ]
            row_sums = [ sum(row) for row in target_rows ]
            min_row_sum = row_sums.index(min(row_sums))
            self._pivrow = all_occ[min_row_sum]
                 
    def _get_pivot(self):
        """Finding the pivot"""
        self._get_pivot_column()
        self._get_pivot_row()
        self._pivot = self.A[self._pivrow, self._pivcol]

    def _update_vbs(self):
        """Updates the basic variables after finding the pivot"""
        self.vbs[1][self._pivrow] = self._cj[1][self._pivcol]
        self.vbs[0][self._pivrow] = self._cj[0][self._pivcol]

    def show_current(self):
        """Shows current simplex tableau"""
        # cj
        print(f"==========={'MAXIMIZATION' if self.opt is max else 'MINIMIZATION'}===========")
        phase = "" if self.phase == 0 else f" PHASE {self.phase} "
        print("-"*45 + f" Iteration {self.iterations:<3}{phase}" + "-"*45)
        print("   {}{:6}".format(WHITE, "Cj"), end='')
        print("    ", end='')
        for item in self._cj[1]:
            print(f"{WHITE}{item:7.2f}{WHITE}", end='  ')
        print("    " + 8*" " + " ")
        # -------------------------
        print(" "*9 + "VB", end='      ')
        for var in self.varnames:
            print(f"{var:6}", end='   ')
        print("b        θ")
        # -------------------------
        for vb_vars, vb_values, row, b, ratio in zip(self.vbs[0], self.vbs[1], enumerate(self.A), self.b, self._ratio_column):
            irow, row = row
            print(f"{WHITE}{vb_values:7.2f}{WHITE}", end='  ')
            print(f"{WHITE}{vb_vars:2}{WHITE}", end='  ')
            for icolumn, cell in enumerate(row):
                color = PIVOT_ROW_COLOR if irow == self._pivrow else WHITE
                if icolumn == self._pivcol:
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
        b = self.b[self._pivrow]
        for irow, cell in enumerate(self.b):
            # x = x - (a*b) / pivot
            a = self.A[irow, self._pivcol]
            self.b[irow] = self.b[irow] - a*b / self._pivot
        self.b[self._pivrow] = b / self._pivot

    def next_iter(self):
        """Calculates the next simplex iterations"""
        self.iterations += 1

        # define new A
        next_A = self.A.copy()

        # pivot row calculation
        next_A[self._pivrow] /= self._pivot
        # pivot column calculation
        next_A[:, self._pivcol] = 0
        # set pivot to 1
        next_A[self._pivrow, self._pivcol] = 1

        for irow, row in enumerate(self.A):
            a = self.A[irow, self._pivcol]
            for icolumn, column in enumerate(row):
                if irow == self._pivrow and icolumn == self._pivcol:
                    continue
                elif irow == self._pivrow:
                    continue
                elif icolumn == self._pivcol:
                    continue
                else:
                    # x = x - (a*b) / pivot
                    b = self.A[self._pivrow, icolumn]
                    next_A[irow, icolumn] = self.A[irow, icolumn] - ( ( a * b ) / self._pivot )

        self._update_b()

        self.A = next_A

        # search for next pivot
        self._get_pivot()

        # update basic variables
        self._update_vbs()

        # calculate z
        self._calc_z()

    def is_not_maximized(self):
        for item in self.cj_zj:
            if item > 0:
                return True

    def is_not_minimized(self):
        for item in self.cj_zj:
            if item < 0:
                return True

    def print_result(self):
        """Prints the final result ( not very beautiful )"""
        if self.is_not_optimized():
            raise NotSolutionError("Solution not reached yet.")
        else:
            print("-"*25, "RESULT", "-"*25, sep="-")
            for vb, value in zip(self.vbs[0], self.b):
                print(" "*20, vb, "=", f"{value:10.3f}", " "*20, end=' |\n')
            print(" "*20, f"{self.objfunc.fname} ", "=", f"{self.z:10.3f}", " "*20, end=' |\n')

    def calc(self, verbose=True, init=True, show_first=True, show_result=True):
        # this is basically the main function
        if init:
            self.init_mat()
        if verbose and show_first:
            self.show_current()
        while self.is_not_optimized():
            self.next_iter()
            if verbose:
                self.show_current()

        if self.phase == 1:
            if not self._calc_two_phase():
                return
        if show_result:
            self.print_result()

    def silent_calc(self):
        self.init_mat()
        while self.is_not_optimized():
            self.next_iter()
        if self.phase == 1:
            if not self._calc_two_phase():
                return


    def _calc_two_phase(self):
        if self.z != 0:
            print("There is no solution for this problem.")
        else:
            self.phase += 1
            self.iterations = 1
            # set original function
            self._set_optimize(self._opt)
            # replace original Cj
            self._cj = self.cj
            # remove artificial variables now
            lenart = len(self.constraints.artificials)
            self.A = self.A[:, :-lenart]
            # remove from cj aswell
            for iterator in self._cj:
                for i in range(lenart):
                    iterator.pop(-1)

            # replace real coefficients of basic variables
            for i, var in enumerate(self.vbs[0]):
                real_coeff = self.cjdict[var]
                self.vbs[1][i] = real_coeff

            # calculate new pivot
            self._get_pivot()

            # new iteration
            self.iterations += 1

            self.calc(init=False, show_first=False, show_result=False)
            return True


if __name__ == "__main__":

    from tests import *

    test6()
    


