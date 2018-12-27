from linparse import Constraint, ObjectiveFunction
from linprog import LinearProgramming


def test1():
    objfunc = ObjectiveFunction("max z = 7t1 + 5t2 + 4t3")
    c1 = Constraint("t1 + t2 + t3 <= 25")
    c2 = Constraint("2t1 + t2 + t3 <= 100")
    c3 = Constraint("t1 + t2 >= -1")
    c4 = Constraint("t2 + t3 >= 5")
    c = c1 + c2 + c3 + c4

    linprog = LinearProgramming(objfunc, c)
    linprog.calc()


def test2():
    z = ObjectiveFunction("max Z = 24x1 + 20 x2")
    c = Constraint("x1 + x2 <= 30") + Constraint("x1 + 2 x2 >= 40")
    
    linprog = LinearProgramming(z, c)
    linprog.calc()


def test3_silent():
    # Interrogation
    z = ObjectiveFunction("max W = 7x1 + 5x2 + 5x3 + 4x4")
    c = Constraint("2x1 + 4x2 + 2x3 + 3 x4 <= 450")
    c = c + Constraint("x1 + x2 <= 60")
    c = c + Constraint("x3 + x4 <= 70")
    c = c + Constraint("x1 + x3 <= 50")
    c = c + Constraint("x2 + x4 <= 60")

    linprog = LinearProgramming(z, c)
    linprog.silent_calc()


def test3():
    # Interrogation
    z = ObjectiveFunction("max W = 7x1 + 5x2 + 5x3 + 4x4")
    c = Constraint("2x1 + 4x2 + 2x3 + 3 x4 <= 450")
    c = c + Constraint("x1 + x2 <= 60")
    c = c + Constraint("x3 + x4 <= 70")
    c = c + Constraint("x1 + x3 <= 50")
    c = c + Constraint("x2 + x4 <= 60")

    linprog = LinearProgramming(z, c)
    linprog.calc()


def test4():
    z = ObjectiveFunction("max f = 9x1 + 8 x2 + 3x3 + 4x4 + 6x5 + 7 x6")
    c1 = Constraint("4x1 + 3x2 -x3 + x6 <= 100")
    c2 = Constraint("3x2 + 12 x3 + 17 x4 + 20 x5 <= 1000")
    c3 = Constraint("3x3 + x5 + 12x6 <= 520")
    c4 = Constraint("x1 + x5 >= 60")
    c5 = Constraint("x3 + x5 + 3 x6 <= 300")
    c = c1 + c2 + c3 + c4 + c5
    linprog = LinearProgramming(z, c)
    linprog.calc()


def test5():
    # maximise 5x1 + 6x2
    # x1 + x2 <= 10

    # x1 - x2 >= 3

    # 5x1 + 4x2 <= 35
    z = ObjectiveFunction("max f = 5x1 + 6x2")
    c = Constraint("x1 + x2 <= 10") + Constraint("5x1 + 4x2 <= 35")
    linprog = LinearProgramming(z, c)
    linprog.calc()