import numpy as np
from simplex import *


def bland_example():
    return (
        np.array([10, -57, -9, -24]),
        np.array([[0.5, -5.5, -2.5, 9], [0.5, -1.5, -0.5, 1], [1, 0, 0, 0]]),
        np.array([0, 0, 1]),
    )


def example1():
    return (
        np.array([5, 4, 3]),
        np.array([[2, 3, 1], [4, 1, 2], [3, 4, 2]]),
        np.array([5, 11, 8]),
    )


def example2():
    return (
        np.array([-2, -1]),
        np.array([[-1, 1], [-1, -2], [0, 1]]),
        np.array([-1, -2, 1]),
    )


def integer_pivoting_example():
    return (
        np.array([5, 2]), 
        np.array([[3, 1], [2, 5]]), 
        np.array([7, 5])
    )


def exercise2_5():
    return (
        np.array([1, 3]),
        np.array([[-1, -1], [-1, 1], [1, 2]]),
        np.array([-3, -1, 4]),
    )


def exercise2_6():
    return (
        np.array([1, 3]),
        np.array([[-1, -1], [-1, 1], [1, 2]]),
        np.array([-3, -1, 2]),
    )


def exercise2_7():
    return (
        np.array([1, 3]),
        np.array([[-1, -1], [-1, 1], [-1, 2]]),
        np.array([-3, -1, 2]),
    )


def random_lp(n, m, sigma=10):
    return (
        np.round(sigma * np.random.randn(n)),
        np.round(sigma * np.random.randn(m, n)),
        np.round(sigma * np.abs(np.random.randn(m))),
    )

def run_examples():
    # Example 1
    c, A, b = example1()
    D = Dictionary(c, A, b)
    print("Example 1 with Fraction")
    print("Initial dictionary:")
    print(D)
    print("x1 is entering and x4 leaving:")
    D.pivot(0, 0)
    print(D)
    print("x3 is entering and x6 leaving:")
    D.pivot(2, 2)
    print(D)
    print()

    # Example 1 float
    D = Dictionary(c, A, b, np.float64)
    print("Example 1 with np.float64")
    print("Initial dictionary:")
    print(D)
    print("x1 is entering and x4 leaving:")
    D.pivot(0, 0)
    print(D)
    print("x3 is entering and x6 leaving:")
    D.pivot(2, 2)
    print(D)
    print()

    # Example 2
    c, A, b = example2()
    print("Example 2")
    print("Auxillary dictionary")
    D = Dictionary(None, A, b)
    print(D)
    print("x0 is entering and x4 leaving:")
    D.pivot(2, 1)
    print(D)
    print("x2 is entering and x3 leaving:")
    D.pivot(1, 0)
    print(D)
    print("x1 is entering and x0 leaving:")
    D.pivot(0, 1)
    print(D)
    print()

    # Solve Example 2 using lp_solve
    c, A, b = example2()
    print("lp_solve Example 2:")
    res, D = lp_solve(c, A, b)
    print(res)
    print(D)
    print()

    # Solve Exercise 2.5 using lp_solve
    c, A, b = exercise2_5()
    print("lp_solve Exercise 2.5:")
    res, D = lp_solve(c, A, b)
    print(res)
    print(D)
    print()

    # Solve Exercise 2.6 using lp_solve
    c, A, b = exercise2_6()
    print("lp_solve Exercise 2.6:")
    res, D = lp_solve(c, A, b)
    print(res)
    print(D)
    print()

    # Solve Exercise 2.7 using lp_solve
    c, A, b = exercise2_7()
    print("lp_solve Exercise 2.7:")
    res, D = lp_solve(c, A, b)
    print(res)
    print(D)
    print()

    # Integer pivoting
    c, A, b = example1()
    D = Dictionary(c, A, b, int)
    print("Example 1 with int")
    print("Initial dictionary:")
    print(D)
    print("x1 is entering and x4 leaving:")
    D.pivot(0, 0)
    print(D)
    print("x3 is entering and x6 leaving:")
    D.pivot(2, 2)
    print(D)
    print()

    c, A, b = integer_pivoting_example()
    D = Dictionary(c, A, b, int)
    print("Integer pivoting example from lecture")
    print("Initial dictionary:")
    print(D)
    print("x1 is entering and x3 leaving:")
    D.pivot(0, 0)
    print(D)
    print("x2 is entering and x4 leaving:")
    D.pivot(1, 1)
    print(D)

    return

def run_bland():
    c, A, b = bland_example()
    D = Dictionary(c, A, b)
    print(D)
    D.pivot(0, 0)
    print(D)
    D.pivot(1, 1)
    print(D)
    D.pivot(2, 0)
    print(D)
    D.pivot(3, 1)
    print(D)
    D.pivot(0, 0)
    print(D)
    D.pivot(2, 1)
    print(D)
    D.pivot(0, 2)
    print(D)
    bland(D, 0.000001)


def run_timed_example1():

    # Solve Example 1 using lp_solve
    c, A, b = example1()
    print("lp_solve Example 1:")
    start = timer()
    res, D = lp_solve(c, A, b)
    end = timer()
    print(res)
    print(D)
    print()

    time_ms1 = 1000 * (end - start)
    print(f"time for lp_solve %.4f" % (time_ms1), "ms")

    start2 = timer()
    res = linprog(c, A, b, method="simplex")
    end2 = timer()
    time_ms2 = 1000 * (end2 - start2)
    print(f"time for simplex %.4f" % (time_ms2), "ms")

    start3 = timer()
    res = linprog(c, A, b, method="highs-ds")
    end3 = timer()
    time_ms3 = 1000 * (end3 - start3)
    print(f"time for highs-ds %.4f" % (time_ms3), "ms")


def run_random_lp(n, m, sigma):
    c, A, b = random_lp(n, m, sigma)
    print("lp_solve Random LP")
    start = timer()
    res, D = lp_solve(c, A, b)
    end = timer()
    print(f"Hvad er vores tid", end - start)


def time_all_using_solver(solver):

    c, A, b = example1()
    start = timer()
    solver(c, A, b)
    end = timer()

    c, A, b = bland_example()
    start3 = timer()
    solver(c, A, b)
    end3 = timer()

    c, A, b = integer_pivoting_example()
    start4 = timer()
    solver(c, A, b)
    end4 = timer()

    c, A, b = exercise2_5()
    start5 = timer()
    solver(c, A, b)
    end5 = timer()

    c, A, b = exercise2_6()
    start6 = timer()
    solver(c, A, b)
    end6 = timer()

    c, A, b = exercise2_7()
    start7 = timer()
    solver(c, A, b)
    end7 = timer()

    time_ms1 = 1000 * (
        (end - start)
        + (end3 - start3)
        + (end4 - start4)
        + (end5 - start5)
        + (end6 - start6)
        + (end7 - start7)
    )
    print(f"timing for all exercises %.4f" % (time_ms1), "ms")


def time_all_using_solver_with_method(solver, meth):

    c, A, b = example1()
    start = timer()
    solver(c, A, b, method=meth)
    end = timer()

    c, A, b = bland_example()
    start3 = timer()
    solver(c, A, b, method=meth)
    end3 = timer()

    c, A, b = integer_pivoting_example()
    start4 = timer()
    solver(c, A, b, method=meth)
    end4 = timer()

    c, A, b = exercise2_5()
    start5 = timer()
    solver(c, A, b, method=meth)
    end5 = timer()

    c, A, b = exercise2_6()
    start6 = timer()
    solver(c, A, b, method=meth)
    end6 = timer()

    c, A, b = exercise2_7()
    start7 = timer()
    solver(c, A, b, method=meth)
    end7 = timer()

    time_ms1 = 1000 * (
        (end - start)
        + (end3 - start3)
        + (end4 - start4)
        + (end5 - start5)
        + (end6 - start6)
        + (end7 - start7)
    )
    print(f"timing for all exercises %.4f" % (time_ms1), "ms")


def main():
    # time_all_using_solver(lp_solve)
    # time_all_using_solver_with_method(linprog, "simplex")
    # time_all_using_solver_with_method(linprog, "highs-ds")
    c, A, b = example1()


if __name__ == "__main__":
    main()