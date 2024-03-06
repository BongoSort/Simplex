import numpy as np
from simplex import *
from scipy.optimize import linprog
from timeit import default_timer as timer

# Cycle
def bland_example():
    return (
        np.array([10, -57, -9, -24]),
        np.array([[0.5, -5.5, -2.5, 9], [0.5, -1.5, -0.5, 1], [1, 0, 0, 0]]),
        np.array([0, 0, 1]),
    )

# Optimal
def example1():
    return (
        np.array([5, 4, 3]),
        np.array([[2, 3, 1], [4, 1, 2], [3, 4, 2]]),
        np.array([5, 11, 8]),
    )

# Infeasible, but can be made feasible
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

# Optimal, 5
def exercise2_5():
    return (
        np.array([1, 3]),
        np.array([[-1, -1], [-1, 1], [1, 2]]),
        np.array([-3, -1, 4]),
    )

# Infeasible
def exercise2_6():
    return (
        np.array([1, 3]),
        np.array([[-1, -1], [-1, 1], [1, 2]]),
        np.array([-3, -1, 2]),
    )

# Unbounded
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

def run_example1_fraction():
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

def run_example1_float():
    c, A, b = example1()
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

# Example for auxiliary dictionary in textbook
def run_example2():
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

def run_example2_with_lp_solve():
    # Solve Example 2 using lp_solve
    c, A, b = example2()
    print("lp_solve Example 2:")
    res, D = lp_solve(c, A, b)
    print(res)
    print(D)
    print()

def run_exercise2_5_with_lp_solve():
    # Solve Exercise 2.5 using lp_solve
    c, A, b = exercise2_5()
    print("lp_solve Exercise 2.5:")
    res, D = lp_solve(c, A, b)
    print(res)
    print(D)
    print()

def run_exercise2_6_with_lp_solve():
    # Solve Exercise 2.6 using lp_solve
    c, A, b = exercise2_6()
    print("lp_solve Exercise 2.6:")
    res, D = lp_solve(c, A, b)
    print(res)
    print(D)
    print()

def run_exercise2_7_with_lp_solve():
    # Solve Exercise 2.7 using lp_solve
    c, A, b = exercise2_7()
    print("lp_solve Exercise 2.7:")
    res, D = lp_solve(c, A, b)
    print(res)
    print(D)
    print()

def run_all_examples():
    # Example 1
    run_example1_fraction()

    # Example 1 float
    run_example1_float()

    # Example 2
    run_example2()

    # Solve Example 2 using lp_solve
    run_example2_with_lp_solve()

    # Solve Exercise 2.5 using lp_solve
    run_exercise2_5_with_lp_solve()

    # Solve Exercise 2.6 using lp_solve
    run_exercise2_6_with_lp_solve()

    # Solve Exercise 2.7 using lp_solve
    run_exercise2_7_with_lp_solve()

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


# Comparint the time for lp_solve on feasible dictionaries with different data types
def compare_data_types(pivotrule=lambda D: largest_coefficient(D, eps=0), verbose=False):
    frac_time = float_time = 0.
    for i in range(1, 12):
        size = i*5
        c, A, b = random_lp(size, size)
        start_frac = timer()
        lp_solve(c, A, b, dtype=Fraction, verbose=verbose, pivotrule=pivotrule)
        end_frac = timer()
        start_float = timer()
        lp_solve(c, A, b, dtype=np.float64, verbose=verbose, pivotrule=pivotrule)
        end_float = timer()
        frac_time += end_frac - start_frac
        float_time += end_float - start_float
        print(f"Time for size {size} x {size}: Fraction {round((end_frac - start_frac) * 1000, 1)} ms, Float {round((end_float - start_float) * 1000, 1)} ms")
    print(f"Total time with dtype=Fraction is: %.4f" % (frac_time), "seconds")
    print(f"Total time with dtype=np.float64 is: %.4f" % (float_time), "seconds")

# Comparing the time for different solvers on feasible dictionaries with data type np.float64
def compare_scipy_methods(pivotrule=lambda D: largest_coefficient(D, eps=0)):
    lp_solve_time = simplex_time = highs_time = 0.
    for i in range(1, 15):
        size = i*5
        c, A, b = random_lp(size, size)
        start_lp = timer()
        lp_solve(c, A, b, dtype=np.float64, pivotrule=pivotrule)
        end_lp = timer()
        start_simplex = timer()
        # Linprog uses minimization, so we need to negate the objective function for both methods
        linprog(-c, A_ub=A, b_ub=b, method="simplex")
        end_simplex = timer()
        start_highs = timer()
        linprog(-c, A_ub=A, b_ub=b, method="highs-ds")
        end_highs = timer()
        lp_solve_time += end_lp - start_lp
        simplex_time += end_simplex - start_simplex
        highs_time += end_highs - start_highs
        print(f"Time for size {size} x {size}: lp_solve {round((end_lp - start_lp) * 1000, 1)} ms, Simplex {round((end_simplex - start_simplex) * 1000, 1)} ms, Highs {round((end_highs - start_highs) * 1000, 1)} ms")
    print(f"Total time for lp_solve is: %.4f" % (lp_solve_time), "seconds")
    print(f"Total time for Simplex is: %.4f" % (simplex_time), "seconds")
    print(f"Total time for Highs is: %.4f" % (highs_time), "seconds")

def compare_pivotrules():
    bland_time = largest_coefficient_time = largest_increase_time = 0.
    for i in range(1, 15):
        size = i*5
        c, A, b = random_lp(size, size)
        start_bland = timer()
        lp_solve(c, A, b, dtype=np.float64, pivotrule=lambda D: bland(D, eps=0))
        end_bland = timer()
        start_largest_coefficient = timer()
        lp_solve(c, A, b, dtype=np.float64, pivotrule=lambda D: largest_coefficient(D, eps=0))
        end_largest_coefficient = timer()
        start_largest_increase = timer()
        lp_solve(c, A, b, dtype=np.float64, pivotrule=lambda D: largest_increase(D, eps=0))
        end_largest_increase = timer()
        bland_time += end_bland - start_bland
        largest_coefficient_time += end_largest_coefficient - start_largest_coefficient
        largest_increase_time += end_largest_increase - start_largest_increase
        print(f"Time for size {size} x {size}: Bland {round((end_bland - start_bland) * 1000, 1)} ms, Largest Coefficient {round((end_largest_coefficient - start_largest_coefficient) * 1000, 1)} ms, Largest Increase {round((end_largest_increase - start_largest_increase) * 1000, 1)} ms")
    print(f"Total time for Bland is: %.4f" % (bland_time), "seconds")
    print(f"Total time for Largest Coefficient is: %.4f" % (largest_coefficient_time), "seconds")
    print(f"Total time for Largest Increase is: %.4f" % (largest_increase_time), "seconds")

def run_example_with_both_solvers(example, pivotrule=None, verbose=False):
    c, A, b = example
    linprog_res = linprog(-c, A_ub=A, b_ub=b, method="highs-ds", verbose=verbose)
    print("linprog result:")
    print(linprog_res)
    lp_res, D = lp_solve(c, A, b, pivotrule=pivotrule, dtype=Fraction)
    print("lp_solve result:")
    print(lp_res, "\n", D)
    


if __name__ == "__main__":
    compare_scipy_methods()