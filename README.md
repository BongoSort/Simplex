# Simplex

Gruppeaflevering 1 Optimering

## Programming language

The programming project is concerned with implementing and experimenting with the simplex method in Python 3, utilizing the NumPy library.

A template solution (simplex.py) is supplied which you are expected to use and follow.

## Sub-projects

The programming project consists of 5 sub-projects which should be solved in the given order. During grading each sub-project is awarded 0, 1, or 2 points.
Only the solution handed in before the deadline receives points.

### Sub-project 1 (one-phase simplex method)

Implement the one-phase simplex method with dictionaries having entries of data type Fraction and np.float64, and using Bland's rule for pivoting.

### Sub-project 2 (experimentation)

Design and conduct a computational experiment evaluating the performance of your implementation for both data types Fraction and np.float64. You may for instance take inspiration from Chapter 4 of Vanderbei. Compare also your implementation using floats to the Simplex method accessible with the function scipy.optimize.linprog in the package scipy.optimize of the SciPy library. Recall that you need to explicitly specify that the simplex algorithm should be used (method='highs-ds' or method='simplex'). The method='simplex' is deprecated since version 1.9.0 of SciPy, but can still be fun to use, since you will be able to make a better and faster solver than that. The method='highs-ds' uses an optimized Dual Simplex implementation contained in the HiGHS solver.

### Sub-project 3 (two-phase simplex method)

Extend your implementation to a two-phase simplex algorithm. You may chose either the auxiliary dictionary approach or the dual based approach.

### Sub-project 4 (pivoting rules and further experimentation)

Implement the largest-coefficient and the largest-increase pivoting rules. Recall that the largest-coefficient rule picks an entering variable with largest coefficient in the equation for the objective function, and the largest-increase pick an entering variable that will give the largest increase in value of the current basic feasible solution. Repeat the computational experiment using these and using floats, and compare to you experiments in sub-project 2.

### Sub-project 5 (integer pivoting)

Implement integer pivoting as described at Lecture 5 and compare experimentally against using data type Fraction. The implementation should also work for the two-phase simplex method you chose.

### What to hand in

Hand in the following:

* All your source code.
* A document describing and documenting computational experiments performed.

### A couple of hints

NumPy is built for operating on vectors and matrices in a "vectorized" way. Using this will reduce the number of required keystrokes (as well as the running time of your code!).
Make sure that your experiments are repeatable and usable for comparisons. If using the "random number" generator of NumPy this is ensured by setting the random seed by calling np.random.seed.
It may be useful to use the Python unit testing framework during implementation. See test.py for an example.