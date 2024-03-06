import numpy as np
from fractions import Fraction
from enum import Enum


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Dictionary:
    # Simplex dictionary as defined by Vanderbei
    #
    # 'C' is a (m+1)x(n+1) NumPy array that stores all the coefficients
    # of the dictionary.
    #
    # 'dtype' is the type of the entries of the dictionary. It is
    # supposed to be one of the native (full precision) Python types
    # 'int' or 'Fraction' or any Numpy type such as 'np.float64'.
    #
    # dtype 'int' is used for integer pivoting. Here an additional
    # variables 'lastpivot' is used. 'lastpivot' is the negative pivot
    # coefficient of the previous pivot operation. Dividing all
    # entries of the integer dictionary by 'lastpivot' results in the
    # normal dictionary.
    #
    # Variables are indexed from 0 to n+m. Variable 0 is the objective
    # z. Variables 1 to n are the original variables. Variables n+1 to
    # n+m are the slack variables. An exception is when creating an
    # auxillary dictionary where variable n+1 is the auxillary
    # variable (named x0) and variables n+2 to n+m+1 are the slack
    # variables (still names x{n+1} to x{n+m}).
    #
    # 'B' and 'N' are arrays that contain the *indices* of the basic and
    # nonbasic variables.
    #
    # 'varnames' is an array of the names of the variables.

    def __init__(self, c, A, b, dtype=Fraction):
        # Initializes the dictionary based on linear program in
        # standard form given by vectors and matrices 'c','A','b'.
        # Dimensions are inferred from 'A'
        #
        # If 'c' is None it generates the auxillary dictionary for the
        # use in the standard two-phase simplex algorithm
        #
        # Every entry of the input is individually converted to the
        # given dtype.
        m, n = A.shape
        self.dtype = dtype
        if dtype == int:
            self.lastpivot = 1
        if dtype in [int, Fraction]:
            dtype = object
            if c is not None:
                c = np.array(c, object)
            A = np.array(A, object)
            b = np.array(b, object)
        self.C = np.empty([m + 1, n + 1 + (c is None)], dtype=dtype)
        self.C[0, 0] = self.dtype(0)
        if c is None:
            self.C[0, 1:] = self.dtype(0)
            self.C[0, n + 1] = self.dtype(-1)
            self.C[1:, n + 1] = self.dtype(1)
        else:
            for j in range(0, n):
                self.C[0, j + 1] = self.dtype(c[j])
        for i in range(0, m):
            self.C[i + 1, 0] = self.dtype(b[i])
            for j in range(0, n):
                self.C[i + 1, j + 1] = self.dtype(-A[i, j])
        self.N = np.array(range(1, n + 1 + (c is None)))
        self.B = np.array(range(n + 1 + (c is None), n + 1 + (c is None) + m))
        self.varnames = np.empty(n + 1 + (c is None) + m, dtype=object)
        self.varnames[0] = "z"
        for i in range(1, n + 1):
            self.varnames[i] = "x{}".format(i)
        if c is None:
            self.varnames[n + 1] = "x0"
        for i in range(n + 1, n + m + 1):
            self.varnames[i + (c is None)] = "x{}".format(i)

    def __str__(self):
        # String representation of the dictionary in equation form as
        # used in Vanderbei.
        m, n = self.C.shape
        varlen = len(max(self.varnames, key=len))
        coeflen = 0
        for i in range(0, m):
            coeflen = max(coeflen, len(str(self.C[i, 0])))
            for j in range(1, n):
                coeflen = max(coeflen, len(str(abs(self.C[i, j]))))
        tmp = []
        if self.dtype == int and self.lastpivot != 1:
            tmp.append(str(self.lastpivot))
            tmp.append("*")
        tmp.append("{} = ".format(self.varnames[0]).rjust(varlen + 3))
        tmp.append(str(self.C[0, 0]).rjust(coeflen))
        for j in range(0, n - 1):
            tmp.append(" + " if self.C[0, j + 1] > 0 else " - ")
            tmp.append(str(abs(self.C[0, j + 1])).rjust(coeflen))
            tmp.append("*")
            tmp.append("{}".format(self.varnames[self.N[j]]).rjust(varlen))
        for i in range(0, m - 1):
            tmp.append("\n")
            if self.dtype == int and self.lastpivot != 1:
                tmp.append(str(self.lastpivot))
                tmp.append("*")
            tmp.append("{} = ".format(self.varnames[self.B[i]]).rjust(varlen + 3))
            tmp.append(str(self.C[i + 1, 0]).rjust(coeflen))
            for j in range(0, n - 1):
                tmp.append(" + " if self.C[i + 1, j + 1] > 0 else " - ")
                tmp.append(str(abs(self.C[i + 1, j + 1])).rjust(coeflen))
                tmp.append("*")
                tmp.append("{}".format(self.varnames[self.N[j]]).rjust(varlen))
        return "".join(tmp)

    def basic_solution(self):
        # Extracts the basic solution defined by a dictionary D
        m, n = self.C.shape
        if self.dtype == int:
            x_dtype = Fraction
        else:
            x_dtype = self.dtype
        x = np.empty(n - 1, x_dtype)
        x[:] = x_dtype(0)
        for i in range(0, m - 1):
            if self.B[i] < n:
                if self.dtype == int:
                    x[self.B[i] - 1] = Fraction(self.C[i + 1, 0], self.lastpivot)
                else:
                    x[self.B[i] - 1] = self.C[i + 1, 0]
        return x

    def value(self):
        # Extracts the value of the basic solution defined by a dictionary D
        if self.dtype == int:
            return Fraction(self.C[0, 0], self.lastpivot)
        else:
            return self.C[0, 0]

    def pivot(self, k, l):
        # Pivot Dictionary with N[k] entering and B[l] leaving
        # Performs integer pivoting if self.dtype==int
        # k = entering variable (column index)
        # l = leaving variable (row index)

        # save pivot coefficient
        pivot_coefficient = self.C[l + 1, k + 1]

        # Update elements in matrix C
        self.C[l + 1] = -1 / pivot_coefficient * self.C[l + 1]
        for index, row in enumerate(self.C):
            if index != l + 1:
                element_from_pivot_column = self.C[index, k + 1]
                pivot_row = self.C[l + 1]
                row_scale = element_from_pivot_column * pivot_row
                self.C[index] = self.C[index] + row_scale
                self.C[index, k + 1] = -row_scale[k + 1] / pivot_coefficient

        # Update pivot coefficient and pivot row
        self.C[l + 1, k + 1] = 1 / pivot_coefficient

        # Update N and B
        self.N[k], self.B[l] = self.B[l], self.N[k]


class LPResult(Enum):
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3


def bland(D, eps):
    # Assumes a feasible dictionary D and finds entering and leaving
    # variables according to Bland's rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable

    k = l = None

    # find entering variable
    possible_entering_vars = [D.N[i] for i in np.where(D.C[0, 1:] > eps)[0]]
    try:
        entering_var = min(possible_entering_vars)
        k = np.where(D.N == entering_var)[0][0]
    except ValueError:
        return k, l

    # find leaving variable
    # Find the possible leaving indices
    possible_leaving_indices = [i for i in np.where(D.C[1:, k + 1] < -eps)[0]]
    if len(possible_leaving_indices) == 0:
        return k, l

    ratios = [np.divide(D.C[i + 1, 0], D.C[i + 1, k + 1]) for i in possible_leaving_indices]
    # Find the best ratio
    best_ratio = max(ratios)
    best_indices = np.where(np.equal(ratios, best_ratio))[0]

    # Bland's rule: Choose the smallest index
    l = possible_leaving_indices[min(best_indices)]

    return k, l


def largest_coefficient(D,eps):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Coefficient rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
    
    k=l=None

    # Find entering variable, choose the largest, and if there are more than one choose the one with the smallest index
    if D.C[0,1:].max() < eps:
        return k, l
    
    k = np.argmax(D.C[0,1:])

    # find leaving variable
    # Find the possible leaving indices
    possible_leaving_indices = [i for i in np.where(D.C[1:, k + 1] < -eps)[0]]
    if len(possible_leaving_indices) == 0:
        return k, l

    ratios = [np.divide(D.C[i + 1, 0], D.C[i + 1, k + 1]) for i in possible_leaving_indices]
    # Find the best ratio
    best_ratio_index = np.argmax(ratios)
    l = possible_leaving_indices[best_ratio_index]

    return k,l


def largest_increase(D, eps):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Increase rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable

    k = l = None
    
    # Find positive coefficients
    possible_entering_indices = [i for i in np.where(D.C[0, 1:] > eps)[0]]
    if len(possible_entering_indices) == 0:
        return k, l

    # Calculate the increase for each variable
    largest_increase = None
    for i in possible_entering_indices: # i is column index
        possible_leaving_indices = [j for j in np.where(D.C[1:, i + 1] < -eps)[0]]
        if len(possible_leaving_indices) == 0:
            continue

        # Find the best ratio
        ratios = [np.divide(D.C[j + 1, 0], D.C[j + 1, i + 1]) for j in possible_leaving_indices]
        best_ratio_index = np.argmax(ratios)
        column_increase = - D.C[0, i + 1] * ratios[best_ratio_index]

        if (largest_increase == None) or (column_increase > largest_increase):
            largest_increase = column_increase
            k = i
            l = possible_leaving_indices[best_ratio_index]

    if largest_increase == None:
        return min(possible_entering_indices), None
        
    return k, l



def lp_solve(
    c, A, b, dtype=Fraction, eps=0, pivotrule=lambda D: largest_coefficient(D, eps=0), verbose=False
):
    # Simplex algorithm
    #
    # Input is LP in standard form given by vectors and matrices
    # c,A,b.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0.
    #
    # pivotrule is a rule used for pivoting. Cycling is prevented by
    # switching to Bland's rule as needed.
    #
    # If verbose is True it outputs possible useful information about
    # the execution, e.g. the sequence of pivot operations
    # performed. Nothing is required.
    #
    # If LP is infeasible the return value is LPResult.INFEASIBLE,None
    #
    # If LP is unbounded the return value is LPResult.UNBOUNDED,None
    #
    # If LP has an optimal solution the return value is
    # LPResult.OPTIMAL,D, where D is an optimal dictionary.

    # Phase 1 implemented using the auxillary approach
    # Skip phase 1 if dictionary is feasible
    if b.min() < eps:
        D_aux = Dictionary(None, A, b, dtype)
        if verbose: print(f"Initial auxillary dictionary:\n{D_aux}")
        # We pivot on the "most infeasible" variable
        if verbose: print(f"Pivoting on entering variable {D_aux.varnames[D_aux.N[D_aux.N.size - 1]]} and leaving {D_aux.varnames[D_aux.B[np.argmin(D_aux.C[1:, 0])]]}")
        D_aux.pivot(D_aux.N.size - 1, np.argmin(D_aux.C[1:, 0]))
        if verbose: print(f"Auxilliary dictionary after pivot:\n{D_aux}")
        # We now have a feasible dictionary
        while True:
            k, l = pivotrule(D_aux)
            if k is None:
                if verbose: print(f"Optimal value {D_aux.value()} found, constructing feasible dictionary.")
                break
            if verbose: print(f"Pivoting on entering variable {D_aux.varnames[D_aux.N[k]]} and leaving {D_aux.varnames[D_aux.B[l]]}")
            D_aux.pivot(k, l)
            if verbose: print(f"Auxilliary dictionary after pivot:\n{D_aux}")
        if D_aux.value() < -eps:
            if verbose: print(f"Problem is infeasible with dictionary:\n{D_aux}")
            return LPResult.INFEASIBLE, None
        # Remove auxiliary variable from dictionary
        aux_index = np.where(D_aux.N == D_aux.N.size)[0][0]
        D_aux.N = np.delete(D_aux.N, aux_index)
        D_aux.C = np.delete(D_aux.C, aux_index + 1, axis=1)
        for i in range(len(c)):
            if i + 1 in D_aux.B:
                D_aux.C[0, :] += c[i] * D_aux.C[np.where(D_aux.B == i + 1)[0][0] + 1, :]
            else:
                D_aux.C[0, i] += c[i]
        D = D_aux
        if verbose: print(f"New feasible dictionary:\n{D}\nContinuing to phase 2.") 
    else:
        D = Dictionary(c, A, b, dtype) 
        if verbose: print(f"Initial dictionary is feasible\n{D}\nContinuing to phase 2.")

    
    pivot_counter = 0
    # Phase 2
    while True:
        if pivot_counter == 1000:
            if verbose: print(f"Too many pivots, cycling detected. Switching to Bland's rule.")
            pivotrule = lambda D: bland(D, eps)
        k, l = pivotrule(D)
        if k is None:
            if verbose: print(f"Optimal value {D.value()} found with dictionary:\n{D}")
            return LPResult.OPTIMAL, D
        if l is None:
            if verbose: print(f"Unbounded solution found with dictionary:\n{D}")
            return LPResult.UNBOUNDED, None
        
        if verbose: print(f"x{D.N[k]} is entering and x{D.B[l]} is leaving:")
        D.pivot(k, l)
        pivot_counter += 1
        if verbose: print(f"New Dictionary after pivot:\n{D}")

