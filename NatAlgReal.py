#################################################################################
#### PLEASE READ ALL COMMENTS BELOW AND MAKE SURE YOU FOLLOW MY INSTRUCTIONS ####
#################################################################################

# This is the skeleton program 'NatAlgReal.py' around which you should build your implementation.
# Please read through this program and follow the instructions given.

# There are no input or output files, with the results printed to the standard output.

# As regards the two values to be entered below
# - make sure that the first two values appear within double quotes
# - make sure that 'username' is lower-case
# - make sure that no comments are inserted after you have entered the values.

# Ensure that your implementation works for *arbitrary* hard-coded functions of arbitrary
# dimension and arbitrary min- and max-ranges!

##############################
#### ENTER YOUR USER-NAME ####
##############################

username = "jvgw34"

###############################################################
#### ENTER THE CODE FOR THE ALGORITHM YOU ARE IMPLEMENTING ####
###############################################################

alg_code = "FF"

################################################################
#### DO NOT TOUCH ANYTHING BELOW UNTIL I TELL YOU TO DO SO! ####
####      THIS INCLUDES IMPORTING ADDITIONAL MODULES!       ####
################################################################

import time
import random
import math
import sys
import os
import datetime

def compute_f(point):
    f = -1 * math.sin(point[0])*math.sqrt(point[0]) * math.sin(point[1])*math.sqrt(point[1]) * \
        math.sin(point[2])*math.sqrt(point[2]) * math.sin(point[3])*math.sqrt(point[3])
    return f

    

n = 4

min_range = [0, 0, 0, 0]
max_range = [10, 10, 10, 10]


start_time = time.time()

#########################################################################################
#### YOU SHOULDN'T HAVE TOUCHED *ANYTHING* UP UNTIL NOW APART FROM SUPPLYING VALUES  ####
####                 FOR 'username' and 'alg_code' AS REQUESTED ABOVE.               ####
####                        NOW READ THE FOLLOWING CAREFULLY!                        ####
#########################################################################################

# The function 'f' is 'n'-dimensional and you are attempting to MINIMIZE it.
# To compute the value of 'f' at some point 'point', where 'point' is a list of 'n' integers or floats,
# call the function 'compute_f(point)'.
# The ranges for the values of the components of 'point' are given above. The lists 'min_range' and
# 'max_range' above hold the minimum and maximum values for each component and you should use these
# list variables in your code.

# On termination your algorithm should be such that:
#   - the reserved variable 'min_f' holds the minimum value that you have computed for the
#     function 'f' 
#   - the reserved variable 'minimum' is a list of 'n' entries (integer or float) holding the point at which
#     your value of 'min_f' is attained.

# Note that the variables 'username', 'alg_code', 'f', 'point', 'min_f', 'n', 'min_range', 'max_range' and
# 'minimum' are all reserved.

# FOR THE RESERVED VARIABLES BELOW, YOU MUST ENSURE THAT ON TERMINATION THE TYPE
# OF THE RESPECTIVE VARIABLE IS AS SHOWN.

#  - 'min_f'                int or float
#  - 'minimum'              list of int or float

# You should ensure that your code works on any function hard-coded as above, using the
# same reserved variables and possibly of a dimension different to that given above. I will
# run your code with a different such function/dimension to check that this is the case.

# The various algorithms all have additional parameters (see the lectures). These parameters
# are detailed below and are referred to using the following reserved variables.
#
# AB (Artificial Bee Colony)
#   - 'n' = dimension of the optimization problem       int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of employed bees / food sources      int
#   - 'M' = number of onlooker bees                     int
#   - 'lambbda' = limit threshold                       float or int
#
# FF (Firefly)
#   - 'n' = dimension of the optimization problem       int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of fireflies                         int
#   - 'lambbda' = light absorption coefficient          float or int
#   - 'alpha' = scaling parameter                       float or int
#
# CS (Cuckoo Search)
#   - 'n' = dimension of optimization problem           int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of nests                             int
#   - 'p' = fraction of local flights to undertake      float or int
#   - 'q' = fraction of nests to abandon                float or int
#   - 'alpha' = scaling factor for Levy flights         float or int
#   - 'beta' = parameter for Mantegna's algorithm       float or int
#
# WO (Whale Optimization)
#   - 'n' = dimension of optimization problem           int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of whales                            int
#   - 'b' = spiral constant                             float or int
#
# BA (Bat)
#   - 'n' = dimension of optimization problem           int
#   - 'num_cyc' = number of cycles to iterate           int
#   - 'N' = number of bats                              int
#   - 'sigma' = scaling factor                          float or int
#   - 'f_min' = minimum frequency                       float or int
#   - 'f_max' = maximum frequency                       float or int

# These are reserved variables and need to be treated as such, i.e., use these names for these
# parameters and don't re-use the names. Don't forget to ensure that on termination all the above
# variables have the stated type. In particular, if you use specific numpy types then you'll need
# to ensure that they are changed prior to termination (this is checked).

# INITIALIZE THE ACTUAL PARAMETERS YOU USE FOR YOUR ALGORITHM BELOW. ENSURE THAT YOU INITIALIZE
# *ALL* OF THE PARAMETERS REQUIRED APPROPRIATELY (SEE ABOVE) FOR YOUR CHOSEN ALGORITHM.

# In summary, before you input the bulk of your code, ensure that you:
# - import any (legal) modules you wish to use in the space provided below 
# - initialize your parameters in the space provided below
# - ensure that reserved variables have the correct type on termination.

###########################################
#### NOW YOU CAN ENTER YOUR CODE BELOW ####
###########################################
####################################################
#### FIRST IMPORT ANY MODULES IMMEDIATELY BELOW ####
####################################################
import numpy as np
from copy import deepcopy as deepcp
##########################################################
#### NOW INITIALIZE YOUR PARAMETERS IMMEDIATELY BELOW ####
##########################################################

'''
IMPORTANT:
All of the comments I have made are optional but recommended to read.

Foreword:
Defining a lot of this rigorously is quite hard and (i think) out of the scope of this assignment. While this is very interesting, I don't have the time to go too far into depth.

There are some key definitions I have loosely defined and would like to explain to better understand my notes:

Exploration:

    Coverage and searching of the whole search space based on the min/max ranges provided.

Exploitation:

    Localised searching around a convergent point to examine whether the minima goes deeper.


Optimal convergence:

    I am going to loosely define optimal convergence as: Lambda not being so strong as to force every point to converge into one.
    And, Lambda not being so weak, so as the movement appears random. But, Lambda being in the range, where grouped points or "niches" can explore local minima without being
    too attracted to other groups of points such that they dont all converge to a local minima.

    TLDR: the attraction not being so strong as to force all points to converge to one, but not so weak as to only rely on random movement for exploration

        ------------------------------------------------------------------------------------------------------------------------
    VERY IMPORTANT: I am testing this by visualising the points and determining, by eye, whether it is optimal.
    This leads to a lot of room for error and misinterpretation, however the results I have achieved from this have improved the algorithm
        ------------------------------------------------------------------------------------------------------------------------
    P.S. For this algorithm implementation, this is quite well shown on the schwefel function.
    
'''


N = 200
'''
More fireflies statistically reduces the chances of getting stuck in local minima.
If there are too few fireflies, There will not be enough to explore the search space and group points for minima exploration.

'''
alpha = 0.04
'''
Alpha is also highly range sensitive. As lower values for small ranges are required, while higher values for bigger ranges are required.
As soon as the range is higher, the random movement will no longer be enough to encourage exploration.
A low alpha leads to much slower convergence, More random movement will need to occur to reach another fireflies attraction neighbourhood

Alpha should also be made adaptive, as stated in papers regarding the firefly algorithm.
The current issue is the papers I have read that provide an adaptive method for alpha do not provide any proofs or reasoning for their choice
For example, In [Wang, Wang, Zhou, et al.], he defines alpha as a(t+1) = (1/9000)^(1/t). This converges to zero far before any meaningful random exploration can be done.
This can be seen by simply plugging in a couple values into this function, It reaches negligable values before even 100 iterations.
The paper they quote for this does not even mention the alpha formula they are quoting.
'''
lambbda = 50
rho = 2
'''

Lambda is heavily range sensitive.
With longer ranges, a lower lambda is required, otherwise the function only moves at random determined by alpha, the chance of the firefly coming into
the attraction neighbourhood of another firefly is low in this case.

To solve this issue. A scaling of lambda is worth doing.

I came to this conclusion by comparing convergence at a range of [0, 10] and [-500, 500]
The lambda at which they began converging optimally was approximately a factor of 10^4 apart (values of 0.5 and 5x10 ^ -5)

This relationship is not linear. From some experimentation and analysis, 
I have found that a good scaling methodology for lambda is: scaled lambda = lambda/delta ^ ⍴ where rho is a scaling constant and delta is the difference
between max_range and min_range in dimension n.
For now, this is quite a loose approximation, but it seems to hold for any range I throw, tested from delta = [0.1, 100000], dim = [2,4]. 
Providing a good balance between exploration and exploitation.
I reached the values of ƛ = 50 and ⍴ = 2 by solving 0.5 = ƛ / (10 ^ ⍴) and 5x10 ^ - 5 = ƛ / (1000 ^ ⍴). Solving for ƛ and ⍴, gave me 50 and 2 respectively.
The only issue with this is performance at higher dimensions, I currently have no idea how to account for higher dimensionality with this method.
'''
num_cyc = 350 # 350 Determined by 20 run maximum. Lowest value I have achieved while all 20 runs reaching the global minima of -62
kN = 3
'''

The comparison neighbourhood logic mentioned in the paper appears to hold true.
Testing at kN: [1,3,11] (two extremes and the suggested value)
1: Little convergence/More exploration
3: Satisfactory level of convergence and exploration
11: Far faster convergence, and not enough exploration.

Tested with the definitions of convergence and exploration defined earlier.
'''
Bzero = 1
Bmin = 0
'''
Bmin allows for some attractiveness regardless of range. In my experimentation, this leads to early convergence and little to no exploration.
Again, not exactly quantitative, however rigorously defining early convergence is out of the scope of this assignment, and for this I am using the vague definition
mentioned before.
For this reason, I have set Bmin to 0, and kept the standard value of Bzero = 1 for this algorithm.
'''
lenght_scale = [max_range[x] - min_range[x] for x in range(len(max_range))]
'''
Above is the length scale definition we use for alpha scaling. 
'''
###########################################
#### NOW INCLUDE THE REST OF YOUR CODE ####
###########################################

#Alternative test functions below
def compute_f2(xx):
    """
    SCHWEFEL FUNCTION
    INPUT:
    xx = [x1, x2, ..., xd]
    """
    d = len(xx)
    sum_term = np.sum(xx * np.sin(np.sqrt(np.abs(xx))))
    y = 418.9829 * d - sum_term
    return y

def compute_f3(xx):
    """
    Griewank Function.
    
    Parameters:
    xx : array-like
        Input vector (x1, x2, ..., xd).
    
    Returns:
    float
        Value of the Griewank function.
    """
    xx = np.array(xx)
    ii = np.arange(1, len(xx) + 1)
    sum_term = np.sum(xx**2 / 4000)
    prod_term = np.prod(np.cos(xx / np.sqrt(ii)))
    
    y = sum_term - prod_term + 1
    return y

# Range redefinitions for each function
#max_range = [500] * n #Schwefel range
#min_range = [-500] * n
#min_range = [-600] * n #Griewank range
#max_range = [600] * n
    

class memeticAlpha:
    '''
    Flawed implementation of the memetic alpha strategy metioned in [Wang, Wang, Zhou, et al.]
    As mentioned before, this strategy converges far too quickly to zero to be practical to the algorithm
    Reaches negligable values of alpha in under 100 iterations.
    An exponential strategy based on iterations seems to be a poor choice here.
    
    An idea for an alternative strategy that I have thought of is to have a relationship between alpha and distance.
    As distance decreses, alpha decreases. That way fireflies that have converged together can better exploit their minima.
    A problem with this is that, at the start of the algorithm, as soon as a firefly reaches the influence of another firefly, alpha will drop to near zero. Could be tuned by some alternative parameter...
    This could lead to early convergence, or it might work correctly, or it could completely fail. Who knows, I didn't have time to experiment with this.

    Further research into alpha adaptation is definately needed.
    '''
    def __init__(self, startingValue):
        self.curAlpha = startingValue
        self.decayConstant = (1/9000)
        self.t = 1


    def incrementAlpha(self):
        self.curAlpha = (self.decayConstant ** (1 / self.t)) * self.curAlpha
        self.t += 1

    def getAlpha(self):
        return self.curAlpha
    

def attractiveness(radius, lmdba):
    return (Bmin + (Bzero - Bmin)) * math.exp(-lmdba * (radius**2))

def euclidDist(point1, point2):
    return np.linalg.norm(np.subtract(point1, point2)) # Slightly faster euclidian norm calculation.

def FireflyAlgorithm(N, n, num_cyc, lambbda, alpha, historyFlag):
    '''
    Paper for neighbourhood attraction
    https://www.sciencedirect.com/science/article/pii/S0020025516320497

    Memetic FA:
    https://www.sciencedirect.com/science/article/pii/S0020025516320497#eq0007

    Went with a clipping border strategy:
    Paper for clipping strategy:
    https://www.scs-europe.net/dlib/2018/ecms2018acceptedpapers/0170_is_ecms2018_0868.pdf
    Reason for going for clipping strategy:
        Keeps information from previous iterations.

    '''
    start = time.time()
    X = []
    for k in range(N):
        # Random population generation
        tmp = []
        for dim in range(n):
            lmin = min_range[dim]
            lmax = max_range[dim]
            tmp.append(np.random.uniform(lmin,lmax))
        X.append(np.array(tmp))    
    X = np.array([x for x in X])
    X_h = [] # X_h is the history of the random population at each iteration. Used for visualisation.
    fitness = np.array([compute_f(c) for c in X])
    min_flocal = float('inf')
    minimumlocal = []
    X_h.append(deepcp(X))
    print("Alpha:",alpha, " Lambda:", lambbda, " N:", N)


    for t in range(num_cyc):
        
        if time.time() - start >= 9.9:
            break
        if (t % 100) == 0:
            print(t, min_flocal)

        # Sort by fitness value
        sortedIndices = np.argsort(fitness)
        X = X[sortedIndices]
        sortedFitness = fitness[sortedIndices]

        if (t % 100) == 0:
            '''
            Re-randomise bottom 80% every 100 iterations:
            This frees up the "worst" fireflies from exploring non optimal minima and allows them to further explore while the best solutions can exploit
            their position and minimas.

            From Four 10 run trials.
            2 without randomisation, 2 with.
            The 2 trials without randomisation each hit a non optimal local minima of 118 at least once
            The 2 trials with randomisation did not get stuck in a local minima.

            This is an improvement brought back from NatAlgDiscrete.

            Comment out from lines 401 to 431 to disable this, (useful for visualising minima convergence.)
            '''
            dmod = int(X.shape[0] * 0.8)
            for i in range(X.shape[0] - dmod, X.shape[0]):
                tmp = []
                for dim in range(n):
                    lmin = min_range[dim]
                    lmax = max_range[dim]
                    tmp.append(np.random.uniform(lmin,lmax))

                tmp = np.array(tmp)
                newfitness = compute_f(tmp)
                X[i] = tmp
                sortedFitness[i] = newfitness
            
            sortedIndices = np.argsort(fitness)
            X = X[sortedIndices]
            sortedFitness = fitness[sortedIndices]

        for idx, fireflyi in enumerate(X):
            fitnessi = sortedFitness[idx]
            for jx in range((idx - kN) + 1, idx + kN + 1):
                '''
                Iterating the comparison neighbourhood here, to get it we sort our fireflies by fitness and get the
                '''
                if jx == idx:
                    continue
                j = (jx + N) % N
                fitnessj = sortedFitness[j]
                fireflyj = X[j]
                r = euclidDist(fireflyi, fireflyj)
                randweights = np.random.uniform(-0.5, 0.5, n)
                
                if fitnessi < fitnessj:
                    for k, xjk in np.ndenumerate(fireflyj):
                        k = k[0]

                        llambbda = lambbda / (lenght_scale[k] ** rho)
                        brightness = attractiveness(r, llambbda)
                        
                        fireflyj[k] = max(min(fireflyj[k] + (brightness * (fireflyi[k] - xjk)) + (alpha * randweights[k] * lenght_scale[k]),max_range[k]),min_range[k])
                
                elif fitnessi > fitnessj:
                    
                    for k, xik in np.ndenumerate(fireflyi):
                        k = k[0]

                        llambbda = lambbda / (lenght_scale[k] ** rho)
                        brightness = attractiveness(r, llambbda)

                        fireflyi[k] = max(min(fireflyi[k] + (brightness * (fireflyj[k] - xik)) + (alpha * randweights[k] * lenght_scale[k]),max_range[k]),min_range[k])
                    
                else:
                    for k, i in np.ndenumerate(fireflyi):
                        k = k[0]
                        rndshift = alpha * randweights[k] * lenght_scale[k]
                        fireflyi[k] = max(min(fireflyi[k] + rndshift, max_range[k]),min_range[k])
                        fireflyj[k] = max(min(fireflyj[k] + rndshift, max_range[k]),min_range[k])
        
        fitness = np.array([compute_f(c) for c in X])
        cminf = min(fitness)
        if cminf < min_flocal:
            min_flocal = cminf
            minimumlocal = X[np.where(fitness == cminf)[0][0]]
        if historyFlag:
            X_h.append(deepcp(X))
        #mem.incrementAlpha()
    print(t)
    return minimumlocal, min_flocal, X_h

'''
#Code for benchmarking and visualisation.
it = 0 # Variable defining number of runs to calculate averages for.
sm = 0
rmax = -float('inf')
res = []
for i in range(it):
    minimum, min_f, firefly_history = FireflyAlgorithm(N, n, num_cyc, lambbda, alpha, True)
    minimum = list(minimum)
    min_f = float(min_f)
    print(min_f)
    res.append(min_f)
    rmax = max(min_f, rmax)
    sm += min_f


if it > 0:
    print(f"{it} run median: ", np.median(np.array(res)), f" {it} run average: ", sm/it, f" {it} run maximum: ", rmax)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get the minimum and maximum values for the x, y, and z axes
    x_min = min([f[0] for fireflies in firefly_history for f in fireflies])
    x_max = max([f[0] for fireflies in firefly_history for f in fireflies])
    y_min = min([f[1] for fireflies in firefly_history for f in fireflies])
    y_max = max([f[1] for fireflies in firefly_history for f in fireflies])
    z_min = min([compute_f(f) for fireflies in firefly_history for f in fireflies])
    z_max = max([compute_f(f) for fireflies in firefly_history for f in fireflies])

    for t, fireflies in enumerate(firefly_history):
        ax.clear()
        ax.scatter([f[0] for f in fireflies], [f[1] for f in fireflies], [compute_f(f) for f in fireflies], c='r', s=50, alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title(f"Iteration {t}")
        plt.pause(0.05)

    plt.show()
'''

minimum, min_f, firefly_history = FireflyAlgorithm(N, n, num_cyc, lambbda, alpha, False)
minimum = list(minimum)
min_f = float(min_f)



#########################################################
#### YOU SHOULD HAVE NOW FINISHED ENTERING YOUR CODE ####
####     DO NOT TOUCH ANYTHING BELOW THIS COMMENT    ####
#########################################################

# At this point in the execution, you should have computed your minimum value for the function 'f' in the
# variable 'min_f' and the variable 'minimum' should hold a list containing the values of the point 'point'
# for which function 'f(point)' attains your minimum.

now_time = time.time()
elapsed_time = round(now_time - start_time, 1)

error = []

try:
    n
    try:
        y = n
    except:
        error.append("*** error: 'n' has not been initialized")
        n = -1
except:
    error.append("*** error: the variable 'n' does not exist\n")
    n = -1
try:
    num_cyc
    try:
        y = num_cyc
    except:
        error.append("*** error: 'num_cyc' has not been initialized")
        num_cyc = -1
except:
    error.append("*** error: the variable 'num_cyc' does not exist")
    num_cyc = -1

if alg_code == "AB":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        M
        try:
           y = M
        except:
            error.append("*** error: 'M' has not been initialized")
            M = -1
    except:
        error.append("*** error: the variable 'M' does not exist")
        M = -1
    try:
        lambbda
        try:
           y = lambbda
        except:
            error.append("*** error: 'lambbda' has not been initialized")
            lambbda = -1
    except:
        error.append("*** error: the variable 'lambbda' does not exist")
        lambbda = -1
if alg_code == "FF":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        alpha
        try:
           y = alpha
        except:
            error.append("*** error: 'alpha' has not been initialized")
            alpha = -1
    except:
        error.append("*** error: the variable 'alpha' does not exist")
        alpha = -1
    try:
        lambbda
        try:
           y = lambbda
        except:
            error.append("*** error: 'lambbda' has not been initialized")
            lambbda = -1
    except:
        error.append("*** error: the variable 'lambbda' does not exist")
        lambbda = -1
if alg_code == "CS":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        p
        try:
           y = p
        except:
            error.append("*** error: 'p' has not been initialized")
            p = -1
    except:
        error.append("*** error: the variable 'p' does not exist")
        p = -1
    try:
        q
        try:
           y = q
        except:
            error.append("*** error: 'q' has not been initialized")
            q = -1
    except:
        error.append("*** error: the variable 'q' does not exist")
        q = -1
    try:
        alpha
        try:
           y = alpha
        except:
            error.append("*** error: 'alpha' has not been initialized")
            alpha = -1
    except:
        error.append("*** error: the variable 'alpha' does not exist")
        alpha = -1
    try:
        beta
        try:
           y = beta
        except:
            error.append("*** error: 'beta' has not been initialized")
            beta = -1
    except:
        error.append("*** error: the variable 'beta' does not exist")
        beta = -1
if alg_code == "WO":
    try:
        N
        try:
           y = N
        except:
            error.append("*** error: 'N' has not been initialized")
            N = -1
    except:
        error.append("*** error: the variable 'N' does not exist")
        N = -1
    try:
        b
        try:
           y = b
        except:
            error.append("*** error: 'b' has not been initialized")
            b = -1
    except:
        error.append("*** error: the variable 'b' does not exist")
        b = -1
if alg_code == "BA":
    try:
        sigma
        try:
           y = sigma
        except:
            error.append("*** error: 'sigma' has not been initialized")
            sigma = -1
    except:
        error.append("*** error: the variable 'sigma' does not exist")
        sigma = -1
    try:
        f_max
        try:
           y = f_max
        except:
            error.append("*** error: the variable 'f_max' has not been initialized")
            f_max = -1
    except:
        error.append("*** error: the variable 'f_max' does not exist")
        f_max = -1
    try:
        f_min
        try:
           y = f_min
        except:
            error.append("*** error: 'f_min' has not been initialized")
            f_min = -1
    except:
        error.append("*** error: the variable 'f_min' does not exist")
        f_min = -1

if type(n) != int:
    error.append("*** error: 'n' is not an integer: it is {0} and it has type {1}".format(n, type(n)))
if type(num_cyc) != int:
    error.append("*** error: 'num_cyc' is not an integer: it is {0} and it has type {1}".format(num_cyc, type(num_cyc)))

if alg_code == "AB":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}".format(N, type(N)))
    if type(M) != int:
        error.append("*** error: 'M' is not an integer: it is {0} and it has type {1}".format(M, type(M)))
    if type(lambbda) != int and type(lambbda) != float:
        error.append("*** error: 'lambbda' is not an integer or a float: it is {0} and it has type {1}".format(lambbda, type(lambbda)))

if alg_code == "FF":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}".format(N, type(N)))
    if type(lambbda) != int and type(lambbda) != float:
        error.append("*** error: 'lambbda' is not an integer or a float: it is {0} and it has type {1}".format(lambbda, type(lambbda)))
    if type(alpha) != int and type(alpha) != float:
        error.append("*** error: 'alpha' is not an integer or a float: it is {0} and it has type {1}".format(alpha, type(alpha)))

if alg_code == "CS":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}".format(N, type(N)))
    if type(p) != int and type(p) != float:
        error.append("*** error: 'p' is not an integer or a float: it is {0} and it has type {1}".format(p, type(p)))
    if type(q) != int and type(q) != float:
        error.append("*** error: 'q' is not an integer or a float: it is {0} and it has type {1}".format(q, type(q)))
    if type(alpha) != int and type(alpha) != float:
        error.append("*** error: 'alpha' is not an integer or a float: it is {0} and it has type {1}".format(alpha, type(alpha)))
    if type(beta) != int and type(beta) != float:
        error.append("*** error: 'beta' is not an integer or a float: it is {0} and it has type {1}".format(beta, type(beta)))

if alg_code == "WO":
    if type(N) != int:
        error.append("*** error: 'N' is not an integer: it is {0} and it has type {1}\n".format(N, type(N)))
    if type(b) != int and type(b) != float:
        error.append("*** error: 'b' is not an integer or a float: it is {0} and it has type {1}".format(b, type(b)))

if alg_code == "BA":
    if type(sigma) != int and type(sigma) != float:
        error.append("*** error: 'sigma' is not an integer or a float: it is {0} and it has type {1}".format(sigma, type(sigma)))
    if type(f_min) != int and type(f_min) != float:
        error.append("*** error: 'f_min' is not an integer or a float: it is {0} and it has type {1}".format(f_min, type(f_min)))
    if type(f_max) != int and type(f_max) != float:
        error.append("*** error: 'f_max' is not an integer or a float: it is {0} and it has type {1}".format(f_max, type(f_max)))

if type(min_f) != int and type(min_f) != float:
    error.append("*** error: there is no real-valued variable 'min_f'")
if type(minimum) != list:
    error.append("*** error: there is no tuple 'minimum' giving the minimum point")
elif type(n) == int and len(minimum) != n:
    error.append("*** error: there is no {0}-tuple 'minimum' giving the minimum point; you have a {1}-tuple".format(n, len(minimum)))
elif type(n) == int:
    for i in range(0, n):
        if not "int" in str(type(minimum[i])) and not "float" in str(type(minimum[i])):
            error.append("*** error: the value for component {0} (ranging from 1 to {1}) in the minimum point is not numeric\n".format(i + 1, n))

if error != []:
    print("\n*** ERRORS: there were errors in your execution:")
    length = len(error)
    for i in range(0, length):
        print(error[i])
    print("\n Fix these errors and run your code again.\n")
else:
    print("\nYou have found a minimum value of {0} and a minimum point of {1}.".format(min_f, minimum))
    print("Your elapsed time was {0} seconds.\n".format(elapsed_time))
    

