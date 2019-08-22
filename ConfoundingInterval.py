#=============================================================================
#   Copyright (c) 2019 by Mark A. Abramson
#
#   ConfoundingInterval.py is free software; you can redistribute it and/or
#   modify it under the terms of the GNU General Public License as published
#   by the Free Software Foundation; either version 3 of the License, or
#   (at your option) any later version.
#
#   ConfoundingInterval.py is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#   Public License for more details.  A copy of the GNU General Public License
#   is available at
#   http://www.gnu.org/licenses/gpl-3.0.en.html.
#=============================================================================

__author__ = 'Mark Abramson'
from math import *
import numpy          as np
import scipy.optimize as opt

#=============================================================================
# ConfoundingInterval:  Class for computing confounding interval
#-----------------------------------------------------------------------------
#  VARIABLES:
#    rho           = measured correlation coefficient
#    sy            = measured standard deviation of X data
#    sx            = measured standard deviation of X data
#    lb/ub         = 3-element arrays of lower/upper bounds on the variables
#      [0]         =   Coefficient of determination in x
#      [1]         =   Coefficient of determination in y
#      [2]         =   Pearson correlation coefficient
#    tol           = constraint tolerance (to allow for roundoff error)
#    fx_min/fx_max = minimum/maximum function values
#    x_min/x_max   = points that yield the minimum/maximum values of f
#    msg           = part of an error message
#-----------------------------------------------------------------------------
#  METHODS:
#    f        = compute objective function value of the optimization problem
#    g        = compute nonlinear constraint value of the optimization problem
#    optimize = solve the optimization problem, either by AOK or SLSQP
#    opt_msg  = construct optimization output message
#=============================================================================

class ConfoundingInterval:

    # Class constructor
    def __init__(self, rho, sy, sx, lb, ub, tol=1e-12):

        # Input error checking
        if not (len(lb) == 3 and len(ub) == 3):
            raise Exception( 'Vectors LB and UB must have length 3.')
        if not (-1 < rho < 1):
            raise ValueError('Constant rho(x,y) must lie in (-1,1).')
        if not (sy >= 0):
            raise ValueError(u'Standard Deviation sigma(y) must be \u2265 0.')
        if not (sx >= 0):
            raise ValueError(u'Standard Deviation sigma(x) must be \u2265 0.')

        msg = u' bounds must have 0 \u2264 LB \u2264 UB < 1.'
        if not ( 0 <= lb[0] <= ub[0] <  1):
            raise ValueError('Parameter 1' + msg)
        if not ( 0 <= lb[1] <= ub[1] <  1):
            raise ValueError('Parameter 2' + msg)
        if not (-1 <= lb[2] <= ub[2] <= 1):
            raise ValueError('Parameter 3' + msg)
        if not (0 < tol < 1):
            raise ValueError('Tolerance tol must be in (0, 0.1).')

        # Class variables
        self.rho = rho
        self.sx = sx
        self.sy = sy
        self.lb = np.array([sqrt(lb[0]), sqrt(lb[1]), lb[2]])
        self.ub = np.array([sqrt(ub[0]), sqrt(ub[1]), ub[2]])
        self.tol = tol
        self.x_min = []
        self.x_max = []
        self.fx_min = np.nan
        self.fx_max = np.nan

    #=========================================================================
    # f:  Class method for computing objective function valuee
    #-------------------------------------------------------------------------
    #  VARIABLES:
    #   x = 3-d array to be evaluated
    #   s = parameter that gets a value in {-1,1}
    #=========================================================================
    def f(self, x, s = 1):
        return s*(self.rho - np.prod(x))/(1-x[0]**2)

    #=========================================================================
    # g:  Class method for computing nonlinear constraint value
    #-------------------------------------------------------------------------
    #  VARIABLES:
    #   x = 3-d array to be evaluated
    #   s = parameter that gets a value in {-1,1}
    #=========================================================================
    def g(self, x, s = 1):
        gx = s*1e+5
        if (x[0] <= 1 and x[1] <= 1 and x[0] > 0 and x[1] > 0):
            gx = (self.rho + s*sqrt(1-x[0]**2)*sqrt(1-x[1]**2))/(x[0]*x[1])
        return gx

    #=========================================================================
    # getX:  Class method for computing set X of candidate optima
    #-------------------------------------------------------------------------
    #  FUNCTIONS: q, G
    #  VARIABLES:
    #   Case1A = array of candidate optima from Case 1A (see paper) 
    #   Case1B = array of candidate optima from Case 1B (see paper)
    #   Case2A = array of candidate optima from Case 2A (see paper)
    #   Case2B = array of candidate optima from Case 2B (see paper)
    #   Case2C = array of candidate optima from Case 2C (see paper)
    #   X      = array containing the concatenation of the 5 cases
    #   <rest> = temporary storage used in constructing the cases
    #=========================================================================
    def getX(self):

        #=====================================================================
        # q:  Computing real roots of quadratic function via quadratic formula
        #=====================================================================
        def q(A,B,C,s):
            phi = B**2 - 4*A*C
            return 2.0 if (phi < 0 or A == 0) else (-B + s*sqrt(phi))/(2*A)

        # Shortcuts
        lb  = self.lb
        ub  = self.ub
        c   = self.rho
        tol = self.tol
        x23 = np.array([ lb[1]*lb[2], ub[1]*lb[2], lb[1]*ub[2], ub[1]*ub[2] ])
        num = np.array([ sqrt(1+c), sqrt(1+c), sqrt(1-c), sqrt(1-c) ])
        den = np.array([ sqrt(1+lb[2]), sqrt(1+ub[2]), 
                         sqrt(1-lb[2]), sqrt(1-ub[2]) ])
        ind = np.argwhere(np.logical_or(den != 0, num < den))
        y   = num[ind]/den[ind]
        y   = y[y < 1]
        Q   = np.array([
              q((lb[2]**2-1)*ub[0]**2+1,-2*c*lb[2]*ub[0],ub[0]**2+c**2-1, 1),
              q((ub[2]**2-1)*ub[0]**2+1,-2*c*ub[2]*ub[0],ub[0]**2+c**2-1, 1),
              q((lb[2]**2-1)*ub[0]**2+1,-2*c*lb[2]*ub[0],ub[0]**2+c**2-1,-1),
              q((ub[2]**2-1)*ub[0]**2+1,-2*c*ub[2]*ub[0],ub[0]**2+c**2-1,-1),
              q((lb[2]**2-1)*lb[1]**2+1,-2*c*lb[2]*lb[1],lb[1]**2+c**2-1, 1),
              q((ub[2]**2-1)*lb[1]**2+1,-2*c*ub[2]*lb[1],lb[1]**2+c**2-1, 1),
              q((lb[2]**2-1)*lb[1]**2+1,-2*c*lb[2]*lb[1],lb[1]**2+c**2-1,-1),
              q((ub[2]**2-1)*lb[1]**2+1,-2*c*ub[2]*lb[1],lb[1]**2+c**2-1,-1),
              ])

        # Construct array of candidate optima, one case at a time
        case_1a = np.array([ [lb[0], lb[1], lb[2]], [lb[0], ub[1], lb[2]],
                             [lb[0], lb[1], ub[2]], [lb[0], ub[1], ub[2]],
                             [ub[0], lb[1], lb[2]], [ub[0], ub[1], lb[2]],
                             [ub[0], lb[1], ub[2]], [ub[0], ub[1], ub[2]] ])

        case_1b = np.array([ [q(x23[0],-2*c,x23[0], 1), lb[1], lb[2]],
                             [q(x23[1],-2*c,x23[1], 1), ub[1], lb[2]],
                             [q(x23[2],-2*c,x23[2], 1), lb[1], ub[2]],
                             [q(x23[3],-2*c,x23[3], 1), ub[1], ub[2]],
                             [q(x23[0],-2*c,x23[0],-1), lb[1], lb[2]],
                             [q(x23[1],-2*c,x23[1],-1), ub[1], lb[2]],
                             [q(x23[2],-2*c,x23[2],-1), lb[1], ub[2]],
                             [q(x23[3],-2*c,x23[3],-1), ub[1], ub[2]] ])

        case_2a = np.array([ [ub[0], lb[1], self.g([ub[0],lb[1]], 1)],
                             [ub[0], lb[1], self.g([ub[0],lb[1]],-1)] ])

        case_2b = np.empty(shape=[0,3])
        for k in range(len(y)):
            case_2b = np.concatenate((case_2b,
                      np.array([[y[k],y[k],self.g([y[k],y[k]], 1)], 
                                [y[k],y[k],self.g([y[k],y[k]],-1)]])),0)

        case_2c = np.array([ [ub[0],  Q[0], self.g([ub[0], Q[0]], 1)],
                             [ub[0],  Q[0], self.g([ub[0], Q[0]],-1)],
                             [ub[0],  Q[1], self.g([ub[0], Q[1]], 1)],
                             [ub[0],  Q[1], self.g([ub[0], Q[1]],-1)],
                             [ub[0],  Q[2], self.g([ub[0], Q[2]], 1)],
                             [ub[0],  Q[2], self.g([ub[0], Q[2]],-1)],
                             [ub[0],  Q[3], self.g([ub[0], Q[3]], 1)],
                             [ub[0],  Q[3], self.g([ub[0], Q[3]],-1)],
                             [ Q[4], lb[1], self.g([ Q[4],lb[1]], 1)],
                             [ Q[4], lb[1], self.g([ Q[4],lb[1]],-1)],
                             [ Q[5], lb[1], self.g([ Q[5],lb[1]], 1)],
                             [ Q[5], lb[1], self.g([ Q[5],lb[1]],-1)],
                             [ Q[6], lb[1], self.g([ Q[6],lb[1]], 1)],
                             [ Q[6], lb[1], self.g([ Q[6],lb[1]],-1)],
                             [ Q[7], lb[1], self.g([ Q[7],lb[1]], 1)],
                             [ Q[7], lb[1], self.g([ Q[7],lb[1]],-1)] ])

        X = np.concatenate((case_1a,case_1b,case_2a,case_2b,case_2c),0)
        # Delete infeasible candidate optima
        X = X[np.logical_and(X[:,0] >= lb[0]-tol, X[:,0] <= ub[0]+tol)]
        X = X[np.logical_and(X[:,1] >= lb[1]-tol, X[:,1] <= ub[1]+tol)]
        X = X[np.logical_and(X[:,2] >= lb[2]-tol, X[:,2] <= ub[2]+tol)]
        X = X[np.logical_and(X[:,2] >= np.apply_along_axis(self.g,1,X,-1)-tol,
                             X[:,2] <= np.apply_along_axis(self.g,1,X,1)+tol)]
        return X

    #=========================================================================
    # optimize:  Call a solver to solve the optimization problem
    #-------------------------------------------------------------------------
    #  FUNCTIONS:  AOK, getStartPoints, callSLSQP
    #  VARIABLES:
    #   algorithm     = choice of algorithm [AOK,SLSQP]
    #   type_points   = choice of SLSQP point generating method [grid,random]
    #   n_points      = number of random start points or axis grid points
    #   X0            = array of SLSQP start points
    #   fx_min/fx_max = minimum/maximum function values
    #   x_min/x_max   = points that yield the minimum/maximum values of f
    #=========================================================================
    def optimize(self,algorithm,type_points='random',n_points=10):

        #=====================================================================
        # aok:  Compute min and max feasible values from a list
        #=====================================================================
        def aok():
            X = self.getX()
            if (len(X) == 0):  return
            fX = np.apply_along_axis(self.f,1,X)
            i_min = np.argmin(fX)
            i_max = np.argmax(fX)
            x_min = X[i_min,:]
            x_max = X[i_max,:]
            self.fx_min = fX[i_min]*self.sy/self.sx
            self.fx_max = fX[i_max]*self.sy/self.sx
            self.x_min = np.array([x_min[0]**2,x_min[1]**2,x_min[2]])
            self.x_max = np.array([x_max[0]**2,x_max[1]**2,x_max[2]])

        #=====================================================================
        # get_start_points: Generate start points for Multistart SLSQP solver
        #=====================================================================
        def get_start_points(type_points,n_points=10):
            if (n_points < 1):
                raise Exception('Must have at least one starting point.')
            if (type_points == 'random'):
                x0 = np.random.uniform(self.lb,self.ub,(nPoints,3))
            elif (type_points == 'grid'):
                if (n_points > 5):  n_points = 5
                x  = np.linspace(self.lb[0],self.ub[0],n_points)
                y  = np.linspace(self.lb[1],self.ub[1],n_points)
                z  = np.linspace(self.lb[2],self.ub[2],n_points+1)
                x0 = np.vstack(np.meshgrid(x,y,z)).reshape(3,-1).T
            else:
                raise TypeError("Invalid type_points; must be [random,grid]")
            return x0

        #=====================================================================
        # call_slsqp: Optimize using a Multistart SLSQP algorithm
        #=====================================================================
        def call_slsqp(x0,s):
            n_points = len(x0)
            f       = lambda x: self.f(x,s)
            bnds    = np.array([self.lb, self.ub]).T
            con     = lambda x: np.array([self.g([x[0],x[1]], 1)-x[2],
                                         x[2]-self.g([x[0],x[1]],-1)])
            X  = np.empty(shape=[0,3])
            for k in range(n_points):
                [x,fx,nI,iMode,sMode] = opt.fmin_slsqp(f,x0[k,:],
                                            f_ieqcons=con,bounds=bnds,
                                            acc=self.tol,iprint=0,
                                            full_output=True)
                if iMode == 0: X = np.append(X,x.reshape((1,len(self.lb))),0)
            fX = np.apply_along_axis(f,1,X)
            i_opt = np.argmin(fX)
            x_opt = X[i_opt]
            fx_opt = self.f(x_opt)*self.sy/self.sx
            x_opt = np.array([x_opt[0]**2, x_opt[1]**2, x_opt[2]])
            return [x_opt,fx_opt]

        # Call optimizer
        if algorithm == 'AOK':
            aok()
        elif algorithm == 'SLSQP':
            x0 = get_start_points(type_points,n_points)
            [self.x_min,self.fx_min] = call_slsqp(x0, 1)
            [self.x_max,self.fx_max] = call_slsqp(x0,-1)
        else:
            raise TypeError('Invalid optimizer type [AOK, SLSQP]')

    #=========================================================================
    # opt_msg:  Store optimal solution for multiple output sources
    #-------------------------------------------------------------------------
    #  VARIABLES:
    #   msg = output message
    #   ci  = confounding interval
    #-------------------------------------------------------------------------
    #  FUNCTIONS:  array2str
    #=========================================================================
    def opt_msg(self):
        ci  = [self.fx_min, self.fx_max]
        msg = 'Confounding Interval:  ' + self.array2str(np.array(ci),4)
        msg += '\n\n   Points that generate interval (R_wx^2, R_wy^2, rho_xy):'
        msg += '\n      Min:  ' + self.array2str(self.x_min,4)
        msg += '\n      Max:  ' + self.array2str(self.x_max,4)
        return msg

    #=========================================================================
    # array2str: Convert a 2-d numpy array to string with d-digit display
    #=========================================================================
    def array2str(self,x,d):
        fmt  = '{0:.' + str(d) + 'f}' 
        x_str = '( '
        for k in range(len(x)):
            x_str = x_str + str.format(fmt,x[k]) + ', '
        x_str = x_str[0:-2] + ' )'
        return x_str

    #=========================================================================
    # getPlotData:  Generate CI data for many fixed Pearson coefficent values
    # ------------------------------------------------------------------------
    #  VARIABLES:
    #   alg = selected optimization algorithm
    #   x   = vector evenly spaced points in (-1,1) for plotting
    #   lo  = confounding interval lower bounds for Pearson coefficient = x
    #   hi  = confounding interval upper bounds for Pearson coefficient = x
    #=========================================================================
    def get_plot_data(self,alg='AOK'):
        x    = np.arange(-0.99, 0.99, 0.01)
        lo   = np.array([None]*len(x))
        hi   = np.array([None]*len(x))
        for k in range(len(x)):
            self.lb[2] = x[k]
            self.ub[2] = x[k]
            self.optimize(alg,'random',36)
            lo[k] = self.fx_min
            hi[k] = self.fx_max
        return [x,lo,hi]

#=============================================================================
# TestProblemLibrary: Dictionary of test problems for ConfoundingInterval
#=============================================================================
TestProblemLibrary = {'Default'  : {'rho': -0.4, 'sy': 1.0,  'sx': 1.0,   
                                    'lb' : [0.4, 0.4, -0.9],  
                                    'ub' : [0.9, 0.9,  0.9]},
                      'Eskenazi' : {'rho': -0.11,'sy': 14.60,'sx': 0.34, 
                                    'lb' : [0.1, 0.0,  0.0],
                                    'ub' : [0.5, 0.2,  1.0]}
                      }
