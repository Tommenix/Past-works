"""
Here I imported functions from Scipy.integrate to satisfy the implementation requirement in the document of odesolver.
"""

import numpy as np
import scipy.integrate
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
from scipy.integrate._ivp.base import check_arguments
from scipy.interpolate import interp1d
from scipy.integrate._ivp.common import (validate_max_step, validate_tol, select_initial_step,
                     norm, warn_extraneous, validate_first_step)
from scipy.integrate import odeint, solve_ivp

def fwd_euler_step(fun, t0, y, f, h, direction):
    """
    Perform a single Forward Euler step.
    
    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Starting time.
    y : ndarray, shape (n,)
        Current state.
    h : float
        Step to use.
        If h is not entered, then we modify it to the defult step ((t_bound - t0) / 100).
    f : ndarray, shape (n,)
        Current value of the derivative, i.e., ``fun(x, y)``.
    A : ndarray, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.
        The last row is a linear combination of the previous rows with
        coefficients
    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.
    """
    s = direction
    y_new = y + h*f
    f_new = fun(t0+s*h,y_new)
    
    return y_new,f_new


class ForwardEuler(OdeSolver):
    '''
    Base class for Forward Euler methods.
    
    Use object.dense_output() to get the interpolated result.
    '''
    
    
    def __init__(self, fun, y, t0= NotImplemented, t_bound= NotImplemented, 
                   h= NotImplemented, vectorized = bool, **extraneous):
        """
        store ts and ys computed in forward Euler method
        
        These will be used for evaluation.
        
        Parameters
        ----------
        here are some default values: 
        
        h : float
            Step to use.
            If h is not entered, then we modify it to the defult step ((t_bound - t0) / 100).
        
        t0 : float
            Starting time.
            default = 0.
        t_bound : 
            Ending time.
            default = 1.
        y : ndarray, shape (n,)
            Current state.
            If input is 1d, auto convert to 1d array.

        The relevant parameters are (from OdeSolver __init__):
        Returns
        -------
        self.h = h
        
        self.t = t0
        self.t_bound = t_bound
        self._fun, self.y = check_arguments(fun, y0, support_complex)
        self.t_bound = t_bound
        self.vectorized = vectorized
        
        self.fun = fun
        self.fun_single = fun_single
        self.fun_vectorized = fun_vectorized

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.n = self.y.size

        self.nfev = 0
        self.njev = 0
        self.nlu = 0
        """
        warn_extraneous(extraneous)
        if t0 == NotImplemented:
            # We take t0 to default value 0 if it is not implemented by user.
            t0 = 0
        if t_bound == NotImplemented:
            # We take t_bound to default value 1 if it is not implemented by user.
            t_bound = 1
        if h == NotImplemented:
            # We take h to default value if it is not implemented by user.
            h = (t_bound-t0)/100
        if h > (t_bound-t0)/10:
            return "Your step size is too large. Please choose a step size that is smaller than 1/10 of the total time."
        if type(y) == int or type(y) == float:
            # This is to satisfy the condition of check arguments of vectorized solution.
            y = [y]

        self.h = h
        super(ForwardEuler, self).__init__(fun, t0, y, t_bound, vectorized,support_complex=True)
        self.t_old = NotImplemented

    
    def _step_impl(self):
        t = self.t
        y_new = self.y
        h = self.h
        s = self.direction
        N = int(s*(self.t_bound-t)//h)
        t_new = t
        f_new = self.fun(t_new,y_new)
        ys = [y_new[0]]
        for i in range(N+1):
            y_new, f_new = fwd_euler_step(self.fun, t_new, y_new, f_new, h, s)
            t_new += h
            ys.append(y_new[0])
        return ys
            
    def _dense_output_impl(self):
        t = self.t
        y = self.y
        h = self.h
        s = self.direction
        N = int(s*(self.t_bound-t)//h)
        ts = []
        for i in range(N+2):
            ts.append(t+i*h)
        ys = self._step_impl()
        return ForwardEulerOutput(ts, ys)
    
    def test(self):
        return self._step_impl()
        
    def testt(self):
        t = self.t
        y = self.y
        h = self.h
        s = self.direction
        N = int(s*(self.t_bound-t)//h)
        ts = []
        for i in range(N+2):
            ts.append(t+i*h)
        return ts
    
    
class ForwardEulerOutput(DenseOutput):
    """
    Interpolate ForwardEuler output

    """
    def __init__(self, ts, ys):

        """
        store ts and ys computed in forward Euler method

        These will be used for evaluation
        """
        super(ForwardEulerOutput, self).__init__(np.min(ts), np.max(ts))
        self.interp = interp1d(ts, ys, kind='linear', copy=True)



    def _call_impl(self, t):
        """
        Evaluate on a range of values
        """
        return self.interp(t)