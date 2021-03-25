import numpy as np
from scipy import integrate
from math import *
##Modulation et démodulation DSB-BC 


def f1(t,theta):
    return exp(t*cos(theta))

def I(t):
    '''fonction de bessel modifiée d'ordre 0'''
    res1=scipy.integrate.quad(f1,0,pi,args=(t))
    return (1/pi)*res1_(sin)

def signal_atransmettre(t,alpha):
    '''on choisit une fenètre de type Kaiser'''
    if abs(t)<=B/2:
        return I(pi*alpha*sqrt(1-(2*t/B)**2))/I(pi*alpha)
    else:
        return I(pi*alpha)