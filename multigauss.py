# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:18:49 2024

@author: Louis
"""

import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from scipy.integrate import quad
from sklearn.mixture import GaussianMixture ###Must install sklearn using miniconda terminal use conda install scikit-learn 
import scipy as scipy


#LateX fonts. Makes the appearence better for latex documents#
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = "Times New Roman"

###Parameters###

##For 100,000 events, simulation takes around 7 mins to finish. Any more 
#and the computing time isnt worth the reward. Can compute for fewer simulations
#but the fitting gets a bit worse around 20,000. This can take about half the time. 

tf = 10  # Final time
N = 1000  # Interval size
dt = tf / N  # Infinitesimal time
sigma = 0.35  # Variance of the random fluctuations Best between 0.4 and 0.3
T = -0.45# Value of T
Events = 100000 # Number of events of Monte Carlo simulation


###Integral Limits###
a1 = -2.5  # Integral Lower limit curve 1
b1 = 0  # Integral Upper limit curve 1

a2 = 0  # Integral Lower limit curve 2
b2 = 2.5  # Integral Upper limit curve 2


###Plotting Limits###
x_i = -2  # x lower limit
x_f = 2   # x upper limit

###Stochastic Differential Equation ###
def sde(tf, N, sigma, T):
    X = [0]

    for t in np.linspace(0, tf, N - 1):
        dW = np.random.normal(0, np.sqrt(dt))
        dX = -(X[-1] ** 3 - X[-1] + T) * dt + sigma * dW
        X.append(X[-1] + dX)
    return X

# Begin Timer #
start = timer()

###Monte-Carlo Simulation###
X_val_f = []
for n in range(Events):
    X_vals = sde(tf, N, sigma, T)
    X_val_f.append(X_vals[-1])

fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
plt.title(r'Probability density of Ito equation for T = {}, sigma ={} '.format(T,sigma), fontsize=20)
plt.ylabel(r'Probability Density', fontsize=20)
plt.xlabel(r'$X \left(t\right)$', fontsize=20)

ax.grid(which="major", linewidth=1)
ax.grid(which="minor", linewidth=0.2)
ax.minorticks_on()
#Turns off the histograms by uncommenting 
#yvals, cont,_ = plt.hist(X_val_f, bins=300, density=True, label= 'Histogram Plot')

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=2)
gmm.fit(np.array(X_val_f).reshape(-1, 1))

# Plot the Gaussian Mixture Model
xset = np.linspace(x_i, x_f, 10000)
density = np.exp(gmm.score_samples(xset.reshape(-1, 1)))
plt.plot(xset, density, 'r', label='Numerical Solution')
plt.fill_between(xset, density,color= "b",alpha= 0.2 )
# Integral under curve
def gmm_density(x):
    if np.isscalar(x):
        return np.exp(gmm.score_samples(np.array([x]).reshape(-1, 1)))
    else:
        return np.exp(gmm.score_samples(x.reshape(-1, 1)))

I1, err1 = quad(gmm_density, a1, b1)
I2, err2 = quad(gmm_density, a2, b2)

print('P(', '{}'.format(a1), '≤ x ≤', '{})'.format(b1), '=', I1, '±', err1, 'curve 1')
print('P(', '{}'.format(a2), '≤ x ≤', '{})'.format(b2), '=', I2, '±', err2, 'curve 2')
print('Sum of probabilities = ', I1 + I2, '±', np.sqrt((err1)**2+(err2)**2) )


def analytical(C,s,T,x):
    
    U = x**4 /4 - x**2 /2 + T*x
    
    
    return C*np.exp((-2*U)/(s**2))


#plt.plot(xset,analytical(0.018,sigma,T, xset), label = 'Analytical Solution')

#I3, err3 = quad(analytical(0.018,sigma,T,xset), a1, b1)
#I4, err4 = quad(analytical(0.018,sigma,T,xset), a2, b2)


print('-------------------Numerical Solutions-----------------')
print('P(', '{}'.format(a1), '≤ x ≤', '{})'.format(b1), '=', I1, '±', err1, 'curve 1')
print('P(', '{}'.format(a2), '≤ x ≤', '{})'.format(b2), '=', I2, '±', err2, 'curve 2')
print('Sum of probabilities = ', I1 + I2, '±', np.sqrt((err1)**2+(err2)**2) )



#popt, pcov = scipy.optimize.curve_fit(analytical, xset, density)

#print(popt)


plt.legend()
# Time stops #
end = timer()
print('calculated in', end - start, 's')