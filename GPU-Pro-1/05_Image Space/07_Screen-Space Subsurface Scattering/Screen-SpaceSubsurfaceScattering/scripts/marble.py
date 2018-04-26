# Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)
# Copyright (C) 2009 Diego Gutierrez (diegog@unizar.es)
#
# This script calculates a 4-gaussian fit of the marble profile described in
# [Jensen et al. 01] using the Levenberg-Marquardt algorithm. By changing the
# parameters 'Rd', 'sigma_sp' and 'sigma_a' it is possible to find fits for
# other materials (see [Jensen et al. 01] for a table). 
# We output the variances and weights of the fit, alongside with the
# corresponding C++ code that can be copy pasted into our app (some
# modifications are required; see comments in the README). 
#
# This script was tested using:
# - python 2.5.2
# - numpy 1.3.0
# - scipy 0.7.1
# - matplotlib 0.91.4

from math import *
from pylab import *
from numpy import *
from scipy import *
from scipy.optimize import leastsq

# Marble parameters (see [Jensen et al. 01])
Rd = [0.83, 0.79, 0.75] # [r, g, b]
sigma_sp = [2.19, 2.62, 3.0]
sigma_a = [0.0021, 0.0041, 0.0071]

# Diffuse BSSRDF, as defined in [Jensen et al. 01]
def R(r, sigma_sp, sigma_a, nu):
    albedo = sigma_sp / (sigma_a + sigma_sp)
    sigma_tp = sigma_sp + sigma_a
    sigma_tr = sqrt(3.0 * sigma_a * sigma_tp)

    F_dr = -1.44 / (nu**2) + 0.71 / nu + 0.668 + 0.0636 * nu
    A = (1.0 + F_dr) / (1.0 - F_dr)
    D = 1.0 / (3.0 * sigma_tp)

    z_r = 1.0 / (sigma_tp)
    z_v = z_r + 4.0 * A * D
   
    d_r = sqrt(z_r**2 + r**2)
    d_v = sqrt(z_v**2 + r**2)

    t1 = (sigma_tr * d_r + 1.0) * exp(-sigma_tr * d_r) / (sigma_tp * d_r**3)
    t2 = z_v * (sigma_tr * d_v + 1.0) * exp(-sigma_tr * d_v) / (sigma_tp * d_v**3)

    return albedo *(t1 + t2) / (4.0 * pi)

# [d'Eon and Luebke 07] gaussian definition
def gaussian(v, r):
    return exp((-(r**2)) / (2.0 * v)) / (2.0 * pi * v)

# Sum of gaussians with weights 'w' and variances 'v'
def gaussiansum(r, w, v):
    return sum([ww * gaussian(vv, r) for vv, ww in zip(v, w)], 0)

# Error function with 8 unknowns: 4 for weights and 4 for variances
def error8(x, R_r, r):
    RR_r = [x[0] * gaussian(x[4], rr) +
            x[1] * gaussian(x[5], rr) +
            x[2] * gaussian(x[6], rr) +
            x[3] * gaussian(x[7], rr)
            for rr in r]

    return R_r - RR_r
    
# Error function with 4 unknowns for the gaussian weights
def error4(x, R_r, r, variance):
    RR_r = [x[0] * gaussian(variance[0], rr) +
            x[1] * gaussian(variance[1], rr) +
            x[2] * gaussian(variance[2], rr) +
            x[3] * gaussian(variance[3], rr)
            for rr in r]

    return R_r - RR_r

# Calculate the profiles using the BSSRDF
r = arange(0, 4, 0.01)
R_red = R(r, sigma_sp[0], sigma_a[0], 1.5)
R_green = R(r, sigma_sp[1], sigma_a[1], 1.5)
R_blue = R(r, sigma_sp[2], sigma_a[2], 1.5)

# Find a fit for the red profile, searching for variances and weights
x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
x, sucess = leastsq(error8, x0, args=(R_red, r))
variance = [x[4], x[5], x[6], x[7]]
W_red = [x[0], x[1], x[2], x[3]]
print "F I T"
print "Variance:", variance
print "W_red:", W_red

# For the green profile we search only for weights
x0 = [0.5, 0.5, 0.5, 0.5]
W_green, sucess = leastsq(error4, x0, args=(R_green, r, variance))
print "W_green:", W_green

# And finally, the same for blue
x0 = [0.5, 0.5, 0.5, 0.5]
W_blue, sucess = leastsq(error4, x0, args=(R_blue, r, variance))
print "W_blue:", W_blue

# Normalize the weights
total = (sum(W_red), sum(W_green), sum(W_blue))
WN_red = [ w / total[0] for w in W_red]
WN_green = [ w / total[1] for w in W_green]
WN_blue = [ w / total[2] for w in W_blue]

# Sort them, as we have to perform the gaussians from narrow to widest
gaussians = [a for a in zip(variance, WN_red, WN_green, WN_blue)]
gaussians.sort()

# Output the C++ copy/paste-able code
print
print "C + +   C O D E"
print "D3DXVECTOR3 weights[] = {"
print "    D3DXVECTOR3(0.0f, 0.0f, 0.0f),"
for v, rr, g, b in gaussians:
    print "    D3DXVECTOR3(%sf, %sf, %sf)," % (rr, g, b)
print "};"

print "float variances[] = {",
for i, (v, rr, g, b) in enumerate(gaussians):
    print "%sf%s" % (v, "," if i < len(gaussians) - 1 else ""),
print "};"
print "gaussians = Gaussian::gaussianSum(variances, weights, %s);" % len(gaussians)

# Calculate the profile resulting from our (unnormalized) 4-gaussian fit 
R_newred = gaussiansum(r, W_red, variance) 
R_newgreen = gaussiansum(r, W_green, variance) 
R_newblue = gaussiansum(r, W_blue, variance) 
         
# Fit for the green profile, as found in [d'Eon and Luebke 07]. We include it 
# for validation purposes
R_gemsgreen = gaussiansum(r, [0.070, 0.18, 0.21, 0.29], [0.036, 0.14, 0.91, 7.0]) 

# Plot the result, where the dotted lines represent our 4-gaussian fit
plot(r, R_red, "r")
plot(r, R_newred, "k:")
plot(r, R_green, "g")
plot(r, R_gemsgreen, "m--")
plot(r, R_newgreen, "k:")
plot(r, R_blue, "b")
plot(r, R_newblue, "k:")
show()
