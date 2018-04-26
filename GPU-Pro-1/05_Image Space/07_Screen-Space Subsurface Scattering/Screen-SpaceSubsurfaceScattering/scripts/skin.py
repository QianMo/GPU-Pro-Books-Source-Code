# Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)
# Copyright (C) 2009 Diego Gutierrez (diegog@unizar.es)
#
# This script calculates a 4-gaussian fit of the 6-gaussian sum described in
# [d'Eon and Luebke 07]. We output the variances and weights of the fit, 
# alongside with the corresponding C++ code that can be copy pasted into our app 
# (some modifications are required; see comments in the README). 
#
# This script was tested using:
# - python 2.5.2
# - numpy 1.3.0
# - scipy 0.7.1
# - matplotlib 0.91.4
#
# Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)

from math import *
from pylab import *
from numpy import *
from scipy import *
from scipy.optimize import leastsq


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

# Calculate the 6-gaussian sum described in [d'Eon and Luebke 07] 
v = [0.0064, 0.0484, 0.187, 0.567, 1.99, 7.41]
r = arange(0, 7, 0.01)
R_red = gaussiansum(r, [0.233, 0.1, 0.118, 0.113, 0.358, 0.078], v)
R_green = gaussiansum(r, [0.455, 0.336, 0.198, 0.007, 0.004, 0.0], v)
R_blue = gaussiansum(r, [0.649, 0.344, 0.0, 0.007, 0.0, 0.0], v)

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

# Limit the minimum values
W_red = [ max(0.0, w) for w in W_red]
W_green = [ max(0.0, w) for w in W_green]
W_blue = [ max(0.0, w) for w in W_blue]

# Normalize the weights
total = (sum(W_red), sum(W_green), sum(W_blue))
W_red = [ w / total[0] for w in W_red]
W_green = [ w / total[1] for w in W_green]
W_blue = [ w / total[2] for w in W_blue]

# Sort them, as we have to perform the gaussians from narrow to widest
gaussians = [g for g in zip(variance, W_red, W_green, W_blue)]
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

# Calculate the profile using our 4 gaussian fit
R_newred = gaussiansum(r, W_red, variance) 
R_newgreen = gaussiansum(r, W_green, variance) 
R_newblue = gaussiansum(r, W_blue, variance) 

# Plot the result, where the dotted lines represent our 4-gaussian fit
plot(r, r * R_red, "r")
plot(r, r * R_newred, "k:")
plot(r, r * R_green, "g")
plot(r, r * R_newgreen, "k:")
plot(r, r * R_blue, "b")
plot(r, r * R_newblue, "k:")
show()
