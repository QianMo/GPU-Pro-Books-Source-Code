from pprint import *
from numpy import *
from PIL import Image

SIZE = 32
A = {}

for i in range(0, 64):
    left = 0.5
    t = []
    for j in range(i):
        x = i / 2.0
        right = 0.5 * (x - j - 1) / x
        a = abs((left + right) / 2) if sign(left) == sign(right) or abs(left) != abs(right) else 0.5 * abs(left / 2)
        t += [a]
        left = right
    A[i] = t

T = zeros((SIZE,SIZE))

for left in range(SIZE):
    for right in range(SIZE):
        x = left + right + 1
        T[left][right] = A[x][left]

pprint(T)

image = Image.new("L", (SIZE, SIZE))
for y in range(SIZE):
    for x in range(SIZE):
        val = int(255.0 * T[x][y])
        image.putpixel((x, y), val)
image.save("areas2d.tif")
