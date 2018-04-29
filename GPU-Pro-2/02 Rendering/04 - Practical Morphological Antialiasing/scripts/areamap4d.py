from pprint import *
from PIL import Image

SIZE = 9
def arrange(v1, v2):
    return v1, v2, 0

areas = Image.open("areas2d.tif")
image = Image.new("RGB", (SIZE * 5, SIZE * 5))

for e2 in range(5):
    for e1 in range(5):
        for left in range(SIZE):
            for right in range(SIZE):  
              
                p = left, right
                a = areas.getpixel(p)
                p = p[0] + e1 * SIZE, p[1] + e2 * SIZE
                
                if (e1 == 2) or (e2 == 2):
                    image.putpixel(p, arrange(0,0))
                elif left > right:
                    if e2 == 0:
                        image.putpixel(p, arrange(0,0))
                    elif e2 == 1:
                        image.putpixel(p, arrange(0,a))
                    elif e2 == 3:
                        image.putpixel(p, arrange(a,0))
                    else:
                        image.putpixel(p, arrange(a,a))
                elif left < right:
                    if e1 == 0:
                        image.putpixel(p, arrange(0,0))
                    elif e1 == 1:
                        image.putpixel(p, arrange(0,a))
                    elif e1 == 3:
                        image.putpixel(p, arrange(a,0))
                    else:
                        image.putpixel(p, arrange(a,a))
                else:
                    if (e1+e2) == 0:
                        image.putpixel(p, arrange(0,0))
                    elif (e1+e2) == 1:
                        image.putpixel(p, arrange(0,a))
                    elif (e1+e2) == 2:
                        image.putpixel(p, arrange(0,2*a))
                    elif (e1+e2) == 3:
                        image.putpixel(p, arrange(a,0))
                    elif (e1+e2) == 4:
                        image.putpixel(p, arrange(a,a))
                    elif (e1+e2) == 5:
                        image.putpixel(p, arrange(a,2*a))
                    elif (e1+e2) == 6:
                        image.putpixel(p, arrange(2*a,0))
                    elif (e1+e2) == 7:
                        image.putpixel(p, arrange(2*a,a))
                    else:
                        image.putpixel(p, arrange(2*a,2*a))
                        
image.save("areas4d.tif")
