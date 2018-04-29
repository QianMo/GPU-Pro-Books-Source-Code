from PIL import Image


def store(val):
    val[0] = val[0] / val[2]
    val[1] = val[1] / val[2]
    val[2] = 0
    return val


input = Image.open("in.tif")
w, h = input.size

output = Image.new("RGBA", (w, h))

for x in range(w):
    for y in range(h):
        p = (x, y)
        val = input.getpixel(p)
        
        val = [-1.0 + 2.0 * a / 255.0 for a in val]
        val = store(val)
        val = [int(255.0 * (0.5 + 0.5 * a)) for a in val]
    	val[2] = 0
        
        output.putpixel(p, tuple(val))

output.save("out.tif")
