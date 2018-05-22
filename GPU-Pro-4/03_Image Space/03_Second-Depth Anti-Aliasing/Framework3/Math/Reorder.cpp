
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "Reorder.h"

unsigned int evenDilate(const unsigned int val){
	unsigned int u = ((val & 0x0000ff00) << 8) | (val & 0x000000ff);
	unsigned int v = ((  u & 0x00f000f0) << 4) | (  u & 0x000f000f);
	unsigned int w = ((  v & 0x0c0c0c0c) << 2) | (  v & 0x03030303);
	unsigned int r = ((  w & 0x22222222) << 1) | (  w & 0x11111111);
	return r;
}

#define oddDilate(val) ((evenDilate(val) << 1))

unsigned int mortonToLinear(unsigned int x, unsigned int y){
	return (evenDilate(x) | oddDilate(y));
}


static const unsigned char ihtab[] = { 0x2, 0x4, 0xD, 0x8, 0x9, 0x5, 0xC, 0x3, 0x0, 0xF, 0x6, 0xA, 0xB, 0xE, 0x7, 0x1 };

unsigned int hilbertToLinear(unsigned int x, unsigned int y){
    unsigned int t = 0;
    unsigned int c = 0;
    for (unsigned int i = 0; i < 16; i++){
        unsigned int xi = (x >> 14) & 2;
        unsigned int yi = (y >> 15) & 1;
        x <<= 1;
        y <<= 1;

        unsigned char st = ihtab[(c << 2) | xi | yi];
        t <<= 2;
        t |= (st >> 2);
        c = st & 3;
    }

    return t;
}
