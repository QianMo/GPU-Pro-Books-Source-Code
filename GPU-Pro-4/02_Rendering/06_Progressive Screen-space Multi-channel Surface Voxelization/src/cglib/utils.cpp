
#include <math.h>

int
exp2i (int val)
{
    int     i, p2 = 1;

    for (i = 0; i < val; i++)
        p2 <<= 1;

    return p2;
}

int
getPowerOfTwo (int val)
{
    int     p2 = 1;

    while (p2 < val)
        p2 <<= 1;

    return p2;
}

