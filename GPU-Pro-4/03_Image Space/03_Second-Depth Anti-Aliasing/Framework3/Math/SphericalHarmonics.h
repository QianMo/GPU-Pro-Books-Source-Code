
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

#ifndef _SPHERICALHARMONICS_H_
#define _SPHERICALHARMONICS_H_

#include "../Imaging/Image.h"
#include "Vector.h"

#define MAX_BANDS 256

void initSH();

float SH(const int l, const int m, const float theta, const float phi);
float SH(const int l, const int m, const float3 &pos);
float SH_A(const int l, const int m, const float3 &pos);
float SH_A2(const int l, const int m, const float3 &pos);

template <typename FLOAT>
bool cubemapToSH(FLOAT *dst, const Image &img, const int bands);

template <typename FLOAT>
bool shToCubemap(Image &img, const int size, const FLOAT *src, const int bands);

template <typename FLOAT>
void computeSHCoefficients(FLOAT *dest, const int bands, const float3 &pos, const bool fade);

#endif // _SPHERICALHARMONICS_H_
