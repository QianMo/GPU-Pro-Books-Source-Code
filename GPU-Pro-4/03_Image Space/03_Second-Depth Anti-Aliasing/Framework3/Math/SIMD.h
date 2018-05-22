
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

#ifndef _SIMD_H_
#define _SIMD_H_

#include "Vector.h"
/*
#undef pfacc
#undef pfmul
__forceinline void planeDistance3DNow(const vec4 *plane, const vec4 *point, int *result){
	__asm {
		mov eax, plane
		mov ebx, point
		mov ecx, result

		movq mm0, [eax]     // [nx, ny]
		movq mm1, [eax + 8] // [nz, d]

		movq mm2, [ebx]     // [px, py]
		movq mm3, [ebx + 8] // [pz, 1]

		pfmul mm0, mm2      // [nx * px, ny * py]
		pfmul mm1, mm3      // [nz * pz, d]

		pfacc mm0, mm1
		pfacc mm0, mm0

		movd [ecx], mm0
	}
}
*/

#include "../CPU.h"

forceinline v4sf dot4(v4sf u, v4sf v){
	v4sf m = mulps(u, v);
	v4sf f = shufps(m, m, SHUFFLE(2, 3, 0, 1));
	m = addps(m, f);
	f = shufps(m, m, SHUFFLE(1, 0, 3, 2));
	return addps(m, f);
}

#endif
