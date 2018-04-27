/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/

#ifndef KLMATHS_H
#define KLMATHS_H

#include <math.h>

template<class T> inline T sign( T a ) { return ( a > 0 ) ? (T)(1) : ( ( a < 0 ) ? (T)(-1) : (T)(0) ); }

inline void sincos(float a, float &s, float &c) {
#ifdef _WIN32
    _asm {
        fld		a;
        fsincos;
        mov		ecx, c;
        mov		edx, s;
        fstp	dword ptr [ecx];
        fstp	dword ptr [edx];
    }
#else
    s = sinf( a );
    c = cosf( a );
#endif
}

template<class T> inline T cotan( T a ) {
    return ((T)1) / tan(a); 
}

#ifndef __CUDACC__
inline float log2(float a) {
    return(log(a)/log(2.0f));
}
#endif

inline int iDivUp(int a, int b){
    return (a + (b - 1)) / b;
}

inline bool isPow2(int x) {
    return (x>0) && (!((x)&((x)-1)));
}

template<class T> inline T min(T a, T b) {
    return ( a < b ) ? a : b;
}

template<class T> inline T max(T a, T b) {
    return ( a > b ) ? a : b;
}

#ifndef M_PI
#define M_PI 3.14159265f
#endif

#endif //KLMATHS_H