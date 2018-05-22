
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

#ifndef _CPU_H_
#define _CPU_H_

#include "Platform.h"

extern int cpuCount;
extern int cpuFamily;
extern char cpuVendor[13];
extern char cpuBrandName[49];
extern bool cpuCMOV, cpu3DNow, cpu3DNowExt, cpuMMX, cpuMMXExt, cpuSSE, cpuSSE2;


#if defined(_WIN32)
#define CPUID __asm _emit 0x0F __asm _emit 0xA2
#define RDTSC __asm _emit 0x0F __asm _emit 0x31

#ifdef _WIN64
#include <intrin.h>

#define cpuid(func, a, b, c, d) { int res[4]; __cpuid(res, func); a = res[0]; b = res[1]; c = res[2]; d = res[3]; }

#else
void cpuidAsm(uint32 func, uint32 *a, uint32 *b, uint32 *c, uint32 *d);
#define cpuid(func, a, b, c, d) cpuidAsm(func, &a, &b, &c, &d)


#pragma warning(disable: 4035)
inline uint64 getCycleNumber(){
	__asm {
		RDTSC
	}
}
#endif

#else

#if defined(__APPLE__)

void cpuidAsm(uint32 op, uint32 reg[4]);
#define cpuid(func, a, b, c, d) { uint32 res[4]; cpuidAsm(func, res); a = res[0]; b = res[1]; c = res[2]; d = res[3]; }

#elif defined(LINUX)
#define cpuid(in, a, b, c, d) asm volatile ("cpuid": "=a" (a), "=b" (b), "=c" (c), "=d" (d) : "a" (in));
#endif

#define rdtsc(a, d)\
  asm volatile ("rdtsc" : "=a" (a), "=d" (d));

inline uint64 getCycleNumber(){
	union {
		struct {
			unsigned int a, d;
		};
		long long ad;
	} ads;

	rdtsc(ads.a, ads.d);
	return ads.ad;
}

#endif

uint64 getHz();
void initCPU();




// SIMD tool-kit
#define SHUFFLE(r0, r1, r2, r3) (((r3) << 6) | ((r2) << 4) | ((r1) << 2) | (r0))

#define CONCAT(f0, f1) f0##f1
#define CONCAT2(f0, f1, f2) f0##f1##f2
#define CONCAT3(f0, f1, f2, f3) f0##f1##f2##f3

#ifdef _WIN32
/*
#include <mmintrin.h>
#include <emmintrin.h>
#include <mm3dnow.h>

typedef __m64 v8qi;
typedef __m64 v4hi;
typedef __m64 v2si;
typedef __m64 v2sf;
typedef __m64 di;

typedef __m128i v4si;
typedef __m128  v4sf;

typedef __m128d v2df;
typedef __m128i v2di;
typedef __m128i v8hi;
typedef __m128i v16qi;
*/

#define INST(inst) CONCAT(_m_, inst)
#define INSTI(inst) CONCAT(_m_,inst,i)

#define SINST(inst, s) CONCAT3(_mm_,inst,_,s)
#define CINST(inst) CONCAT2(_mm_,inst,_ss)

#else

/*
//typedef int   v8qi __attribute__ ((vector_size(8)));
typedef int   v8qi __attribute__ ((mode(V8QI)));
typedef int   v4hi __attribute__ ((mode(V4HI)));
typedef int   v2si __attribute__ ((mode(V2SI)));
typedef float v2sf __attribute__ ((mode(V2SF)));
typedef int   di   __attribute__ ((mode(DI)));

typedef int   v4si __attribute__ ((mode(V4SI)));
typedef float v4sf __attribute__ ((mode(V4SF)));

typedef float v2df  __attribute__ ((mode(V2DF)));
typedef int   v2di  __attribute__ ((mode(V2DI)));
typedef int   v4si  __attribute__ ((mode(V4SI)));
typedef int   v8hi  __attribute__ ((mode(V8HI)));
typedef int   v16qi __attribute__ ((mode(V16QI)));
*/
#define INST(inst) CONCAT(__builtin_ia32_, inst)
#define INSTI(inst) INST(inst)

#define SINST(inst, s) CONCAT2(__builtin_ia32_, inst, s)
#define CINST(inst) INST(inst)

#endif

// MMX arithmetic ops
#define paddb   INST(paddb)
#define paddw   INST(paddw)
#define paddd   INST(paddd)
#define psubb   INST(psubb)
#define psubw   INST(psubw)
#define psubd   INST(psubd)
#define paddsb  INST(paddsb)
#define paddsw  INST(paddsw)
#define psubsb  INST(psubsb)
#define psubsw  INST(psubsw)
#define paddusb INST(paddusb)
#define paddusw INST(paddusw)
#define psubusb INST(psubusb)
#define psubusw INST(psubusw)
#define pmullw  INST(pmullw)
#define pmulhw  INST(pmulhw)

// MMX logical ops
#define pand   INST(pand)
#define pandn  INST(pandn)
#define por    INST(por)
#define pxor   INST(pxor)
#define psllwv INST (psllw)
#define psllwi INSTI(psllw)
#define pslldv INST (pslld)
#define pslldi INSTI(pslld)
#define psllqv INST (psllq)
#define psllqi INSTI(psllq)
#define psrawv INST (psraw)
#define psrawi INSTI(psraw)
#define psradv INST (psrad)
#define psradi INSTI(psrad)
#define psrlwv INST (psrlw)
#define psrlwi INSTI(psrlw)
#define psrldv INST (psrld)
#define psrldi INSTI(psrld)
#define psrlqv INST (psrlq)
#define psrlqi INSTI(psrlq)

// MMX comparison ops
#define pcmpeqb INST(pcmpeqb)
#define pcmpeqw INST(pcmpeqw)
#define pcmpeqd INST(pcmpeqd)
#define pcmpgtb INST(pcmpgtb)
#define pcmpgtw INST(pcmpgtw)
#define pcmpgtd INST(pcmpgtd)

// MMX pack ops
#define punpckhbw INST(punpckhbw)
#define punpckhwd INST(punpckhwd)
#define punpckhdq INST(punpckhdq)
#define punpcklbw INST(punpcklbw)
#define punpcklwd INST(punpcklwd)
#define punpckldq INST(punpckldq)
#define packsswb  INST(packsswb)
#define packssdw  INST(packssdw)
#define packuswb  INST(packuswb)

// MMX misc ops
#ifdef _WIN32
#define emms      _m_empty
#define setzero64 _mm_setzero_si64
#else
#define emms      __builtin_ia32_emms
#define setzero64 __builtin_ia32_mmx_zero
#endif



// 3DNow!
#define femms      INST(femms)
#define pavgusb    INST(pavgusb)
#define pf2id      INST(pf2id)
#define pfacc      INST(pfacc)
#define pfadd      INST(pfadd)
#define pfcmpeq    INST(pfcmpeq)
#define pfcmpge    INST(pfcmpge)
#define pfcmpgt    INST(pfcmpgt)
#define pfmax      INST(pfmax)
#define pfmin      INST(pfmin)
#define pfmul      INST(pfmul)
#define pfrcp      INST(pfrcp)
#define pfrcpit1   INST(pfrcpit1)
#define pfrcpit2   INST(pfrcpit2)
#define pfrsqrt    INST(pfrsqrt)
#define pfrsqrtit1 INST(pfrsqrtit1)
#define pfsub      INST(pfsub)
#define pfsubr     INST(pfsubr)
#define pi2fd      INST(pi2fd)
#define pmulhrw    INST(pmulhrw)

// 3DNow! Extended
#define pf2iw    INST(pf2iw)
#define pfnacc   INST(pfnacc)
#define pfpnacc  INST(pfpnacc)
#define pi2fw    INST(pi2fw)
#define pswapdsf INST(pswapdsf)
#define pswapdsi INST(pswapdsi)

// SSE arithmetic ops
#define addps   SINST(add, ps)
#define subps   SINST(sub, ps)
#define mulps   SINST(mul, ps)
#define divps   SINST(div, ps)
#define addss   SINST(add, ss)
#define subss   SINST(sub, ss)
#define mulss   SINST(mul, ss)
#define divss   SINST(div, ss)
#define maxps   SINST(max, ps)
#define maxss   SINST(max, ss)
#define minps   SINST(min, ps)
#define minss   SINST(min, ss)
#define rcpps   SINST(rcp,   ps)
#define rsqrtps SINST(rsqrt, ps)
#define sqrtps  SINST(sqrt,  ps)
#define rcpss   SINST(rcp,   ss)
#define rsqrtss SINST(rsqrt, ss)
#define sqrtss  SINST(sqrt,  ss)

// SSE misc ops
#define movss    SINST(MOVE,   ss)
#define movhlps  SINST(MOVEHL, ps)
#define movlhps  SINST(MOVELH, ps)

#ifdef _WIN32
#define shufps _mm_shuffle_ps
#else
#define shufps __builtin_ia32_shufps
#endif

// SSE conversion ops
#define unpckhps  SINST(UNPACKHI, ps)
#define unpcklps  SINST(UNPACKLO, ps)
#define cvtpi2ps  SINST(cvt,  pi2ps)
#define cvtsi2ss  SINST(cvt,  si2ss)
#define cvtps2pi  SINST(cvt,  ps2pi)
#define cvtss2si  SINST(cvt,  ss2si)
#define cvttps2pi SINST(cvtt, ps2pi)
#define cvttss2si SINST(cvtt, ss2si)

// SSE integer ops
#define pmulhuw  INST(pmulhuw)
#define pavgb    INST(pavgb)
#define pavgw    INST(pavgw)
#define psadbw   INST(psadbw)
#define pmaxub   INST(pmaxub)
#define pmaxsw   INST(pmaxsw)
#define pminub   INST(pminub)
#define pminsw   INST(pminsw)
#define pextrw   INST(pextrw)
#define pinsrw   INST(pinsrw)
#define pmovmskb INST(pmovmskb)
#define maskmovq INST(maskmovq)
#define movntq   INST(movntq)
#define sfence   INST(sfence)
#define pshufw   INST(pshufw)

// SSE logical ops
#define andps  SINST(and,  ps)
#define andnps SINST(andn, ps)
#define orps   SINST(or,   ps)
#define xorps  SINST(xor,  ps)

// SSE comparison ops
#define comieq     CINST(comieq)
#define comineq    CINST(comineq)
#define comilt     CINST(comilt)
#define comile     CINST(comile)
#define comigt     CINST(comigt)
#define comige     CINST(comige)
#define ucomieq    CINST(ucomieq)
#define ucomineq   CINST(ucomineq)
#define ucomilt    CINST(ucomilt)
#define ucomile    CINST(ucomile)
#define ucomigt    CINST(ucomigt)
#define ucomige    CINST(ucomige)
#define cmpeqps    SINST(cmpeq,    ps)
#define cmpltps    SINST(cmplt,    ps)
#define cmpleps    SINST(cmple,    ps)
#define cmpgtps    SINST(cmpgt,    ps)
#define cmpgeps    SINST(cmpge,    ps)
#define cmpunordps SINST(cmpunord, ps)
#define cmpneqps   SINST(cmpneq,   ps)
#define cmpnltps   SINST(cmpnlt,   ps)
#define cmpnleps   SINST(cmpnle,   ps)
#define cmpngtps   SINST(cmpngt,   ps)
#define cmpngeps   SINST(cmpnge,   ps)
#define cmpordps   SINST(cmpord,   ps)
#define cmpeqss    SINST(cmpeq,    ss)
#define cmpltss    SINST(cmplt,    ss)
#define cmpless    SINST(cmple,    ss)
#define cmpunordss SINST(cmpunord, ss)
#define cmpneqss   SINST(cmpneq,   ss)
#define cmpnltss   SINST(cmpnlt,   ss)
#define cmpnless   SINST(cmpnle,   ss)
#define cmpordss   SINST(cmpord,   ss)

// Load and set ops
#define loadss    SINST(load,    ss)
#define setss     SINST(set,     ss)
#define loadaps   SINST(load,    ps)
#define loadups   SINST(loadu,   ps)
#define setps     SINST(set,     ps)
#define setzerops SINST(setzero, ps)

#endif // _CPU_H_
