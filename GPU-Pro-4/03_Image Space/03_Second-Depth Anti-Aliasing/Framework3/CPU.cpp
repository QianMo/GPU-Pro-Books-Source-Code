
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

#include "CPU.h"

int cpuCount;
int cpuFamily;
char cpuVendor[13];
char cpuBrandName[49];
bool cpuCMOV, cpu3DNow, cpu3DNowExt, cpuMMX, cpuMMXExt, cpuSSE, cpuSSE2;


#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include "Windows.h"

#ifndef _WIN64
uint64 getHz(){
	LARGE_INTEGER t1,t2,tf;
	uint64 c1,c2;

	QueryPerformanceFrequency(&tf);
	QueryPerformanceCounter(&t1);
	c1 = getCycleNumber();

	// Some spin-wait
	for (volatile int i = 0; i < 1000000; i++);

	QueryPerformanceCounter(&t2);
	c2 = getCycleNumber();

	return ((c2 - c1) * tf.QuadPart / (t2.QuadPart - t1.QuadPart));
}

void cpuidAsm(uint32 func, uint32 *a, uint32 *b, uint32 *c, uint32 *d){
	__asm {
		PUSH	EAX
		PUSH	EBX
		PUSH	ECX
		PUSH	EDX

		MOV		EAX, func
		CPUID
		MOV		EDI, a
		MOV		[EDI], EAX
		MOV		EDI, b
		MOV		[EDI], EBX
		MOV		EDI, c
		MOV		[EDI], ECX
		MOV		EDI, d
		MOV		[EDI], EDX

		POP		EDX
		POP		ECX
		POP		EBX
		POP		EAX
	}
}
#endif

#else

#include <unistd.h>
#include <sys/time.h>

uint64 getHz(){
	static struct timeval t1, t2;
	static struct timezone tz;
	unsigned long long c1,c2;

	gettimeofday(&t1, &tz);
	c1 = getCycleNumber();

	// Some spin-wait
	for (volatile int i = 0; i < 1000000; i++);

	gettimeofday(&t2, &tz);
	c2 = getCycleNumber();

	return (1000000 * (c2 - c1)) / ((t2.tv_usec - t1.tv_usec) + 1000000 * (t2.tv_sec - t1.tv_sec));
}

#if defined(__APPLE__)
void cpuidAsm(uint32 op, uint32 reg[4]){
	asm volatile(
		"pushl %%ebx      \n\t"
		"cpuid            \n\t"
		"movl %%ebx, %1   \n\t"
		"popl %%ebx       \n\t"
		: "=a"(reg[0]), "=r"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
		: "a"(op)
		: "cc"
	);
}
#endif

#endif

void initCPU(){
#if defined(_WIN32)
	SYSTEM_INFO sysInfo;
	GetSystemInfo(&sysInfo);
	cpuCount = sysInfo.dwNumberOfProcessors;

#elif defined(LINUX)
	//cpuCount = sysconf(_SC_NPROCESSORS_CONF);
	cpuCount = sysconf(_SC_NPROCESSORS_ONLN);

#elif defined(__APPLE__)

	// TODO: Fix ...
	cpuCount = 1;

#endif
	// Just in case ...
	if (cpuCount < 1) cpuCount = 1;

	uint32 maxi, maxei, a, b, c, d;

	cpuVendor[12]   = '\0';
	cpuBrandName[0] = '\0';

	cpuCMOV     = false;
	cpuMMX      = false;
	cpuSSE      = false;
	cpuSSE2     = false;
	cpu3DNow    = false;
	cpu3DNowExt = false;
	cpuMMXExt   = false;
	cpuFamily   = 0;

	cpuid(0, maxi, ((uint32 *) cpuVendor)[0], ((uint32 *) cpuVendor)[2], ((uint32 *) cpuVendor)[1]);

	if (maxi >= 1){
		cpuid(1, a, b, c, d);
		cpuCMOV = (d & 0x00008000) != 0;
		cpuMMX  = (d & 0x00800000) != 0;
		cpuSSE  = (d & 0x02000000) != 0;
		cpuSSE2 = (d & 0x04000000) != 0;
		cpuFamily = (a >> 8) & 0x0F;

		cpuid(0x80000000, maxei, b, c, d);
		if (maxei >= 0x80000001){
			cpuid(0x80000001, a, b, c, d);
			cpu3DNow    = (d & 0x80000000) != 0;
			cpu3DNowExt = (d & 0x40000000) != 0;
			cpuMMXExt   = (d & 0x00400000) != 0;

			if (maxei >= 0x80000002){
				cpuid(0x80000002, ((uint32 *) cpuBrandName)[0], ((uint32 *) cpuBrandName)[1], ((uint32 *) cpuBrandName)[2], ((uint32 *) cpuBrandName)[3]);
				cpuBrandName[16] = '\0';

				if (maxei >= 0x80000003){
					cpuid(0x80000003, ((uint32 *) cpuBrandName)[4], ((uint32 *) cpuBrandName)[5], ((uint32 *) cpuBrandName)[6], ((uint32 *) cpuBrandName)[7]);
					cpuBrandName[32] = '\0';

					if (maxei >= 0x80000004){
						cpuid(0x80000004, ((uint32 *) cpuBrandName)[8], ((uint32 *) cpuBrandName)[9], ((uint32 *) cpuBrandName)[10], ((uint32 *) cpuBrandName)[11]);
						cpuBrandName[48] = '\0';
					}
				}
			}
		}
	}
}
