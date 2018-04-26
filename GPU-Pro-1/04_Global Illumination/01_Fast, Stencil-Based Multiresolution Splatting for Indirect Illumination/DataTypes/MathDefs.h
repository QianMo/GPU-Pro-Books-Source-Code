/******************************************************************/
/* MathDefs.h                                                     */
/* -----------------------                                        */
/*                                                                */
/* There are a lot of mathematical constants and macros I take    */
/*    for granted that simply aren't guaranteed to be defined.    */
/*    This file checks if they are defined.  If not, it defines   */
/*    them.                                                       */
/*                                                                */
/* Chris Wyman (01/30/2008)                                       */
/******************************************************************/



#ifndef __MATHDEFS_H__
#define __MATHDEFS_H__

// Maximum values for floats, doubles, and ints
#ifndef  MAXFLOAT
#define  MAXFLOAT		((float)3.40282347e+38) 
#endif
#ifndef MAXDOUBLE
#define MAXDOUBLE		1.7976931348623157e+308
#endif
#ifndef MAXINTEGER
#define MAXINTEGER		((int)(~(unsigned int)0 >> 1))
#endif
#ifndef MININTEGER
#define MININTEGER		((int)(~(unsigned int)0))
#endif


// Common multiple of PI.  This undefines any
//   existing ones to make sure I have consistant
//   behavior across all platforms, though these
//   were taken from a Linux <math.h> header.
#undef M_PI
#define M_PI         3.14159265358979323846f  
#undef M_1_PI
#define M_1_PI       0.31830988618379067154f
#undef M_PI_4
#define M_PI_4       0.78539816339744830962f
#undef M_PI_2
#define M_PI_2       1.57079632679489661923f
#undef M_2_PI
#define M_2_PI       0.63661977236758134308f
#define M_2PI        (2.0f*M_PI)
#define M_4PI        (4.0f*M_PI)
#define M_1_2PI      (0.5f*M_1_PI)
#define M_1_4PI      (0.25f*M_1_PI)


// Some min and max macros
#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b)) 
#endif
#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b)) 
#endif
#ifndef ABS
#define ABS(a)          ((a) >= 0 ? (a) : -(a))
#endif


#define MAXLINELENGTH       512

#endif

