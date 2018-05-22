#pragma once

#include <stdlib.h>

#define TRUE	1
#define FALSE	0

#ifndef INTEL_COMPILER
//#define INTEL_COMPILER
#endif

#ifdef DBL
	#undef DBL
#endif
#define DBL float

#define RAND_LENGTH 4096

#define PI       3.14159265358979323846f /* F for SP float */
#define PI_D     3.14159265358979323846  /* slower DP for more precision */

#define DEG2RAD(_x)   ((_x) * 0.017453292519943295f)	// ((_x) * PI  /180.0f)
#define DEG2RAD_D(_x) ((_x) * 0.017453292519943295)		// ((_x) * PI_D/180.0)

#define RAD2DEG(_x)   ((_x) * 57.295779513082323f)		// ((_x) * 180.0f/PI)
#define RAD2DEG_D(_x) ((_x) * 57.295779513082323)		// ((_x) * 180.0 /PI_D)

////////////////////////////////////////////////////////////////////////////////

#ifdef WIN32
#define DIR_DELIMITER   '\\'

#define STR_DUP(_a)            ( _strdup (_a))
#define STR_EQUAL(_a,_b)       (!_stricmp (_a,_b))
#define SUBSTR_EQUAL(_a,_b)    (!_strnicmp (_a,_b, strlen (_b)))
#else
#define DIR_DELIMITER   '/'

#define STR_DUP(_a)            ( strdup (_a))
#define STR_EQUAL(_a,_b)       (!strcasecmp (_a,_b))
#define SUBSTR_EQUAL(_a,_b)    (!strncasecmp (_a,_b, strlen (_b)))
#endif

#define MAXSTRING       256

////////////////////////////////////////////////////////////////////////////////

#define SAFEFREE(_x)        { if ((_x) != NULL) { free    ((_x)); _x = NULL; } }
#define SAFEDELETE(_x)      { if ((_x) != NULL) { delete   (_x);  _x = NULL; } }
#define SAFEDELETEARRAY(_x) { if ((_x) != NULL) { delete[] (_x);  _x = NULL; } }

////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#ifdef EAZD_DEBUG
    #define EAZD_ASSERTALWAYS(_x)   \
        if (! (_x))                 \
        {                           \
            std::cerr << "Assertion failed @ " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit (EXIT_FAILURE);    \
        }

    #ifdef WIN32
        #define EAZD_TRACE(_x) \
            std::cerr << "[" << __FUNCSIG__ << " @ " << __FILE__ << \
                ":" << __LINE__ << "]" << "  " << _x << std::endl
    #else
        #define EAZD_TRACE(_x) \
            std::cerr << "[" << __PRETTY_FUNCTION__ << " @ " << __FILE__ << \
                ":" << __LINE__ << "]" << "  " << _x << std::endl
    #endif
#else
	#define EAZD_ASSERTALWAYS(_x) {}
#endif

#define EAZD_PRINT(_x) std::cout << _x << std::endl
#define EAZD_TRACE(_x) std::cerr << "[" << __FUNCSIG__ << " @ " << __FILE__ << ":" << __LINE__ << "]" << "  " << _x << std::endl

////////////////////////////////////////////////////////////////////////////////

#define ZERO_TOLERANCE_12    1.0e-012f
#define ZERO_TOLERANCE_9     1.0e-09f
#define ZERO_TOLERANCE_8     1.0e-08f
#define ZERO_TOLERANCE_7     1.0e-07f
#define ZERO_TOLERANCE_6     1.0e-06f
#define ZERO_TOLERANCE_5     1.0e-05f
#define ZERO_TOLERANCE_4     1.0e-04f
#define ZERO_TOLERANCE_3     1.0e-03f
#define ZERO_TOLERANCE_2     1.0e-02f
#define ZERO_TOLERANCE_1     1.0e-01f
#define ZERO_TOLERANCE       ZERO_TOLERANCE_6

#define NEAR_ZERO(_s)        (ABS ((_s)) < ZERO_TOLERANCE_6)
#define NEAR_ZERO_TOL(_s, _tol) (ABS ((_s)) < (_tol))

////////////////////////////////////////////////////////////////////////////////

#define SRAND(_seed)     srand (_seed)
#define RAND()           rand ()
#define RAND_0TO1()      ((float) RAND () / (float) RAND_MAX)
#define RAND_0TO1_D()    ((double) RAND () / (double) RAND_MAX)
// return a rand() value in [_start, _end)
#define RAND_I(_start, _end)  ((_start) + (int) (((_end) - (_start)) * RAND_0TO1 ()))

////////////////////////////////////////////////////////////////////////////////

#define SQUARE(_x) ((_x)*(_x))
#define MIN2(a,b) ((a) < (b) ? (a) : (b))
#define MAX2(a,b) ((a) > (b) ? (a) : (b))
#define MIN3(a,b,c) ((a) < (b) ? MIN2(a,c) : MIN2(b,c))
#define MAX3(a,b,c) ((a) > (b) ? MAX2(a,c) : MAX2(b,c))
#define MIN4(a,b,c,d) ((a) < (b) ? MIN3(a,c,d) : MIN3(b,c,d))
#define MAX4(a,b,c,d) ((a) > (b) ? MAX3(a,c,d) : MAX3(b,c,d))

////////////////////////////////////////////////////////////////////////////////

#define CLAMP(_x, _lo, _hi) \
    (((_x) < (_lo)) ? (_lo) : (_x) > (_hi) ? (_hi) : (_x))

#define BETWEEN(_x, _lo, _hi) ((_x) >= (_lo) && ((_x) <= (_hi)))

////////////////////////////////////////////////////////////////////////////////

/*
 *  ABS is faster than calling fabsf
 *  ABSLT etc are faster than (ABS(x1) < x2)
 * 
 */
#define ABS(_x1)       ((_x1 < 0) ? -(_x1) : (_x1))
#define ABSLT(_x1,_x2) ((_x1) <  (_x2) && -(_x1) <  (_x2))
#define ABSGT(_x1,_x2) ((_x1) >  (_x2) || -(_x1) >  (_x2))
#define ABSLE(_x1,_x2) ((_x1) <= (_x2) && -(_x1) <= (_x2))
#define ABSGE(_x1,_x2) ((_x1) >= (_x2) || -(_x1) >= (_x2))

////////////////////////////////////////////////////////////////////////////////

#define ROUND_I(_x)       ((int) ((_x) + 0.5))
#define ROUND_UC(_x)      ((unsigned char) ((_x) + 0.5))

////////////////////////////////////////////////////////////////////////////////

#define ONE_OVER_127_5			0.00784313771873712540 // float value
#define ONE_OVER_255_0          0.00392156885936856270 // float value
#define ONE_OVER_32767_5		0.00003051804378628731 // float value
#define ONE_OVER_65535			0.00001525902189314365 // float value
#define ONE_OVER_4294967295		0.00000000023283064365 // float value

#define PACKINTOUBYTE_0TO1(_x)		((unsigned char) ((_x)*255.0))
// #define UNPACKUBYTE_0TO1(_x)		((float) (_x)/255.0f)
#define UNPACKUBYTE_0TO1(_x)            ((float) (_x)*ONE_OVER_255_0)

#define PACKINTOUBYTE_MINUS1TO1(_x)     ((unsigned char) ((_x)*127.5+127.5))
// #define UNPACKUBYTE_MINUS1TO1(_x)       (((float) (_x)-127.5)/127.5f)
#define UNPACKUBYTE_MINUS1TO1(_x)       (((float) (_x)-127.5)*ONE_OVER_127_5)

#define PACKINTOUSHORT_0TO1(_x)		((unsigned short) ((_x)*65535.0))
// #define UNPACKUSHORT_0TO1(_x)		((float) (_x)/65535.0f)
#define UNPACKUSHORT_0TO1(_x)		((float) (_x)*ONE_OVER_65535)

#define PACKINTOUSHORT_MINUS1TO1(_x)    ((unsigned short) ((_x)*32767.5+32767.5))
// #define UNPACKUSHORT_MINUS1TO1(_x)      (((float) (_x)-32767.5)/32767.5f)
#define UNPACKUSHORT_MINUS1TO1(_x)      (((float) (_x)-32767.5)*ONE_OVER_32767_5)

#define PACKINTOSHORT_0TO1(_x)		((short) ((_x)*65535-32767.5))
// #define UNPACKSHORT_0TO1(_x)		((float) ((_x)+32767.5)/65535.0f)
#define UNPACKSHORT_0TO1(_x)		((float) ((_x)+32767.5)*ONE_OVER_65535)

#define PACKINTOSHORT_MINUS1TO1(_x)	((short) ((_x)*32767.5+32767.5))
// #define UNPACKSHORT_MINUS1TO1(_x)	(((float) (_x)-32767.5)/32767.5f)
#define UNPACKSHORT_MINUS1TO1(_x)	(((float) (_x)-32767.5)*ONE_OVER_32767_5)

#define PACKINTOSHORT_SMINUS1TO1(_x)	((short) ((_x)*32767.5))
// #define UNPACKSHORT_SMINUS1TO1(_x)	(((float) (_x))/32767.5f)
#define UNPACKSHORT_SMINUS1TO1(_x)	(((float) (_x))*ONE_OVER_32767_5)

#define PACKINTOUINT_0TO1(_x)		((unsigned int) ((_x)*4294967295.0))
// #define UNPACKUINT_0TO1(_x)		((float) (_x)/4294967295.0f)
#define UNPACKUINT_0TO1(_x)		((float) (_x)*ONE_OVER_4294967295)

////////////////////////////////////////////////////////////////////////////////

