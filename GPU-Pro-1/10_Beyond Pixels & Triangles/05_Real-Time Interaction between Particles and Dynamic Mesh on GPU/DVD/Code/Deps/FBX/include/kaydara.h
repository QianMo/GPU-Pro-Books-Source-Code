/*!  \file kaydara.h
 */

#ifndef _Kaydara_h
#define _Kaydara_h

/**************************************************************************************

 Copyright © 2001 - 2008 Autodesk, Inc. and/or its licensors.
 All Rights Reserved.

 The coded instructions, statements, computer programs, and/or related material 
 (collectively the "Data") in these files contain unpublished information 
 proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 Canada and United States of America federal copyright law and by international 
 treaties. 
 
 The Data may not be disclosed or distributed to third parties, in whole or in
 part, without the prior written consent of Autodesk, Inc. ("Autodesk").

 THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 
 IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 OR DAMAGE. 

**************************************************************************************/


typedef signed char		kByte;        
typedef unsigned char		kUByte;       

#ifndef FALSE
	#define FALSE   0
#endif
#ifndef TRUE
	#define TRUE    1
#endif

#include <float.h>

typedef bool			    kBool;
typedef unsigned char     kBoolean;     

typedef char				kChar;		
#define K_CHAR(x)			kChar(x)
typedef signed char		kSChar;
#define K_SCHAR(x)			kSChar(x)
typedef unsigned char		kUChar;
#define K_UCHAR(x)			kUChar(x##L)
#define K_CHAR_MIN			K_CHAR(0)
#define K_CHAR_MAX			K_CHAR(127)
#define K_SCHAR_MIN			K_SCHAR(-128)
#define K_SCHAR_MAX			K_SCHAR(127)
#define K_UCHAR_MIN			K_UCHAR(0)
#define K_UCHAR_MAX			K_UCHAR(255)
#define K_SIZEOF_CHAR		sizeof(kChar)
typedef kChar *			kCharPtr;
typedef kChar				kTChar;
typedef kChar *			kTCharPtr;

#ifndef _T
	#define _T(a) a
#endif

typedef signed short		kShort;
#define K_SHORT(x)			kShort(x)
typedef unsigned short	kUShort;
#define K_USHORT(x)			kUShort(x##U)
#define K_SHORT_MIN			K_SHORT(-32768)
#define K_SHORT_MAX			K_SHORT(32767)
#define K_USHORT_MIN		K_USHORT(0)
#define K_USHORT_MAX		K_USHORT(65535)
#define K_SIZEOF_SHORT		sizeof(kShort)

typedef signed int		kInt;
#define K_INT(x)			(x)
typedef unsigned int		kUInt;
#define K_UINT(x)			(x##U)
#define K_INT_MIN			K_INT(0x80000000)
#define K_INT_MAX			K_INT(0x7fffffff)
#define K_UINT_MIN			K_UINT(0)
#define K_UINT_MAX			K_UINT(0xffffffff)
#define K_SIZEOF_INT		sizeof(kInt)

// 32 bit Windows machines AND Intel 64 bit machines
#if defined(_WIN32) || defined(__ia64__) || defined(_M_IA64)
    typedef signed int		kLong;
    #define K_LONG(x)			(x)
    typedef unsigned int		kULong;
    #define K_ULONG(x)			(x##U)
#else
    typedef signed long		kLong;
    #define K_LONG(x)			(x##L)
    typedef unsigned long		kULong;
    #define K_ULONG(x)			(x##UL)
#endif
#define K_SIZEOF_LONG   sizeof(kLong)

#if defined(__sgi)
    #if (_MIPS_SIM == _ABI64)
            #define KLONG_ABI64 1
    #endif
#endif    

#if defined(KLONG_ABI64)
    #define K_LONG_MIN			K_LONG(-0x8000000000000000)
    #define K_LONG_MAX			K_LONG(0x7fffffffffffffff)
    #define K_ULONG_MIN			K_ULONG(0)
    #define K_ULONG_MAX			K_ULONG(0xffffffffffffffff)
#else
    #define K_LONG_MIN			K_LONG(0x80000000)
    #define K_LONG_MAX			K_LONG(0x7fffffff)
    #define K_ULONG_MIN			K_ULONG(0)
    #define K_ULONG_MAX			K_ULONG(0xffffffff)
#endif

#ifdef _MSC_VER
    typedef signed __int64		kLongLong;
    #define K_LONGLONG(x)		(x##i64)
    typedef unsigned __int64	kULongLong;
    #define K_ULONGLONG(x)		(x##Ui64)
#else
    typedef signed long long	kLongLong;
    #define K_LONGLONG(x)		(x##LL)
    typedef unsigned long long	kULongLong;
    #define K_ULONGLONG(x)		(x##ULL)
#endif

#define K_LONGLONG_MIN		K_LONGLONG(0x8000000000000000)
#define K_LONGLONG_MAX		K_LONGLONG(0x7fffffffffffffff)
#define K_ULONGLONG_MIN		K_ULONGLONG(0)
#define K_ULONGLONG_MAX		K_ULONGLONG(0xffffffffffffffff)
#define K_SIZEOF_LONGLONG	sizeof(kLongLong)

typedef float				kFloat;
#define K_FLOAT(x)			(x##F)
#define K_FLOAT_MIN			FLT_MIN	
#define K_FLOAT_MAX			FLT_MAX	
#define K_SIZEOF_FLOAT		sizeof(kFloat)

typedef double				kDouble;
#define K_DOUBLE(x)			(x)
#define K_DOUBLE_MIN		DBL_MIN	
#define K_DOUBLE_MAX		DBL_MAX	
#define K_SIZEOF_DOUBLE		sizeof(kDouble)

typedef kSChar				kInt8;
typedef kUChar				kUInt8;
#define K_INT8_MIN			K_SCHAR_MIN
#define K_INT8_MAX			K_SCHAR_MAX
#define K_UINT8_MIN			K_UCHAR_MIN
#define K_UINT8_MAX			K_UCHAR_MAX
#define K_SIZEOF_INT8		sizeof(kInt8)

typedef kShort				kInt16;
typedef kUShort			kUInt16;
#define K_INT16_MIN			K_SHORT_MIN
#define K_INT16_MAX			K_SHORT_MAX
#define K_UINT16_MIN		K_USHORT_MIN
#define K_UINT16_MAX		K_USHORT_MAX
#define K_SIZEOF_INT16		sizeof(kInt16)

typedef kInt				kInt32;
typedef kUInt				kUInt32;
#define K_INT32_MIN			K_INT_MIN
#define K_INT32_MAX			K_INT_MAX
#define K_UINT32_MIN		K_UINT_MIN
#define K_UINT32_MAX		K_UINT_MAX
#define K_SIZEOF_INT32		sizeof(kInt32)

typedef kLongLong			kInt64;
typedef kULongLong			kUInt64;
#define K_INT64_MIN			K_LONGLONG_MIN
#define K_INT64_MAX			K_LONGLONG_MAX
#define K_UINT64_MIN		K_ULONGLONG_MIN
#define K_UINT64_MAX		K_ULONGLONG_MAX
#define K_SIZEOF_INT64		sizeof(kInt64)

typedef kInt32				kInteger;
typedef kUInt32			kUInteger;
#define K_INTEGER_MIN		K_INT32_MIN
#define K_INTEGER_MAX		K_INT32_MAX
#define K_UINTEGER_MIN		K_UINT32_MIN
#define K_UINTEGER_MAX		K_UINT32_MAX
#define K_SIZEOF_INTEGER	sizeof(kInteger)

typedef kInt				kEnum;
typedef void *				kPointer;

#define K_SIZEOF_POINTER	sizeof(kPointer)
#define K_SHIFT_POINTER		(K_SIZEOF_POINTER >> 1)

#define K_NULL				0

// 64 bits architectures    
#if defined(__x86_64) || defined(_M_AMD64) || defined(__ia64__) || defined(_M_IA64)
    typedef kULongLong  kOffset;
#elif defined(__sgi)
    typedef kULong      kOffset;
#else    
    typedef kUInt32     kOffset;
#endif    

#define K_SIZEOF_OFFSET		sizeof(kOffset)
#define K_SHIFT_OFFSET		(K_SIZEOF_OFFSET >> 1)


typedef kOffset				kReference;

typedef enum { K_SUCCESS = 0 , K_ERROR , K_NOTIMPL } kResult;

#if defined(linux) || defined(__sgi)

        #ifndef strcmpi
                #define strcmpi strcasecmp
        #endif
        #ifndef stricmp
                #define stricmp strcasecmp
        #endif
        #ifndef strncmpi
                #define strncmpi strncasecmp
        #endif
        #ifndef strnicmp
                #define strnicmp strncasecmp
        #endif
#endif

// Deprecated macro, to remind people to stop using certain functions

#if defined(__GNUG__)       // GNU compiler
    #define K_DEPRECATED __attribute__((deprecated))
#elif defined(__INTEL_COMPILER) // Intel compiler
    #if (__INTEL_COMPILER >= 810) // Maybe we could use an earlier version, we should check this some day
        #define K_DEPRECATED __declspec(deprecated)
    #else
        #define K_DEPRECATED
    #endif
#elif defined(_MSC_VER)    // Microsoft compiler
    #if (_MSC_VER >= 1300) // 7.1
        #define K_DEPRECATED __declspec(deprecated)
    #else
        #define K_DEPRECATED
    #endif
#elif defined(__sgi)      // SGI compiler
    #define K_DEPRECATED
#elif defined(__MWERKS__) // Metrowerks compiler
    #define K_DEPRECATED
#else // Unknown compiler
    #define K_DEPRECATED
#endif


// Compiler specific
#ifdef _MSC_VER
    #pragma warning(disable: 4514) //unreferenced inline function has been removed (Level 4) optimizer  
    #pragma warning(disable: 4100) // warning C4100: '?' : unreferenced formal parameter
    #pragma warning(disable: 4710) // function not inlined
    
    /*For 4251:
    In fact, the documentation explicitly states this in the page titled "Using dllimport and dllexport in C++ Classes" :
    As a rule, everything that is accessible to the DLL's client (according to C++ access rules) should be part of the exportable interface. This includes private data members referenced in inline functions
    This is the problem. If you have a template SomeTemplate<T>, it is not marked with dllexport because you can only export definitions, not declarations, and a template is not a definition.
    */
    #pragma warning( disable : 4251 )
#endif

#define K_FORWARD( ClassName ) class ClassName; typedef ClassName *H##ClassName;

#endif // #ifndef _Kaydara_h




