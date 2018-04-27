/******************************************************************************

 @File         PVRTFixedPoint.h

 @Title        PVRTFixedPoint

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independant

 @Description  Set of macros and functions to make fixed-point easier to program.

******************************************************************************/
#ifndef _PVRTFIXEDPOINT_H_
#define _PVRTFIXEDPOINT_H_

#if defined(BUILD_OGLES) || defined(BUILD_D3DM)
	#include "PVRTFixedPointAPI.h"
#else
	#define VERTTYPE float
	#ifdef PVRT_FIXED_POINT_ENABLE
		#error Build option not supported: PVRT_FIXED_POINT_ENABLE
	#endif
#endif

/* Define a 64-bit type for various platforms */
#if defined(__int64) || defined(WIN32)
#define PVR64BIT __int64
#elif defined(TInt64)
#define PVR64BIT TInt64
#else
#define PVR64BIT long long
#endif

/* Fixed-point macros */
#define PVRTF2X(f)		( (int) ( (f)*(65536) ) )
#define PVRTX2F(x)		((float)(x)/65536.0f)
#define PVRTXMUL(a,b)	( (int)( ((PVR64BIT)(a)*(b)) / 65536 ) )
#define PVRTXDIV(a,b)	( (int)( (((PVR64BIT)(a))<<16)/(b) ) )
#define PVRTABS(a)		((a) <= 0 ? -(a) : (a) )

/* Define trig table macros */
#include "PVRTMathTable.h"

/* Useful values */
#define PVRT_PI_OVER_TWOf	(3.1415926535f / 2.0f)
#define PVRT_PIf			(3.1415926535f)
#define PVRT_TWO_PIf		(3.1415926535f * 2.0f)
#define PVRT_ONEf		(1.0f)

#define PVRT_PI_OVER_TWOx	PVRTF2X(PVRT_PI_OVER_TWOf)
#define PVRT_PIx			PVRTF2X(PVRT_PIf)
#define PVRT_TWO_PIx		PVRTF2X(PVRT_TWO_PIf)
#define PVRT_ONEx			PVRTF2X(PVRT_ONEf)

/* Fixed-point trig function lookups */
#define PVRTXCOS(x)		(cos_val[(PVRTXMUL(((PVRTXDIV((x)<0? -(x):(x), PVRT_TWO_PIx)) & 0x0000FFFF), (NUM_ENTRIES-1)))])
#define PVRTXSIN(x)		(sin_val[(PVRTXMUL(((PVRTXDIV((x)<0 ? PVRT_PIx-(x):(x), PVRT_TWO_PIx)) & 0x0000FFFF), (NUM_ENTRIES-1)))])
#define PVRTXTAN(x)		( (x)<0 ? -tan_val[(PVRTXMUL(((PVRTXDIV(-(x), PVRT_TWO_PIx)) & 0x0000FFFF), (NUM_ENTRIES-1)))] : tan_val[(PVRTXMUL(((PVRTXDIV(x, PVRT_TWO_PIx)) & 0x0000FFFF), (NUM_ENTRIES-1)))] )
#define PVRTXACOS(x)	(acos_val[PVRTXMUL(((((x) + PVRTF2X(1.0f))>>1) & 0x0000FFFF), (NUM_ENTRIES-1))])

/* Floating-point trig functions lookups (needed by some tools chains that have problems with real math functions) */
#ifdef USE_TRIGONOMETRIC_LOOKUP_TABLES

	/* If trig tables are forced ON in non-fixed-point builds then convert fixed-point trig tables results to float */
	#define	PVRTFCOS(x)				PVRTX2F(PVRTXCOS(PVRTF2X(x)))
	#define	PVRTFSIN(x)				PVRTX2F(PVRTXSIN(PVRTF2X(x)))
	#define	PVRTFTAN(x)				PVRTX2F(PVRTXTAN(PVRTF2X(x)))
	#define	PVRTFACOS(x)			PVRTX2F(PVRTXACOS(PVRTF2X(x)))

#else

	/* Trig abstraction macros default to normal math trig functions for full float mode */
	#define	PVRTFCOS(x)				((float)cos(x))
	#define	PVRTFSIN(x)				((float)sin(x))
	#define	PVRTFTAN(x)				((float)tan(x))
	#define	PVRTFACOS(x)			((float)acos(x))

#endif


/* Fixed/float macro abstraction */
#ifdef PVRT_FIXED_POINT_ENABLE

	/* Fixed-point operations, including trig tables */
	#define VERTTYPEMUL(a,b)			PVRTXMUL(a,b)
	#define VERTTYPEDIV(a,b)			PVRTXDIV(a,b)
	#define VERTTYPEABS(a)				PVRTABS(a)

	#define f2vt(f) 					PVRTF2X(f)
	#define vt2f(x) 					PVRTX2F(x)

	#define PVRT_PI_OVER_TWO			PVRT_PI_OVER_TWOx
	#define PVRT_PI						PVRT_PIx
	#define PVRT_TWO_PI					PVRT_TWO_PIx
	#define PVRT_ONE					PVRT_ONEx

	#define	PVRTCOS(x)					PVRTXCOS(x)
	#define	PVRTSIN(x)					PVRTXSIN(x)
	#define	PVRTTAN(x)					PVRTXTAN(x)
	#define	PVRTACOS(x)					PVRTXACOS(x)

#else

	/* Floating-point operations */
	#define VERTTYPEMUL(a,b)			( (VERTTYPE)((a)*(b)) )
	#define VERTTYPEDIV(a,b)			( (VERTTYPE)((a)/(b)) )
	#define VERTTYPEABS(a)				( (VERTTYPE)(fabs(a)) )

	#define f2vt(x)						(x)
	#define vt2f(x)						(x)

	#define PVRT_PI_OVER_TWO			PVRT_PI_OVER_TWOf
	#define PVRT_PI						PVRT_PIf
	#define PVRT_TWO_PI					PVRT_TWO_PIf
	#define PVRT_ONE					PVRT_ONEf

	/* If trig tables are forced ON in non-fixed-point builds then convert fixed-point trig tables results to float */
	#define	PVRTCOS(x)					PVRTFCOS(x)
	#define	PVRTSIN(x)					PVRTFSIN(x)
	#define	PVRTTAN(x)					PVRTFTAN(x)
	#define	PVRTACOS(x)					PVRTFACOS(x)

#endif


// Structure Definitions

/*!***************************************************************************
 @Struct HeaderStruct_Mesh
 @Brief  Defines the format of a header-object as exported by the MAX plugin.
*****************************************************************************/
typedef struct {
	unsigned int      nNumVertex;
    unsigned int      nNumFaces;
    unsigned int      nNumStrips;
    unsigned int      nFlags;
    unsigned int      nMaterial;
    float             fCenter[3];
    float             *pVertex;
    float             *pUV;
    float             *pNormals;
    float             *pPackedVertex;
    unsigned int      *pVertexColor;
    unsigned int      *pVertexMaterial;
    unsigned short    *pFaces;
    unsigned short    *pStrips;
    unsigned short    *pStripLength;
    struct
    {
        unsigned int  nType;
        unsigned int  nNumPatches;
        unsigned int  nNumVertices;
        unsigned int  nNumSubdivisions;
        float         *pControlPoints;
        float         *pUVs;
    } Patch;
}   HeaderStruct_Mesh;


#ifdef PVRT_FIXED_POINT_ENABLE

/*!***************************************************************************
 Defines the format of a header-object as when converted to
 fixed point.
*****************************************************************************/
/*!***************************************************************************
 @Struct HeaderStruct_Fixed_Mesh
 @Brief  Defines the format of a header-object as when converted to fixed point.
*****************************************************************************/
typedef struct {
	unsigned int      nNumVertex;
	unsigned int      nNumFaces;
	unsigned int      nNumStrips;
	unsigned int      nFlags;
	unsigned int      nMaterial;
	VERTTYPE          fCenter[3];
	VERTTYPE          *pVertex;
	VERTTYPE          *pUV;
	VERTTYPE          *pNormals;
	VERTTYPE          *pPackedVertex;
	unsigned int      *pVertexColor;
	unsigned int      *pVertexMaterial;
	unsigned short    *pFaces;
	unsigned short    *pStrips;
	unsigned short    *pStripLength;
	struct
	{
		unsigned int  nType;				// for the moment, these are left as floats
		unsigned int  nNumPatches;
		unsigned int  nNumVertices;
		unsigned int  nNumSubdivisions;
		float       *pControlPoints;
		float       *pUVs;
	} Patch;
}   HeaderStruct_Fixed_Mesh;

	typedef HeaderStruct_Fixed_Mesh HeaderStruct_Mesh_Type;
#else
	typedef HeaderStruct_Mesh HeaderStruct_Mesh_Type;
#endif

// Function prototypes

/*!***************************************************************************
 @Function		PVRTLoadHeaderObject
 @Input			headerObj			Pointer to object structure in the header file
 @Return		directly usable geometry in fixed or float format as appropriate
 @Description	Converts the data exported by MAX to fixed point when used in OpenGL
				ES common-lite profile.
*****************************************************************************/
HeaderStruct_Mesh_Type* PVRTLoadHeaderObject(const void *headerObj);

/*!***************************************************************************
 @Function		PVRTUnloadHeaderObject
 @Input			headerObj			Pointer returned by LoadHeaderObject
 @Description	Releases memory allocated by LoadHeaderObject when geometry no longer
				needed.
*****************************************************************************/
void PVRTUnloadHeaderObject(HeaderStruct_Mesh_Type* headerObj);


#endif /* _PVRTFIXEDPOINT_H_ */

/*****************************************************************************
 End of file (PVRTFixedPoint.h)
*****************************************************************************/
