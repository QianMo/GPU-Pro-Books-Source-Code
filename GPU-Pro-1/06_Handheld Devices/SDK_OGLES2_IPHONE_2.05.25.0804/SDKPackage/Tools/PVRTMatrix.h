/******************************************************************************

 @File         PVRTMatrix.h

 @Title        PVRTMatrix

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Vector and Matrix functions for floating and fixed point math. The
               general matrix format used is directly compatible with, for
               example, both DirectX and OpenGL.

******************************************************************************/
#ifndef _PVRTMATRIX_H_
#define _PVRTMATRIX_H_

#include "PVRTGlobal.h"
/****************************************************************************
** Defines
****************************************************************************/
#define MAT00 0
#define MAT01 1
#define MAT02 2
#define MAT03 3
#define MAT10 4
#define MAT11 5
#define MAT12 6
#define MAT13 7
#define MAT20 8
#define MAT21 9
#define MAT22 10
#define MAT23 11
#define MAT30 12
#define MAT31 13
#define MAT32 14
#define MAT33 15

/****************************************************************************
** Typedefs
****************************************************************************/
/*!***************************************************************************
 2D floating point vector
*****************************************************************************/
typedef struct
{
	float x;	/*!< x coordinate */
	float y;	/*!< y coordinate */
} PVRTVECTOR2f;

/*!***************************************************************************
 2D fixed point vector
*****************************************************************************/
typedef struct
{
	int x;	/*!< x coordinate */
	int y;	/*!< y coordinate */
} PVRTVECTOR2x;

/*!***************************************************************************
 3D floating point vector
*****************************************************************************/
typedef struct
{
	float x;	/*!< x coordinate */
	float y;	/*!< y coordinate */
	float z;	/*!< z coordinate */
} PVRTVECTOR3f;

/*!***************************************************************************
 3D fixed point vector
*****************************************************************************/
typedef struct
{
	int x;	/*!< x coordinate */
	int y;	/*!< y coordinate */
	int z;	/*!< z coordinate */
} PVRTVECTOR3x;

/*!***************************************************************************
 4D floating point vector
*****************************************************************************/
typedef struct
{
	float x;	/*!< x coordinate */
	float y;	/*!< y coordinate */
	float z;	/*!< z coordinate */
	float w;	/*!< w coordinate */
} PVRTVECTOR4f;

/*!***************************************************************************
 4D fixed point vector
*****************************************************************************/
typedef struct
{
	int x;	/*!< x coordinate */
	int y;	/*!< y coordinate */
	int z;	/*!< z coordinate */
	int w;	/*!< w coordinate */
} PVRTVECTOR4x;

/*!***************************************************************************
 4x4 floating point matrix
*****************************************************************************/

class PVRTMATRIXf
{
public:
    float* operator [] ( const int Row )
	{
		return &f[Row<<2];
	}
	float f[16];	/*!< Array of float */
};

/*!***************************************************************************
 4x4 fixed point matrix
*****************************************************************************/
class PVRTMATRIXx
{
public:
    int* operator [] ( const int Row )
	{
		return &f[Row<<2];
	}
	int f[16];
};

/*!***************************************************************************
 3x3 floating point matrix
*****************************************************************************/

class PVRTMATRIX3f
{
public:
    float* operator [] ( const int Row )
	{
		return &f[Row*3];
	}
	float f[9];	/*!< Array of float */
};

/*!***************************************************************************
 3x3 fixed point matrix
*****************************************************************************/
class PVRTMATRIX3x
{
public:
    int* operator [] ( const int Row )
	{
		return &f[Row*3];
	}
	int f[9];
};


/****************************************************************************
** Float or fixed
****************************************************************************/
#ifdef PVRT_FIXED_POINT_ENABLE
typedef PVRTVECTOR2x		PVRTVECTOR2;
typedef PVRTVECTOR3x		PVRTVECTOR3;
typedef PVRTVECTOR4x		PVRTVECTOR4;
typedef PVRTMATRIX3x		PVRTMATRIX3;
typedef PVRTMATRIXx			PVRTMATRIX;
#define PVRTMatrixIdentity					PVRTMatrixIdentityX
#define PVRTMatrixMultiply					PVRTMatrixMultiplyX
#define PVRTMatrixTranslation				PVRTMatrixTranslationX
#define PVRTMatrixScaling					PVRTMatrixScalingX
#define PVRTMatrixRotationX					PVRTMatrixRotationXX
#define PVRTMatrixRotationY					PVRTMatrixRotationYX
#define PVRTMatrixRotationZ					PVRTMatrixRotationZX
#define PVRTMatrixTranspose					PVRTMatrixTransposeX
#define PVRTMatrixInverse					PVRTMatrixInverseX
#define PVRTMatrixInverseEx					PVRTMatrixInverseExX
#define PVRTMatrixLookAtLH					PVRTMatrixLookAtLHX
#define PVRTMatrixLookAtRH					PVRTMatrixLookAtRHX
#define PVRTMatrixPerspectiveFovLH			PVRTMatrixPerspectiveFovLHX
#define PVRTMatrixPerspectiveFovRH			PVRTMatrixPerspectiveFovRHX
#define PVRTMatrixOrthoLH					PVRTMatrixOrthoLHX
#define PVRTMatrixOrthoRH					PVRTMatrixOrthoRHX
#define PVRTMatrixVec3Lerp					PVRTMatrixVec3LerpX
#define PVRTMatrixVec3DotProduct			PVRTMatrixVec3DotProductX
#define PVRTMatrixVec3CrossProduct			PVRTMatrixVec3CrossProductX
#define PVRTMatrixVec3Normalize				PVRTMatrixVec3NormalizeX
#define PVRTMatrixVec3Length				PVRTMatrixVec3LengthX
#define PVRTMatrixLinearEqSolve				PVRTMatrixLinearEqSolveX
#else
typedef PVRTVECTOR2f		PVRTVECTOR2;
typedef PVRTVECTOR3f		PVRTVECTOR3;
typedef PVRTVECTOR4f		PVRTVECTOR4;
typedef PVRTMATRIX3f		PVRTMATRIX3;
typedef PVRTMATRIXf			PVRTMATRIX;
#define PVRTMatrixIdentity					PVRTMatrixIdentityF
#define PVRTMatrixMultiply					PVRTMatrixMultiplyF
#define PVRTMatrixTranslation				PVRTMatrixTranslationF
#define PVRTMatrixScaling					PVRTMatrixScalingF
#define PVRTMatrixRotationX					PVRTMatrixRotationXF
#define PVRTMatrixRotationY					PVRTMatrixRotationYF
#define PVRTMatrixRotationZ					PVRTMatrixRotationZF
#define PVRTMatrixTranspose					PVRTMatrixTransposeF
#define PVRTMatrixInverse					PVRTMatrixInverseF
#define PVRTMatrixInverseEx					PVRTMatrixInverseExF
#define PVRTMatrixLookAtLH					PVRTMatrixLookAtLHF
#define PVRTMatrixLookAtRH					PVRTMatrixLookAtRHF
#define PVRTMatrixPerspectiveFovLH			PVRTMatrixPerspectiveFovLHF
#define PVRTMatrixPerspectiveFovRH			PVRTMatrixPerspectiveFovRHF
#define PVRTMatrixOrthoLH					PVRTMatrixOrthoLHF
#define PVRTMatrixOrthoRH					PVRTMatrixOrthoRHF
#define PVRTMatrixVec3Lerp					PVRTMatrixVec3LerpF
#define PVRTMatrixVec3DotProduct			PVRTMatrixVec3DotProductF
#define PVRTMatrixVec3CrossProduct			PVRTMatrixVec3CrossProductF
#define PVRTMatrixVec3Normalize				PVRTMatrixVec3NormalizeF
#define PVRTMatrixVec3Length				PVRTMatrixVec3LengthF
#define PVRTMatrixLinearEqSolve				PVRTMatrixLinearEqSolveF
#endif

/****************************************************************************
** Functions
****************************************************************************/

/*!***************************************************************************
 @Function			PVRTMatrixIdentityF
 @Output			mOut	Set to identity
 @Description		Reset matrix to identity matrix.
*****************************************************************************/
void PVRTMatrixIdentityF(PVRTMATRIXf &mOut);

/*!***************************************************************************
 @Function			PVRTMatrixIdentityX
 @Output			mOut	Set to identity
 @Description		Reset matrix to identity matrix.
*****************************************************************************/
void PVRTMatrixIdentityX(PVRTMATRIXx &mOut);

/*!***************************************************************************
 @Function			PVRTMatrixMultiplyF
 @Output			mOut	Result of mA x mB
 @Input				mA		First operand
 @Input				mB		Second operand
 @Description		Multiply mA by mB and assign the result to mOut
					(mOut = p1 * p2). A copy of the result matrix is done in
					the function because mOut can be a parameter mA or mB.
*****************************************************************************/
void PVRTMatrixMultiplyF(
	PVRTMATRIXf			&mOut,
	const PVRTMATRIXf	&mA,
	const PVRTMATRIXf	&mB);
/*!***************************************************************************
 @Function			PVRTMatrixMultiplyX
 @Output			mOut	Result of mA x mB
 @Input				mA		First operand
 @Input				mB		Second operand
 @Description		Multiply mA by mB and assign the result to mOut
					(mOut = p1 * p2). A copy of the result matrix is done in
					the function because mOut can be a parameter mA or mB.
					The fixed-point shift could be performed after adding
					all four intermediate results together however this might
					cause some overflow issues.
*****************************************************************************/
void PVRTMatrixMultiplyX(
	PVRTMATRIXx			&mOut,
	const PVRTMATRIXx	&mA,
	const PVRTMATRIXx	&mB);

/*!***************************************************************************
 @Function Name		PVRTMatrixTranslationF
 @Output			mOut	Translation matrix
 @Input				fX		X component of the translation
 @Input				fY		Y component of the translation
 @Input				fZ		Z component of the translation
 @Description		Build a transaltion matrix mOut using fX, fY and fZ.
*****************************************************************************/
void PVRTMatrixTranslationF(
	PVRTMATRIXf	&mOut,
	const float	fX,
	const float	fY,
	const float	fZ);
/*!***************************************************************************
 @Function Name		PVRTMatrixTranslationX
 @Output			mOut	Translation matrix
 @Input				fX		X component of the translation
 @Input				fY		Y component of the translation
 @Input				fZ		Z component of the translation
 @Description		Build a transaltion matrix mOut using fX, fY and fZ.
*****************************************************************************/
void PVRTMatrixTranslationX(
	PVRTMATRIXx	&mOut,
	const int	fX,
	const int	fY,
	const int	fZ);

/*!***************************************************************************
 @Function Name		PVRTMatrixScalingF
 @Output			mOut	Scale matrix
 @Input				fX		X component of the scaling
 @Input				fY		Y component of the scaling
 @Input				fZ		Z component of the scaling
 @Description		Build a scale matrix mOut using fX, fY and fZ.
*****************************************************************************/
void PVRTMatrixScalingF(
	PVRTMATRIXf	&mOut,
	const float fX,
	const float fY,
	const float fZ);

/*!***************************************************************************
 @Function Name		PVRTMatrixScalingX
 @Output			mOut	Scale matrix
 @Input				fX		X component of the scaling
 @Input				fY		Y component of the scaling
 @Input				fZ		Z component of the scaling
 @Description		Build a scale matrix mOut using fX, fY and fZ.
*****************************************************************************/
void PVRTMatrixScalingX(
	PVRTMATRIXx	&mOut,
	const int	fX,
	const int	fY,
	const int	fZ);

/*!***************************************************************************
 @Function Name		PVRTMatrixRotationXF
 @Output			mOut	Rotation matrix
 @Input				fAngle	Angle of the rotation
 @Description		Create an X rotation matrix mOut.
*****************************************************************************/
void PVRTMatrixRotationXF(
	PVRTMATRIXf	&mOut,
	const float fAngle);

/*!***************************************************************************
 @Function Name		PVRTMatrixRotationXX
 @Output			mOut	Rotation matrix
 @Input				fAngle	Angle of the rotation
 @Description		Create an X rotation matrix mOut.
*****************************************************************************/
void PVRTMatrixRotationXX(
	PVRTMATRIXx	&mOut,
	const int	fAngle);

/*!***************************************************************************
 @Function Name		PVRTMatrixRotationYF
 @Output			mOut	Rotation matrix
 @Input				fAngle	Angle of the rotation
 @Description		Create an Y rotation matrix mOut.
*****************************************************************************/
void PVRTMatrixRotationYF(
	PVRTMATRIXf	&mOut,
	const float fAngle);

/*!***************************************************************************
 @Function Name		PVRTMatrixRotationYX
 @Output			mOut	Rotation matrix
 @Input				fAngle	Angle of the rotation
 @Description		Create an Y rotation matrix mOut.
*****************************************************************************/
void PVRTMatrixRotationYX(
	PVRTMATRIXx	&mOut,
	const int	fAngle);

/*!***************************************************************************
 @Function Name		PVRTMatrixRotationZF
 @Output			mOut	Rotation matrix
 @Input				fAngle	Angle of the rotation
 @Description		Create an Z rotation matrix mOut.
*****************************************************************************/
void PVRTMatrixRotationZF(
	PVRTMATRIXf	&mOut,
	const float fAngle);
/*!***************************************************************************
 @Function Name		PVRTMatrixRotationZX
 @Output			mOut	Rotation matrix
 @Input				fAngle	Angle of the rotation
 @Description		Create an Z rotation matrix mOut.
*****************************************************************************/
void PVRTMatrixRotationZX(
	PVRTMATRIXx	&mOut,
	const int	fAngle);

/*!***************************************************************************
 @Function Name		PVRTMatrixTransposeF
 @Output			mOut	Transposed matrix
 @Input				mIn		Original matrix
 @Description		Compute the transpose matrix of mIn.
*****************************************************************************/
void PVRTMatrixTransposeF(
	PVRTMATRIXf			&mOut,
	const PVRTMATRIXf	&mIn);
/*!***************************************************************************
 @Function Name		PVRTMatrixTransposeX
 @Output			mOut	Transposed matrix
 @Input				mIn		Original matrix
 @Description		Compute the transpose matrix of mIn.
*****************************************************************************/
void PVRTMatrixTransposeX(
	PVRTMATRIXx			&mOut,
	const PVRTMATRIXx	&mIn);

/*!***************************************************************************
 @Function			PVRTMatrixInverseF
 @Output			mOut	Inversed matrix
 @Input				mIn		Original matrix
 @Description		Compute the inverse matrix of mIn.
					The matrix must be of the form :
					A 0
					C 1
					Where A is a 3x3 matrix and C is a 1x3 matrix.
*****************************************************************************/
void PVRTMatrixInverseF(
	PVRTMATRIXf			&mOut,
	const PVRTMATRIXf	&mIn);
/*!***************************************************************************
 @Function			PVRTMatrixInverseX
 @Output			mOut	Inversed matrix
 @Input				mIn		Original matrix
 @Description		Compute the inverse matrix of mIn.
					The matrix must be of the form :
					A 0
					C 1
					Where A is a 3x3 matrix and C is a 1x3 matrix.
*****************************************************************************/
void PVRTMatrixInverseX(
	PVRTMATRIXx			&mOut,
	const PVRTMATRIXx	&mIn);

/*!***************************************************************************
 @Function			PVRTMatrixInverseExF
 @Output			mOut	Inversed matrix
 @Input				mIn		Original matrix
 @Description		Compute the inverse matrix of mIn.
					Uses a linear equation solver and the knowledge that M.M^-1=I.
					Use this fn to calculate the inverse of matrices that
					PVRTMatrixInverse() cannot.
*****************************************************************************/
void PVRTMatrixInverseExF(
	PVRTMATRIXf			&mOut,
	const PVRTMATRIXf	&mIn);
/*!***************************************************************************
 @Function			PVRTMatrixInverseExX
 @Output			mOut	Inversed matrix
 @Input				mIn		Original matrix
 @Description		Compute the inverse matrix of mIn.
					Uses a linear equation solver and the knowledge that M.M^-1=I.
					Use this fn to calculate the inverse of matrices that
					PVRTMatrixInverse() cannot.
*****************************************************************************/
void PVRTMatrixInverseExX(
	PVRTMATRIXx			&mOut,
	const PVRTMATRIXx	&mIn);

/*!***************************************************************************
 @Function			PVRTMatrixLookAtLHF
 @Output			mOut	Look-at view matrix
 @Input				vEye	Position of the camera
 @Input				vAt		Point the camera is looking at
 @Input				vUp		Up direction for the camera
 @Description		Create a look-at view matrix.
*****************************************************************************/
void PVRTMatrixLookAtLHF(
	PVRTMATRIXf			&mOut,
	const PVRTVECTOR3f	&vEye,
	const PVRTVECTOR3f	&vAt,
	const PVRTVECTOR3f	&vUp);
/*!***************************************************************************
 @Function			PVRTMatrixLookAtLHX
 @Output			mOut	Look-at view matrix
 @Input				vEye	Position of the camera
 @Input				vAt		Point the camera is looking at
 @Input				vUp		Up direction for the camera
 @Description		Create a look-at view matrix.
*****************************************************************************/
void PVRTMatrixLookAtLHX(
	PVRTMATRIXx			&mOut,
	const PVRTVECTOR3x	&vEye,
	const PVRTVECTOR3x	&vAt,
	const PVRTVECTOR3x	&vUp);

/*!***************************************************************************
 @Function			PVRTMatrixLookAtRHF
 @Output			mOut	Look-at view matrix
 @Input				vEye	Position of the camera
 @Input				vAt		Point the camera is looking at
 @Input				vUp		Up direction for the camera
 @Description		Create a look-at view matrix.
*****************************************************************************/
void PVRTMatrixLookAtRHF(
	PVRTMATRIXf			&mOut,
	const PVRTVECTOR3f	&vEye,
	const PVRTVECTOR3f	&vAt,
	const PVRTVECTOR3f	&vUp);
/*!***************************************************************************
 @Function			PVRTMatrixLookAtRHX
 @Output			mOut	Look-at view matrix
 @Input				vEye	Position of the camera
 @Input				vAt		Point the camera is looking at
 @Input				vUp		Up direction for the camera
 @Description		Create a look-at view matrix.
*****************************************************************************/
void PVRTMatrixLookAtRHX(
	PVRTMATRIXx			&mOut,
	const PVRTVECTOR3x	&vEye,
	const PVRTVECTOR3x	&vAt,
	const PVRTVECTOR3x	&vUp);

/*!***************************************************************************
 @Function		PVRTMatrixPerspectiveFovLHF
 @Output		mOut		Perspective matrix
 @Input			fFOVy		Field of view
 @Input			fAspect		Aspect ratio
 @Input			fNear		Near clipping distance
 @Input			fFar		Far clipping distance
 @Input			bRotate		Should we rotate it ? (for upright screens)
 @Description	Create a perspective matrix.
*****************************************************************************/
void PVRTMatrixPerspectiveFovLHF(
	PVRTMATRIXf	&mOut,
	const float	fFOVy,
	const float	fAspect,
	const float	fNear,
	const float	fFar,
	const bool  bRotate = false);
/*!***************************************************************************
 @Function		PVRTMatrixPerspectiveFovLHX
 @Output		mOut		Perspective matrix
 @Input			fFOVy		Field of view
 @Input			fAspect		Aspect ratio
 @Input			fNear		Near clipping distance
 @Input			fFar		Far clipping distance
 @Input			bRotate		Should we rotate it ? (for upright screens)
 @Description	Create a perspective matrix.
*****************************************************************************/
void PVRTMatrixPerspectiveFovLHX(
	PVRTMATRIXx	&mOut,
	const int	fFOVy,
	const int	fAspect,
	const int	fNear,
	const int	fFar,
	const bool  bRotate = false);

/*!***************************************************************************
 @Function		PVRTMatrixPerspectiveFovRHF
 @Output		mOut		Perspective matrix
 @Input			fFOVy		Field of view
 @Input			fAspect		Aspect ratio
 @Input			fNear		Near clipping distance
 @Input			fFar		Far clipping distance
 @Input			bRotate		Should we rotate it ? (for upright screens)
 @Description	Create a perspective matrix.
*****************************************************************************/
void PVRTMatrixPerspectiveFovRHF(
	PVRTMATRIXf	&mOut,
	const float	fFOVy,
	const float	fAspect,
	const float	fNear,
	const float	fFar,
	const bool  bRotate = false);
/*!***************************************************************************
 @Function		PVRTMatrixPerspectiveFovRHX
 @Output		mOut		Perspective matrix
 @Input			fFOVy		Field of view
 @Input			fAspect		Aspect ratio
 @Input			fNear		Near clipping distance
 @Input			fFar		Far clipping distance
 @Input			bRotate		Should we rotate it ? (for upright screens)
 @Description	Create a perspective matrix.
*****************************************************************************/
void PVRTMatrixPerspectiveFovRHX(
	PVRTMATRIXx	&mOut,
	const int	fFOVy,
	const int	fAspect,
	const int	fNear,
	const int	fFar,
	const bool  bRotate = false);

/*!***************************************************************************
 @Function		PVRTMatrixOrthoLHF
 @Output		mOut		Orthographic matrix
 @Input			w			Width of the screen
 @Input			h			Height of the screen
 @Input			zn			Near clipping distance
 @Input			zf			Far clipping distance
 @Input			bRotate		Should we rotate it ? (for upright screens)
 @Description	Create an orthographic matrix.
*****************************************************************************/
void PVRTMatrixOrthoLHF(
	PVRTMATRIXf	&mOut,
	const float w,
	const float h,
	const float zn,
	const float zf,
	const bool  bRotate = false);
/*!***************************************************************************
 @Function		PVRTMatrixOrthoLHX
 @Output		mOut		Orthographic matrix
 @Input			w			Width of the screen
 @Input			h			Height of the screen
 @Input			zn			Near clipping distance
 @Input			zf			Far clipping distance
 @Input			bRotate		Should we rotate it ? (for upright screens)
 @Description	Create an orthographic matrix.
*****************************************************************************/
void PVRTMatrixOrthoLHX(
	PVRTMATRIXx	&mOut,
	const int	w,
	const int	h,
	const int	zn,
	const int	zf,
	const bool  bRotate = false);

/*!***************************************************************************
 @Function		PVRTMatrixOrthoRHF
 @Output		mOut		Orthographic matrix
 @Input			w			Width of the screen
 @Input			h			Height of the screen
 @Input			zn			Near clipping distance
 @Input			zf			Far clipping distance
 @Input			bRotate		Should we rotate it ? (for upright screens)
 @Description	Create an orthographic matrix.
*****************************************************************************/
void PVRTMatrixOrthoRHF(
	PVRTMATRIXf	&mOut,
	const float w,
	const float h,
	const float zn,
	const float zf,
	const bool  bRotate = false);
/*!***************************************************************************
 @Function		PVRTMatrixOrthoRHX
 @Output		mOut		Orthographic matrix
 @Input			w			Width of the screen
 @Input			h			Height of the screen
 @Input			zn			Near clipping distance
 @Input			zf			Far clipping distance
 @Input			bRotate		Should we rotate it ? (for upright screens)
 @Description	Create an orthographic matrix.
*****************************************************************************/
void PVRTMatrixOrthoRHX(
	PVRTMATRIXx	&mOut,
	const int	w,
	const int	h,
	const int	zn,
	const int	zf,
	const bool  bRotate = false);

/*!***************************************************************************
 @Function			PVRTMatrixVec3LerpF
 @Output			vOut	Result of the interpolation
 @Input				v1		First vector to interpolate from
 @Input				v2		Second vector to interpolate form
 @Input				s		Coefficient of interpolation
 @Description		This function performs the linear interpolation based on
					the following formula: V1 + s(V2-V1).
*****************************************************************************/
void PVRTMatrixVec3LerpF(
	PVRTVECTOR3f		&vOut,
	const PVRTVECTOR3f	&v1,
	const PVRTVECTOR3f	&v2,
	const float			s);
/*!***************************************************************************
 @Function			PVRTMatrixVec3LerpX
 @Output			vOut	Result of the interpolation
 @Input				v1		First vector to interpolate from
 @Input				v2		Second vector to interpolate form
 @Input				s		Coefficient of interpolation
 @Description		This function performs the linear interpolation based on
					the following formula: V1 + s(V2-V1).
*****************************************************************************/
void PVRTMatrixVec3LerpX(
	PVRTVECTOR3x		&vOut,
	const PVRTVECTOR3x	&v1,
	const PVRTVECTOR3x	&v2,
	const int			s);

/*!***************************************************************************
 @Function			PVRTMatrixVec3DotProductF
 @Input				v1		First vector
 @Input				v2		Second vector
 @Return			Dot product of the two vectors.
 @Description		This function performs the dot product of the two
					supplied vectors.
*****************************************************************************/
float PVRTMatrixVec3DotProductF(
	const PVRTVECTOR3f	&v1,
	const PVRTVECTOR3f	&v2);
/*!***************************************************************************
 @Function			PVRTMatrixVec3DotProductX
 @Input				v1		First vector
 @Input				v2		Second vector
 @Return			Dot product of the two vectors.
 @Description		This function performs the dot product of the two
					supplied vectors.
					A single >> 16 shift could be applied to the final accumulated
					result however this runs the risk of overflow between the
					results of the intermediate additions.
*****************************************************************************/
int PVRTMatrixVec3DotProductX(
	const PVRTVECTOR3x	&v1,
	const PVRTVECTOR3x	&v2);

/*!***************************************************************************
 @Function			PVRTMatrixVec3CrossProductF
 @Output			vOut	Cross product of the two vectors
 @Input				v1		First vector
 @Input				v2		Second vector
 @Description		This function performs the cross product of the two
					supplied vectors.
*****************************************************************************/
void PVRTMatrixVec3CrossProductF(
	PVRTVECTOR3f		&vOut,
	const PVRTVECTOR3f	&v1,
	const PVRTVECTOR3f	&v2);
/*!***************************************************************************
 @Function			PVRTMatrixVec3CrossProductX
 @Output			vOut	Cross product of the two vectors
 @Input				v1		First vector
 @Input				v2		Second vector
 @Description		This function performs the cross product of the two
					supplied vectors.
*****************************************************************************/
void PVRTMatrixVec3CrossProductX(
	PVRTVECTOR3x		&vOut,
	const PVRTVECTOR3x	&v1,
	const PVRTVECTOR3x	&v2);

/*!***************************************************************************
 @Function			PVRTMatrixVec3NormalizeF
 @Output			vOut	Normalized vector
 @Input				vIn		Vector to normalize
 @Description		Normalizes the supplied vector.
*****************************************************************************/
void PVRTMatrixVec3NormalizeF(
	PVRTVECTOR3f		&vOut,
	const PVRTVECTOR3f	&vIn);
/*!***************************************************************************
 @Function			PVRTMatrixVec3NormalizeX
 @Output			vOut	Normalized vector
 @Input				vIn		Vector to normalize
 @Description		Normalizes the supplied vector.
					The square root function is currently still performed
					in floating-point.
					Original vector is scaled down prior to be normalized in
					order to avoid overflow issues.
*****************************************************************************/
void PVRTMatrixVec3NormalizeX(
	PVRTVECTOR3x		&vOut,
	const PVRTVECTOR3x	&vIn);
/*!***************************************************************************
 @Function			PVRTMatrixVec3LengthF
 @Input				vIn		Vector to get the length of
 @Return			The length of the vector
  @Description		Gets the length of the supplied vector.
*****************************************************************************/
float PVRTMatrixVec3LengthF(
	const PVRTVECTOR3f	&vIn);
/*!***************************************************************************
 @Function			PVRTMatrixVec3LengthX
 @Input				vIn		Vector to get the length of
 @Return			The length of the vector
 @Description		Gets the length of the supplied vector
*****************************************************************************/
int PVRTMatrixVec3LengthX(
	const PVRTVECTOR3x	&vIn);
/*!***************************************************************************
 @Function			PVRTMatrixLinearEqSolveF
 @Input				pSrc	2D array of floats. 4 Eq linear problem is 5x4
							matrix, constants in first column
 @Input				nCnt	Number of equations to solve
 @Output			pRes	Result
 @Description		Solves 'nCnt' simultaneous equations of 'nCnt' variables.
					pRes should be an array large enough to contain the
					results: the values of the 'nCnt' variables.
					This fn recursively uses Gaussian Elimination.
*****************************************************************************/

void PVRTMatrixLinearEqSolveF(
	float		* const pRes,
	float		** const pSrc,
	const int	nCnt);
/*!***************************************************************************
 @Function			PVRTMatrixLinearEqSolveX
 @Input				pSrc	2D array of floats. 4 Eq linear problem is 5x4
							matrix, constants in first column
 @Input				nCnt	Number of equations to solve
 @Output			pRes	Result
 @Description		Solves 'nCnt' simultaneous equations of 'nCnt' variables.
					pRes should be an array large enough to contain the
					results: the values of the 'nCnt' variables.
					This fn recursively uses Gaussian Elimination.
*****************************************************************************/
void PVRTMatrixLinearEqSolveX(
	int			* const pRes,
	int			** const pSrc,
	const int	nCnt);

#endif

/*****************************************************************************
 End of file (PVRTMatrix.h)
*****************************************************************************/
