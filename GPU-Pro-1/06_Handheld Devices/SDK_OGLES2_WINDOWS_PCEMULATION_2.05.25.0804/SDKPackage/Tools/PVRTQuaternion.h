/******************************************************************************

 @File         PVRTQuaternion.h

 @Title        PVRTQuaternion

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Quaternion functions for floating and fixed point math.

******************************************************************************/
#ifndef _PVRTQUATERNION_H_
#define _PVRTQUATERNION_H_

#include "PVRTGlobal.h"
#include "PVRTMatrix.h"

/****************************************************************************
** Typedefs
****************************************************************************/
/*!***************************************************************************
 Floating point Quaternion
*****************************************************************************/
typedef struct
{
	float x;	/*!< x coordinate */
	float y;	/*!< y coordinate */
	float z;	/*!< z coordinate */
	float w;	/*!< w coordinate */
} PVRTQUATERNIONf;
/*!***************************************************************************
 Fixed point Quaternion
*****************************************************************************/
typedef struct
{
	int x;	/*!< x coordinate */
	int y;	/*!< y coordinate */
	int z;	/*!< z coordinate */
	int w;	/*!< w coordinate */
} PVRTQUATERNIONx;

/****************************************************************************
** Float or fixed
****************************************************************************/
#ifdef PVRT_FIXED_POINT_ENABLE
typedef PVRTQUATERNIONx		PVRTQUATERNION;
#define PVRTMatrixQuaternionIdentity		PVRTMatrixQuaternionIdentityX
#define PVRTMatrixQuaternionRotationAxis	PVRTMatrixQuaternionRotationAxisX
#define PVRTMatrixQuaternionToAxisAngle		PVRTMatrixQuaternionToAxisAngleX
#define PVRTMatrixQuaternionSlerp			PVRTMatrixQuaternionSlerpX
#define PVRTMatrixQuaternionNormalize		PVRTMatrixQuaternionNormalizeX
#define PVRTMatrixRotationQuaternion		PVRTMatrixRotationQuaternionX
#define PVRTMatrixQuaternionMultiply		PVRTMatrixQuaternionMultiplyX
#else
typedef PVRTQUATERNIONf		PVRTQUATERNION;
#define PVRTMatrixQuaternionIdentity		PVRTMatrixQuaternionIdentityF
#define PVRTMatrixQuaternionRotationAxis	PVRTMatrixQuaternionRotationAxisF
#define PVRTMatrixQuaternionToAxisAngle		PVRTMatrixQuaternionToAxisAngleF
#define PVRTMatrixQuaternionSlerp			PVRTMatrixQuaternionSlerpF
#define PVRTMatrixQuaternionNormalize		PVRTMatrixQuaternionNormalizeF
#define PVRTMatrixRotationQuaternion		PVRTMatrixRotationQuaternionF
#define PVRTMatrixQuaternionMultiply		PVRTMatrixQuaternionMultiplyF
#endif

/****************************************************************************
** Functions
****************************************************************************/

/*!***************************************************************************
 @Function			PVRTMatrixQuaternionIdentityF
 @Output			qOut	Identity quaternion
 @Description		Sets the quaternion to (0, 0, 0, 1), the identity quaternion.
*****************************************************************************/
void PVRTMatrixQuaternionIdentityF(
	PVRTQUATERNIONf		&qOut);

/*!***************************************************************************
 @Function			PVRTMatrixQuaternionIdentityX
 @Output			qOut	Identity quaternion
 @Description		Sets the quaternion to (0, 0, 0, 1), the identity quaternion.
*****************************************************************************/
void PVRTMatrixQuaternionIdentityX(
	PVRTQUATERNIONx		&qOut);

/*!***************************************************************************
 @Function			PVRTMatrixQuaternionRotationAxisF
 @Output			qOut	Rotation quaternion
 @Input				vAxis	Axis to rotate around
 @Input				fAngle	Angle to rotate
 @Description		Create quaternion corresponding to a rotation of fAngle
					radians around submitted vector.
*****************************************************************************/
void PVRTMatrixQuaternionRotationAxisF(
	PVRTQUATERNIONf		&qOut,
	const PVRTVECTOR3f	&vAxis,
	const float			fAngle);

/*!***************************************************************************
 @Function			PVRTMatrixQuaternionRotationAxisX
 @Output			qOut	Rotation quaternion
 @Input				vAxis	Axis to rotate around
 @Input				fAngle	Angle to rotate
 @Description		Create quaternion corresponding to a rotation of fAngle
					radians around submitted vector.
*****************************************************************************/
void PVRTMatrixQuaternionRotationAxisX(
	PVRTQUATERNIONx		&qOut,
	const PVRTVECTOR3x	&vAxis,
	const int			fAngle);


/*!***************************************************************************
 @Function			PVRTMatrixQuaternionToAxisAngleF
 @Input				qIn		Quaternion to transform
 @Output			vAxis	Axis of rotation
 @Output			fAngle	Angle of rotation
 @Description		Convert a quaternion to an axis and angle. Expects a unit
					quaternion.
*****************************************************************************/
void PVRTMatrixQuaternionToAxisAngleF(
	const PVRTQUATERNIONf	&qIn,
	PVRTVECTOR3f			&vAxis,
	float					&fAngle);

/*!***************************************************************************
 @Function			PVRTMatrixQuaternionToAxisAngleX
 @Input				qIn		Quaternion to transform
 @Output			vAxis	Axis of rotation
 @Output			fAngle	Angle of rotation
 @Description		Convert a quaternion to an axis and angle. Expects a unit
					quaternion.
*****************************************************************************/
void PVRTMatrixQuaternionToAxisAngleX(
	const PVRTQUATERNIONx	&qIn,
	PVRTVECTOR3x			&vAxis,
	int						&fAngle);

/*!***************************************************************************
 @Function			PVRTMatrixQuaternionSlerpF
 @Output			qOut	Result of the interpolation
 @Input				qA		First quaternion to interpolate from
 @Input				qB		Second quaternion to interpolate from
 @Input				t		Coefficient of interpolation
 @Description		Perform a Spherical Linear intERPolation between quaternion A
					and quaternion B at time t. t must be between 0.0f and 1.0f
*****************************************************************************/
void PVRTMatrixQuaternionSlerpF(
	PVRTQUATERNIONf			&qOut,
	const PVRTQUATERNIONf	&qA,
	const PVRTQUATERNIONf	&qB,
	const float				t);

/*!***************************************************************************
 @Function			PVRTMatrixQuaternionSlerpX
 @Output			qOut	Result of the interpolation
 @Input				qA		First quaternion to interpolate from
 @Input				qB		Second quaternion to interpolate from
 @Input				t		Coefficient of interpolation
 @Description		Perform a Spherical Linear intERPolation between quaternion A
					and quaternion B at time t. t must be between 0.0f and 1.0f
					Requires input quaternions to be normalized
*****************************************************************************/
void PVRTMatrixQuaternionSlerpX(
	PVRTQUATERNIONx			&qOut,
	const PVRTQUATERNIONx	&qA,
	const PVRTQUATERNIONx	&qB,
	const int				t);

/*!***************************************************************************
 @Function			PVRTMatrixQuaternionNormalizeF
 @Modified			quat	Vector to normalize
 @Description		Normalize quaternion.
*****************************************************************************/
void PVRTMatrixQuaternionNormalizeF(PVRTQUATERNIONf &quat);

/*!***************************************************************************
 @Function			PVRTMatrixQuaternionNormalizeX
 @Modified			quat	Vector to normalize
 @Description		Normalize quaternion.
					Original quaternion is scaled down prior to be normalized in
					order to avoid overflow issues.
*****************************************************************************/
void PVRTMatrixQuaternionNormalizeX(PVRTQUATERNIONx &quat);

/*!***************************************************************************
 @Function			PVRTMatrixRotationQuaternionF
 @Output			mOut	Resulting rotation matrix
 @Input				quat	Quaternion to transform
 @Description		Create rotation matrix from submitted quaternion.
					Assuming the quaternion is of the form [X Y Z W]:

						|       2     2									|
						| 1 - 2Y  - 2Z    2XY - 2ZW      2XZ + 2YW		 0	|
						|													|
						|                       2     2					|
					M = | 2XY + 2ZW       1 - 2X  - 2Z   2YZ - 2XW		 0	|
						|													|
						|                                      2     2		|
						| 2XZ - 2YW       2YZ + 2XW      1 - 2X  - 2Y	 0	|
						|													|
						|     0			   0			  0          1  |
*****************************************************************************/
void PVRTMatrixRotationQuaternionF(
	PVRTMATRIXf				&mOut,
	const PVRTQUATERNIONf	&quat);

/*!***************************************************************************
 @Function			PVRTMatrixRotationQuaternionX
 @Output			mOut	Resulting rotation matrix
 @Input				quat	Quaternion to transform
 @Description		Create rotation matrix from submitted quaternion.
					Assuming the quaternion is of the form [X Y Z W]:

						|       2     2									|
						| 1 - 2Y  - 2Z    2XY - 2ZW      2XZ + 2YW		 0	|
						|													|
						|                       2     2					|
					M = | 2XY + 2ZW       1 - 2X  - 2Z   2YZ - 2XW		 0	|
						|													|
						|                                      2     2		|
						| 2XZ - 2YW       2YZ + 2XW      1 - 2X  - 2Y	 0	|
						|													|
						|     0			   0			  0          1  |
*****************************************************************************/
void PVRTMatrixRotationQuaternionX(
	PVRTMATRIXx				&mOut,
	const PVRTQUATERNIONx	&quat);

/*!***************************************************************************
 @Function			PVRTMatrixQuaternionMultiplyF
 @Output			qOut	Resulting quaternion
 @Input				qA		First quaternion to multiply
 @Input				qB		Second quaternion to multiply
 @Description		Multiply quaternion A with quaternion B and return the
					result in qOut.
*****************************************************************************/
void PVRTMatrixQuaternionMultiplyF(
	PVRTQUATERNIONf			&qOut,
	const PVRTQUATERNIONf	&qA,
	const PVRTQUATERNIONf	&qB);

/*!***************************************************************************
 @Function			PVRTMatrixQuaternionMultiplyX
 @Output			qOut	Resulting quaternion
 @Input				qA		First quaternion to multiply
 @Input				qB		Second quaternion to multiply
 @Description		Multiply quaternion A with quaternion B and return the
					result in qOut.
					Input quaternions must be normalized.
*****************************************************************************/
void PVRTMatrixQuaternionMultiplyX(
	PVRTQUATERNIONx			&qOut,
	const PVRTQUATERNIONx	&qA,
	const PVRTQUATERNIONx	&qB);

#endif

/*****************************************************************************
 End of file (PVRTQuaternion.h)
*****************************************************************************/
