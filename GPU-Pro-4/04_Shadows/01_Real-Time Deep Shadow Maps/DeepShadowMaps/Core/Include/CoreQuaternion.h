#pragma once

#include "CoreMatrix4x4.h"
#include "CoreVector3.h"
#include <math.h>

class CoreQuaternion;


// Adds 2 Quaternions
CoreQuaternion CoreQuaternionAdd(CoreQuaternion &q1, CoreQuaternion &q2);
// Subs 2 Quaternions
CoreQuaternion CoreQuaternionSub(CoreQuaternion &q1, CoreQuaternion &q2);
// Muls 2 Quaternions
CoreQuaternion CoreQuaternionMul(CoreQuaternion &q1, CoreQuaternion &q2);

// Normalizes a Quaternion
CoreQuaternion CoreQuaternionNormalize(CoreQuaternion &qIn);
// Spherical Linear Interpolation
CoreQuaternion CoreQuaternionSlerp(CoreQuaternion &q1, CoreQuaternion &q2, float fAlpha);

class CoreQuaternion
{
public:
	union
	{
		struct
		{
			float x, y, z, w;
		};

		struct
		{
			CoreVector3 v;
			float w;
		};
	};

	// Constructors
	inline CoreQuaternion()	{ w = 1.0f; x = y = z = 0.0f; }
	inline CoreQuaternion(float x, float y, float z, float w)
	{	this->w = w; this->x = x; this->y = y; this->z = z; }
	// Create a Quaternion from a Matrix
	CoreQuaternion(CoreMatrix4x4 &mIn);
	// Create a Quaternion from an Axis/Angle
	CoreQuaternion(CoreVector3 &vAxis, float fAlpha);


	// Operators
	inline CoreQuaternion operator + (CoreQuaternion &q2)	{ return CoreQuaternionAdd(*this, q2); }
	inline CoreQuaternion operator - (CoreQuaternion &q2)	{ return CoreQuaternionSub(*this, q2); }
	inline CoreQuaternion operator * (CoreQuaternion &q2) { return CoreQuaternionMul(*this, q2); }
	inline CoreQuaternion operator * (float fIn)		{ return CoreQuaternion(x * fIn, y * fIn, z * fIn, w * fIn); }
	inline CoreQuaternion operator / (float fIn)		{ return CoreQuaternion(x / fIn, y / fIn, z / fIn, w / fIn); }

	inline CoreQuaternion operator += (CoreQuaternion &q2) { *this = *this + q2; return *this; }
	inline CoreQuaternion operator -= (CoreQuaternion &q2) { *this = *this - q2; return *this; }
	inline CoreQuaternion operator *= (CoreQuaternion &q2) { *this = *this * q2; return *this; }
	inline CoreQuaternion operator *= (float fIn)		 { *this = *this * fIn; return *this; }
	inline CoreQuaternion operator /= (float fIn)		 { *this = *this / fIn; return *this; }

	CoreQuaternion Normalize()							 { return CoreQuaternionNormalize(*this); }
	void NormalizeThis()								 { *this = CoreQuaternionNormalize(*this); }
	CoreQuaternion Slerp(CoreQuaternion &q2, float fAlpha) { return CoreQuaternionSlerp(*this, q2, fAlpha); }
	void SlerpThis(CoreQuaternion &q2, float fAlpha)	 { *this = CoreQuaternionSlerp(*this, q2, fAlpha); }
};