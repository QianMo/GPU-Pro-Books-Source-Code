#pragma once

#include <math.h>

class CoreMatrix4x4;
class CoreVector3;

// Math. operations

// Adds 2 vectors
CoreVector3 CoreVector3Add(CoreVector3& v1, CoreVector3& v2);

// Subtracts 2 vectors
CoreVector3 CoreVector3Sub(CoreVector3& v1, CoreVector3& v2);

// Multiplies 2 vectors
CoreVector3 CoreVector3Mul(CoreVector3& v1, CoreVector3& v2);

// Multiplies a vector with a float
CoreVector3 CoreVector3Mul(CoreVector3& vIn, float fIn);

// Divides 2 vectors
CoreVector3 CoreVector3Div(CoreVector3& v1, CoreVector3& v2);

// Divides a vector through a float
CoreVector3 CoreVector3Div(CoreVector3& vIn, float fIn);

// Negates a vector
CoreVector3 CoreVector3Negate(CoreVector3& vIn);

// Transform a position-vector
#ifndef WIN64
CoreVector3 CoreVector3TransformCoords_3DNow(CoreVector3& vIn, CoreMatrix4x4& mIn);
#endif
CoreVector3 CoreVector3TransformCoords_Normal(CoreVector3& vIn, CoreMatrix4x4& mIn);
extern CoreVector3 (*CoreVector3TransformCoords)(CoreVector3& vIn, CoreMatrix4x4& mIn);

// Transform a direction-vector
#ifndef WIN64
CoreVector3 CoreVector3TransformNormal_3DNow(CoreVector3& vIn, CoreMatrix4x4& mIn);
#endif
CoreVector3 CoreVector3TransformNormal_Normal(CoreVector3& vIn, CoreMatrix4x4& mIn);
extern CoreVector3 (*CoreVector3TransformNormal)(CoreVector3& vIn, CoreMatrix4x4& mIn);



// Vector functions


// Length of a vector
#ifndef WIN64
float CoreVector3Length_3DNow(CoreVector3& vIn);
#endif
float CoreVector3Length_Normal(CoreVector3& vIn);
extern float (*CoreVector3Length)(CoreVector3& vIn);

// Length of a vector Squared
#ifndef WIN64
float CoreVector3LengthSq_3DNow(CoreVector3& vIn);
#endif
float CoreVector3LengthSq_Normal(CoreVector3& vIn);
extern float (*CoreVector3LengthSq)(CoreVector3& vIn);

// Crossproduct of 2 vectors
#ifndef WIN64
CoreVector3 CoreVector3Cross_3DNow(CoreVector3& v1, CoreVector3& v2);
#endif
CoreVector3 CoreVector3Cross_Normal(CoreVector3& v1, CoreVector3& v2);
extern CoreVector3 (*CoreVector3Cross)(CoreVector3& v1, CoreVector3& v2);

// Dotproduct of 2 vectors
#ifndef WIN64
float CoreVector3Dot_3DNow(CoreVector3& v1, CoreVector3& v2);
#endif
float CoreVector3Dot_Normal(CoreVector3& v1, CoreVector3& v2);
extern float (*CoreVector3Dot)(CoreVector3& v1, CoreVector3& v2);

// Normalizes the vector
#ifndef WIN64
CoreVector3 CoreVector3Normalize_3DNow(CoreVector3& vIn);
#endif
CoreVector3 CoreVector3Normalize_Normal(CoreVector3& vIn);
extern CoreVector3 (*CoreVector3Normalize)(CoreVector3& vIn);


class CoreVector3
{
	public:
		
		union
		{
			struct 
			{
				float x;
				float y;
				float z;
			};
			
			struct
			{
				float u;
				float v;
				float w;
			};
		};

		// Constructors
		inline CoreVector3(const float _x, const float _y, const float _z) : x(_x), y(_y), z(_z)    {}
		inline CoreVector3() { x = y = z = 0.0f; }
		
		// inline functions
		inline float Length()													   { return CoreVector3Length(*this); }
		inline float LengthSq()													   { return CoreVector3LengthSq(*this); }
		inline float Dot(CoreVector3& v2)										   { return CoreVector3Dot(*this, v2); }
		inline void TransformCoordsThis(CoreMatrix4x4& mIn)						   { *this = CoreVector3TransformCoords(*this, mIn); }
		inline CoreVector3 TransformCoords(CoreMatrix4x4& mIn)					   { return CoreVector3TransformCoords(*this, mIn); }
		inline void TransformNormalThis(CoreMatrix4x4& mIn)						   { *this = CoreVector3TransformNormal(*this, mIn); }
		inline CoreVector3 TransformNormal(CoreMatrix4x4& mIn)					   { return CoreVector3TransformNormal(*this, mIn); }
		inline void NormalizeThis()												   { *this = CoreVector3Normalize(*this); }
		inline CoreVector3 Normalize()											   { return CoreVector3Normalize(*this); }
		inline void CrossThis(CoreVector3& v2)									   { *this = CoreVector3Cross(*this, v2); }
		inline CoreVector3 Cross(CoreVector3& v2)								   { return CoreVector3Cross(*this, v2); }

		// Math. operators
		inline CoreVector3 operator + (CoreVector3& v2)								{ return CoreVector3Add(*this, v2); }
		inline CoreVector3 operator += (CoreVector3& v2)							{ *this = *this + v2; return *this; }
		
		inline CoreVector3 operator - (CoreVector3& v2)								{ return CoreVector3Sub(*this, v2); }
		inline CoreVector3 operator -= (CoreVector3& v2)							{ *this = *this - v2; return *this; }

		inline CoreVector3 operator * (CoreVector3& v2)								{ return CoreVector3Mul(*this, v2); }
		inline CoreVector3 operator *= (CoreVector3& v2)							{ *this = *this * v2; return *this; }

		inline CoreVector3 operator * (float fIn)									{ return CoreVector3Mul(*this, fIn); }
		inline CoreVector3 operator *= (float fIn)									{ *this = *this * fIn; return *this; }

		inline CoreVector3 operator / (CoreVector3& v2)								{ return CoreVector3Div(*this, v2); }
		inline CoreVector3 operator /= (CoreVector3& v2)							{ *this = *this / v2; return *this; }

		inline CoreVector3 operator / (float fIn)									{ return CoreVector3Div(*this, fIn); }
		inline CoreVector3 operator /= (float fIn)									{ *this = *this / fIn; return *this; }

		inline CoreVector3 operator -()												{ return CoreVector3Negate(*this); }
		inline void NegateThis()													{ *this = -*this; }

		inline bool operator ==(CoreVector3& vIn)						
		{
			if(x == vIn.x && y == vIn.y && z == vIn.z)
				return true;
			return false;
		}

		inline bool operator !=(CoreVector3& vIn)						
		{
			return !(*this == vIn);
		}
};
