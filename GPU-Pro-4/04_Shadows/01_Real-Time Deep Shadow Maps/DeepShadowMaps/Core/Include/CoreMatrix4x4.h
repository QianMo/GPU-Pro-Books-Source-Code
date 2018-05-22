#pragma once

#include <math.h>

class CoreVector3;
class CoreMatrix4x4;
class CoreQuaternion;

// Matrix Mathematical Ops

// Adds 2 matrices
#ifndef WIN64
CoreMatrix4x4 CoreMatrix4x4Add_3DNow(CoreMatrix4x4& m1, CoreMatrix4x4& m2);
#endif
CoreMatrix4x4 CoreMatrix4x4Add_Normal(CoreMatrix4x4& m1, CoreMatrix4x4& m2);
extern CoreMatrix4x4 (*CoreMatrix4x4Add)(CoreMatrix4x4 &m1, CoreMatrix4x4 &m2);

// Subs 2 matrices
#ifndef WIN64
CoreMatrix4x4 CoreMatrix4x4Sub_3DNow(CoreMatrix4x4& m1, CoreMatrix4x4& m2);
#endif
CoreMatrix4x4 CoreMatrix4x4Sub_Normal(CoreMatrix4x4& m1, CoreMatrix4x4& m2);
extern CoreMatrix4x4 (*CoreMatrix4x4Sub)(CoreMatrix4x4 &m1, CoreMatrix4x4 &m2);

// Multiplies 2 matrices
#ifndef WIN64
CoreMatrix4x4 CoreMatrix4x4Mul_3DNow(CoreMatrix4x4& m1, CoreMatrix4x4& m2);
#endif
CoreMatrix4x4 CoreMatrix4x4Mul_Normal(CoreMatrix4x4& m1, CoreMatrix4x4& m2);
extern CoreMatrix4x4 (*CoreMatrix4x4Mul)(CoreMatrix4x4 &m1, CoreMatrix4x4 &m2);

// Multiplies a matrix with a float
#ifndef WIN64
CoreMatrix4x4 CoreMatrix4x4MulFloat_3DNow(CoreMatrix4x4& mIn, float fIn);
#endif
CoreMatrix4x4 CoreMatrix4x4MulFloat_Normal(CoreMatrix4x4& mIn, float fIn);
extern CoreMatrix4x4 (*CoreMatrix4x4MulFloat)(CoreMatrix4x4& mIn, float fIn);

// Divides a Matrix through a float
#ifndef WIN64
CoreMatrix4x4 CoreMatrix4x4DivFloat_3DNow(CoreMatrix4x4& mIn, float fIn);
#endif
CoreMatrix4x4 CoreMatrix4x4DivFloat_Normal(CoreMatrix4x4& mIn, float fIn);
extern CoreMatrix4x4 (*CoreMatrix4x4DivFloat)(CoreMatrix4x4 &mIn, float fIn);



// Misc matrix functions

// Translation
CoreMatrix4x4 CoreMatrix4x4Translation(CoreVector3& vIn);

// Scaling
CoreMatrix4x4 CoreMatrix4x4Scale(CoreVector3& vIn);

// Rotation X
CoreMatrix4x4 CoreMatrix4x4RotationX(float alpha);

// Rotation Y
CoreMatrix4x4 CoreMatrix4x4RotationY(float alpha);

// Rotation Z
CoreMatrix4x4 CoreMatrix4x4RotationZ(float alpha);

// Rotation
CoreMatrix4x4 CoreMatrix4x4Rotation(CoreVector3& alpha);

//Rotation um bestimmte Axe
CoreMatrix4x4 CoreMatrix4x4RotationAxis(CoreVector3& vIn, float fIn);

// Axis Matrix
CoreMatrix4x4 CoreMatrix4x4Axes(CoreVector3& vXAxis, CoreVector3& vYAxis, CoreVector3& vZAxis);

// Calcs the Determinante of a Matrix
float CoreMatrix4x4Determinante(CoreMatrix4x4& mIn);

// Inverts the Matrix
CoreMatrix4x4 CoreMatrix4x4Invert(CoreMatrix4x4 &mIn);

// Transposes the Matrix
CoreMatrix4x4 CoreMatrix4x4Transpose(CoreMatrix4x4& m);

// Negates the Matrix
CoreMatrix4x4 CoreMatrix4x4Negate(CoreMatrix4x4 &mIn);

// Returns the Trace of the Matrix
float CoreMatrix4x4Trace(CoreMatrix4x4 &mIn);



class CoreMatrix4x4
{
	// Variables
	public:
		
		union
		{
			struct
			{
				float _11, _12, _13, _14,
					  _21, _22, _23, _24,
					  _31, _32, _33, _34,
					  _41, _42, _43, _44;
			};
			float arr[16];
			float arr4x4[4][4];
		};

		// Constructors
		inline CoreMatrix4x4()	
		{ 
			_12 = _13 = _14 = _21 = _23 = _24 = _31 = _32 = _34 = _41 = _42 = _43 = 0.0f;
			_11 = 1.0f;			// Init with Ident
			_22 = 1.0f;
			_33 = 1.0f;
			_44 = 1.0f;
		}
		// Matrix From Quaternion
		CoreMatrix4x4(CoreQuaternion &qIn);

		inline CoreMatrix4x4(float __11, float __12, float __13, float __14,
						float __21, float __22, float __23, float __24,
						float __31, float __32, float __33, float __34,
						float __41, float __42, float __43, float __44) : _11(__11), _12(__12), _13(__13), _14(__14),
																	      _21(__21), _22(__22), _23(__23), _24(__24),
																     	  _31(__31), _32(__32), _33(__33), _34(__34),
																    	  _41(__41), _42(__42), _43(__43), _44(__44){}

	
		// Math. Operations
		inline CoreMatrix4x4 operator + (CoreMatrix4x4& m2)						{  return CoreMatrix4x4Add(*this, m2); }
		inline CoreMatrix4x4 operator += (CoreMatrix4x4& m2)					{  *this = CoreMatrix4x4Add(*this, m2); return *this;}
		
		inline CoreMatrix4x4 operator - (CoreMatrix4x4& m2)						{  CoreMatrix4x4Sub(*this, m2); }
		inline CoreMatrix4x4 operator -= (CoreMatrix4x4& m2)					{  *this = CoreMatrix4x4Sub(*this, m2); return *this;}
		
		inline CoreMatrix4x4 operator * (CoreMatrix4x4& m2)						{  return CoreMatrix4x4Mul(*this, m2); }
		inline CoreMatrix4x4 operator *= (CoreMatrix4x4& m2)					{  *this = CoreMatrix4x4Mul(*this, m2); return *this; }
		inline CoreMatrix4x4 operator * (float fIn)								{  return CoreMatrix4x4MulFloat(*this, fIn); }
		inline CoreMatrix4x4 operator *= (float fIn)							{  *this = CoreMatrix4x4MulFloat(*this, fIn);  return *this;}
			
		inline CoreMatrix4x4 operator / (float fIn)								{  return  CoreMatrix4x4DivFloat(*this, fIn); }
		inline CoreMatrix4x4 operator /= (float fIn)							{  *this = CoreMatrix4x4DivFloat(*this, fIn); return *this;}

		inline CoreMatrix4x4 operator - ()										{ return CoreMatrix4x4Negate(*this); }
		inline void NegateThis()												{ *this = CoreMatrix4x4Negate(*this); }

		
		// coparism
		inline bool Equals(CoreMatrix4x4& mIn)
		{
			if(_11 != mIn._11) return false;
			if(_12 != mIn._12) return false;
			if(_13 != mIn._13) return false;
			if(_14 != mIn._14) return false;

			if(_21 != mIn._21) return false;
			if(_22 != mIn._22) return false;
			if(_23 != mIn._23) return false;
			if(_24 != mIn._24) return false;

			if(_31 != mIn._31) return false;
			if(_32 != mIn._32) return false;
			if(_33 != mIn._33) return false;
			if(_34 != mIn._34) return false;

			if(_41 != mIn._41) return false;
			if(_42 != mIn._42) return false;
			if(_43 != mIn._43) return false;
			return _44 == mIn._44;
		}		
		
		// rest of the functions
		inline float Determinante()									{ return CoreMatrix4x4Determinante(*this); }
		inline CoreMatrix4x4 Invert()								{ return CoreMatrix4x4Invert(*this); }
		inline void InvertThis()									{ *this = CoreMatrix4x4Invert(*this); }
		inline CoreMatrix4x4 Transpose()							{ return CoreMatrix4x4Transpose(*this); }
		inline void TransposeThis()									{ *this = CoreMatrix4x4Transpose(*this); }
		
		// right, left means multiply on the right or left side to the current matrix
		
		inline CoreMatrix4x4 RotateXRight(float alpha)				{ return *this * CoreMatrix4x4RotationX(alpha); }
		inline CoreMatrix4x4 RotateXLeft(float alpha)				{ return CoreMatrix4x4RotationX(alpha) * *this; }
		inline void RotateXRightThis(float alpha)					{ *this *= CoreMatrix4x4RotationX(alpha); }
		inline void RotateXLeftThis(float alpha)					{ *this = CoreMatrix4x4RotationX(alpha) * *this; }

		inline CoreMatrix4x4 RotateYRight(float alpha)				{ return *this * CoreMatrix4x4RotationY(alpha); }
		inline CoreMatrix4x4 RotateYLeft(float alpha)				{ return CoreMatrix4x4RotationY(alpha) * *this; }
		inline void RotateYRightThis(float alpha)					{ *this *= CoreMatrix4x4RotationY(alpha); }
		inline void RotateYLeftThis(float alpha)					{ *this = CoreMatrix4x4RotationY(alpha) * *this; }
		
		inline CoreMatrix4x4 RotateZRight(float alpha)				{ return *this * CoreMatrix4x4RotationZ(alpha); }
		inline CoreMatrix4x4 RotateZLeft(float alpha)				{ return CoreMatrix4x4RotationZ(alpha) * *this; }
		inline void RotateZRightThis(float alpha)					{ *this *= CoreMatrix4x4RotationZ(alpha); }
		inline void RotateZLeftThis(float alpha)					{ *this = CoreMatrix4x4RotationZ(alpha) * *this; }
		
		inline CoreMatrix4x4 TranslateRight(CoreVector3& vIn)		{ return *this * CoreMatrix4x4Translation(vIn); }
		inline CoreMatrix4x4 TranslateLeft(CoreVector3& vIn)		{ return CoreMatrix4x4Translation(vIn) * *this; }
		inline void TranslateRightThis(CoreVector3& vIn)			{ *this *= CoreMatrix4x4Translation(vIn); }
		inline void TranslateLeftThis(CoreVector3& vIn)				{ *this = CoreMatrix4x4Translation(vIn) * *this; }

		inline CoreMatrix4x4 ScaleRight(CoreVector3& vIn)			{ return *this * CoreMatrix4x4Scale(vIn); }
		inline CoreMatrix4x4 ScaleLeft(CoreVector3& vIn)			{ return CoreMatrix4x4Scale(vIn) * *this; }
		inline void ScaleRightThis(CoreVector3& vIn)				{ *this *= CoreMatrix4x4Scale(vIn); }
		inline void ScaleLeftThis(CoreVector3& vIn)					{ *this = CoreMatrix4x4Scale(vIn) * *this; }


		inline float Trace()										{ return CoreMatrix4x4Trace(*this); }
};