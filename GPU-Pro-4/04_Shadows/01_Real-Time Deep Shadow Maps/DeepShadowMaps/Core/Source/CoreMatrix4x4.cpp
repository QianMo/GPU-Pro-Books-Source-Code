#include "Core.h"

// Function pointer declarations
CoreMatrix4x4 (*CoreMatrix4x4Add)(CoreMatrix4x4& m1, CoreMatrix4x4& m2) = &CoreMatrix4x4Add_Normal;
CoreMatrix4x4 (*CoreMatrix4x4Sub)(CoreMatrix4x4& m1, CoreMatrix4x4& m2) = &CoreMatrix4x4Sub_Normal;
CoreMatrix4x4 (*CoreMatrix4x4Mul)(CoreMatrix4x4& m1, CoreMatrix4x4& m2) = &CoreMatrix4x4Mul_Normal;
CoreMatrix4x4 (*CoreMatrix4x4MulFloat)(CoreMatrix4x4& mIn, float fIn) = &CoreMatrix4x4MulFloat_Normal;
CoreMatrix4x4 (*CoreMatrix4x4DivFloat)(CoreMatrix4x4& mIn, float fIn) = &CoreMatrix4x4DivFloat_Normal;

// Matrix From Quaternion
CoreMatrix4x4::CoreMatrix4x4(CoreQuaternion &qIn)
{
	float fXSq = qIn.x * qIn.x;
	float fYSq = qIn.y * qIn.y;
	float fZSq = qIn.z * qIn.z;

	// Quaternion to Matrix formula

	_11 = 1.0f - 2.0f * fYSq - 2.0f * fZSq;
	_12 = 2.0f * qIn.x * qIn.y + 2.0f * qIn.w * qIn.z;
	_13 = 2.0f * qIn.x * qIn.z - 2.0f * qIn.w * qIn.y;
	_14 = 0.0f;

	_21 = 2.0f * qIn.x * qIn.y - 2.0f * qIn.w * qIn.z;
	_22 = 1.0f - 2.0f * fXSq - 2.0f * fZSq;
	_23 = 2.0f * qIn.y * qIn.z + 2.0f * qIn.w * qIn.x;
	_24 = 0.0f;

	_31 = 2.0f * qIn.x * qIn.z + 2.0f * qIn.w * qIn.y;
	_32 = 2.0f * qIn.y * qIn.z - 2.0f * qIn.w * qIn.x;
	_33 = 1.0f - 2.0f * fXSq - 2.0f * fYSq;
	_34 = 0.0f;

	_41 = _42 = _43 = 0.0f;		// No translation
	_44 = 1.0f;
}

#ifndef WIN64
// Adds 2 matrices
CoreMatrix4x4 CoreMatrix4x4Add_3DNow(CoreMatrix4x4& m1, CoreMatrix4x4& m2)
{
	CoreMatrix4x4 mOut;
	_asm
	{
		femms

		mov eax, m1
		mov ebx, m2
		lea ecx, mOut

		prefetch [eax]
		prefetch [ebx]
		
		movq mm0, [eax]
		movq mm1, [ebx]
		pfadd mm0, mm1
		movq [ecx], mm0

		movq mm0, [eax+8]
		movq mm1, [ebx+8]
		pfadd mm0, mm1
		movq [ecx+8], mm0

		movq mm0, [eax+16]
		movq mm1, [ebx+16]
		pfadd mm0, mm1
		movq [ecx+16], mm0

		movq mm0, [eax+24]
		movq mm1, [ebx+24]
		pfadd mm0, mm1
		movq [ecx+24], mm0

		movq mm0, [eax+32]
		movq mm1, [ebx+32]
		pfadd mm0, mm1
		movq [ecx+32], mm0

		movq mm0, [eax+40]
		movq mm1, [ebx+40]
		pfadd mm0, mm1
		movq [ecx+40], mm0

		movq mm0, [eax+48]
		movq mm1, [ebx+48]
		pfadd mm0, mm1
		movq [ecx+48], mm0

		movq mm0, [eax+56]
		movq mm1, [ebx+56]
		pfadd mm0, mm1
		movq [ecx+56], mm0
		
		femms
	}
	return mOut;
}
#endif

// Adds 2 matrices
CoreMatrix4x4 CoreMatrix4x4Add_Normal(CoreMatrix4x4& m1, CoreMatrix4x4& m2)
{
	return CoreMatrix4x4(m1._11 + m2._11, m1._12 + m2._12, m1._13 + m2._13, m1._14 + m2._14, 
				  	 m1._21 + m2._21, m1._22 + m2._22, m1._23 + m2._23, m1._24 + m2._24, 
					 m1._31 + m2._31, m1._32 + m2._32, m1._33 + m2._33, m1._34 + m2._34, 
					 m1._41 + m2._41, m1._42 + m2._42, m1._43 + m2._43, m1._44 + m2._44);
}

#ifndef WIN64

// Subs 2 matrices
CoreMatrix4x4 CoreMatrix4x4Sub_3DNow(CoreMatrix4x4& m1, CoreMatrix4x4& m2)
{
	CoreMatrix4x4 mOut;
	_asm
	{
		femms

		mov eax, m1
		mov ebx, m2
		lea ecx, mOut

		prefetch [eax]
		prefetch [ebx]
		
		movq mm0, [eax]
		movq mm1, [ebx]
		pfsub mm0, mm1
		movq [ecx], mm0

		movq mm0, [eax+8]
		movq mm1, [ebx+8]
		pfsub mm0, mm1
		movq [ecx+8], mm0

		movq mm0, [eax+16]
		movq mm1, [ebx+16]
		pfsub mm0, mm1
		movq [ecx+16], mm0

		movq mm0, [eax+24]
		movq mm1, [ebx+24]
		pfsub mm0, mm1
		movq [ecx+24], mm0

		movq mm0, [eax+32]
		movq mm1, [ebx+32]
		pfsub mm0, mm1
		movq [ecx+32], mm0

		movq mm0, [eax+40]
		movq mm1, [ebx+40]
		pfsub mm0, mm1
		movq [ecx+40], mm0

		movq mm0, [eax+48]
		movq mm1, [ebx+48]
		pfsub mm0, mm1
		movq [ecx+48], mm0

		movq mm0, [eax+56]
		movq mm1, [ebx+56]
		pfsub mm0, mm1
		movq [ecx+56], mm0
		
		femms
	}
	return mOut;
}
#endif

// Subs 2 matrices
CoreMatrix4x4 CoreMatrix4x4Sub_Normal(CoreMatrix4x4& m1, CoreMatrix4x4& m2)
{
	return CoreMatrix4x4(m1._11 - m2._11, m1._12 - m2._12, m1._13 - m2._13, m1._14 - m2._14, 
					 m1._21 - m2._21, m1._22 - m2._22, m1._23 - m2._23, m1._24 - m2._24, 
				  	 m1._31 - m2._31, m1._32 - m2._32, m1._33 - m2._33, m1._34 - m2._34, 
					 m1._41 - m2._41, m1._42 - m2._42, m1._43 - m2._43, m1._44 - m2._44);
}

#ifndef WIN64

// Multiplies a Matrix with a float
CoreMatrix4x4 CoreMatrix4x4MulFloat_3DNow(CoreMatrix4x4& mIn, float fIn)
{
	CoreMatrix4x4 mOut;
	_asm
	{
		femms

		mov eax, mIn
		lea ebx, fIn
		lea ecx, mOut

		prefetch [eax]
		
		movd mm1, [ebx]
		punpckldq mm1, mm1
		
		movq mm0, [eax]
		pfmul mm0, mm1
		movq [ecx], mm0

		movq mm0, [eax+8]
		pfmul mm0, mm1
		movq [ecx+8], mm0

		movq mm0, [eax+16]
		pfmul mm0, mm1
		movq [ecx+16], mm0

		movq mm0, [eax+24]
		pfmul mm0, mm1
		movq [ecx+24], mm0

		movq mm0, [eax+32]
		pfmul mm0, mm1
		movq [ecx+32], mm0

		movq mm0, [eax+40]
		pfmul mm0, mm1
		movq [ecx+40], mm0

		movq mm0, [eax+48]
		pfmul mm0, mm1
		movq [ecx+48], mm0

		movq mm0, [eax+56]
		pfmul mm0, mm1
		movq [ecx+56], mm0
		
		femms
	}
	return mOut;
}



// Multiplies 2 matrices
CoreMatrix4x4 CoreMatrix4x4Mul_3DNow(CoreMatrix4x4& m1, CoreMatrix4x4& m2)
{	
	CoreMatrix4x4 mOut;
	_asm
	{
		femms
		
		mov eax, m1
		mov ebx, m2
		lea ecx, mOut

		prefetch [eax]
		prefetch [ebx]

				//_11
		movq mm0, [eax]
		movd mm1, [ebx]
		movd mm2, [ebx+16]
		punpckldq mm1, mm2
		pfmul mm0, mm1

		movq mm2, [eax+8]
		movd mm3, [ebx+32]
		movd mm4, [ebx+48]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx], mm0

				//_12
		movq mm0, [eax]
		movd mm1, [ebx+4]
		movd mm2, [ebx+20]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+8]
		movd mm3, [ebx+36]
		movd mm4, [ebx+52]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+4], mm0
				
			//_13
		movq mm0, [eax]
		movd mm1, [ebx+8]
		movd mm2, [ebx+24]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+8]
		movd mm3, [ebx+40]
		movd mm4, [ebx+56]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+8], mm0

			//_14
		movq mm0, [eax]
		movd mm1, [ebx+12]
		movd mm2, [ebx+28]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+8]
		movd mm3, [ebx+44]
		movd mm4, [ebx+60]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+12], mm0


				//_21
		movq mm0, [eax+16]
		movd mm1, [ebx]
		movd mm2, [ebx+16]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+24]
		movd mm3, [ebx+32]
		movd mm4, [ebx+48]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+16], mm0

				//_22
		movq mm0, [eax+16]
		movd mm1, [ebx+4]
		movd mm2, [ebx+20]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+24]
		movd mm3, [ebx+36]
		movd mm4, [ebx+52]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+20], mm0
				
			//_23
		movq mm0, [eax+16]
		movd mm1, [ebx+8]
		movd mm2, [ebx+24]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+24]
		movd mm3, [ebx+40]
		movd mm4, [ebx+56]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+24], mm0

			//_24
		movq mm0, [eax+16]
		movd mm1, [ebx+12]
		movd mm2, [ebx+28]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+24]
		movd mm3, [ebx+44]
		movd mm4, [ebx+60]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+28], mm0

				//_31
		movq mm0, [eax+32]
		movd mm1, [ebx]
		movd mm2, [ebx+16]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+40]
		movd mm3, [ebx+32]
		movd mm4, [ebx+48]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+32], mm0

				//_32
		movq mm0, [eax+32]
		movd mm1, [ebx+4]
		movd mm2, [ebx+20]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+40]
		movd mm3, [ebx+36]
		movd mm4, [ebx+52]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+36], mm0
				
			//_33
		movq mm0, [eax+32]
		movd mm1, [ebx+8]
		movd mm2, [ebx+24]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+40]
		movd mm3, [ebx+40]
		movd mm4, [ebx+56]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+40], mm0

			//_34
		movq mm0, [eax+32]
		movd mm1, [ebx+12]
		movd mm2, [ebx+28]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+40]
		movd mm3, [ebx+44]
		movd mm4, [ebx+60]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+44], mm0

				//_41
		movq mm0, [eax+48]
		movd mm1, [ebx]
		movd mm2, [ebx+16]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+56]
		movd mm3, [ebx+32]
		movd mm4, [ebx+48]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+48], mm0

				//_42
		movq mm0, [eax+48]
		movd mm1, [ebx+4]
		movd mm2, [ebx+20]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+56]
		movd mm3, [ebx+36]
		movd mm4, [ebx+52]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+52], mm0
				
			//_43
		movq mm0, [eax+48]
		movd mm1, [ebx+8]
		movd mm2, [ebx+24]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+56]
		movd mm3, [ebx+40]
		movd mm4, [ebx+56]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+56], mm0

			//_44
		movq mm0, [eax+48]
		movd mm1, [ebx+12]
		movd mm2, [ebx+28]
		punpckldq mm1,mm2
		pfmul mm0,mm1

		movq mm2, [eax+56]
		movd mm3, [ebx+44]
		movd mm4, [ebx+60]
		punpckldq mm3,mm4
		pfmul mm2,mm3

		pfadd mm0, mm2
		pfacc mm0, mm0

		movd [ecx+60], mm0

		
		femms
	};
	return mOut;
}
#endif
// Multiplies a Matrix with a float
CoreMatrix4x4 CoreMatrix4x4MulFloat_Normal(CoreMatrix4x4& mIn, float fIn)
{
	return CoreMatrix4x4(mIn._11 * fIn, mIn._12 * fIn, mIn._13 * fIn, mIn._14 * fIn,
					mIn._21 * fIn, mIn._22 * fIn, mIn._23 * fIn, mIn._24 * fIn,
					mIn._31 * fIn, mIn._32 * fIn, mIn._33 * fIn, mIn._34 * fIn,
					mIn._41 * fIn, mIn._42 * fIn, mIn._43 * fIn, mIn._44 * fIn);
}


// Multiplies 2 matrices
CoreMatrix4x4 CoreMatrix4x4Mul_Normal(CoreMatrix4x4& m1, CoreMatrix4x4& m2)
{
	return CoreMatrix4x4(
		m2._11 * m1._11 + m2._21 * m1._12 + m2._31 * m1._13 + m2._41 * m1._14,
		m2._12 * m1._11 + m2._22 * m1._12 + m2._32 * m1._13 + m2._42 * m1._14,
		m2._13 * m1._11 + m2._23 * m1._12 + m2._33 * m1._13 + m2._43 * m1._14,
		m2._14 * m1._11 + m2._24 * m1._12 + m2._34 * m1._13 + m2._44 * m1._14,

		m2._11 * m1._21 + m2._21 * m1._22 + m2._31 * m1._23 + m2._41 * m1._24,
		m2._12 * m1._21 + m2._22 * m1._22 + m2._32 * m1._23 + m2._42 * m1._24,
		m2._13 * m1._21 + m2._23 * m1._22 + m2._33 * m1._23 + m2._43 * m1._24,
		m2._14 * m1._21 + m2._24 * m1._22 + m2._34 * m1._23 + m2._44 * m1._24,

		m2._11 * m1._31 + m2._21 * m1._32 + m2._31 * m1._33 + m2._41 * m1._34,
		m2._12 * m1._31 + m2._22 * m1._32 + m2._32 * m1._33 + m2._42 * m1._34,
		m2._13 * m1._31 + m2._23 * m1._32 + m2._33 * m1._33 + m2._43 * m1._34,
		m2._14 * m1._31 + m2._24 * m1._32 + m2._34 * m1._33 + m2._44 * m1._34,

		m2._11 * m1._41 + m2._21 * m1._42 + m2._31 * m1._43 + m2._41 * m1._44,
		m2._12 * m1._41 + m2._22 * m1._42 + m2._32 * m1._43 + m2._42 * m1._44,
		m2._13 * m1._41 + m2._23 * m1._42 + m2._33 * m1._43 + m2._43 * m1._44,
		m2._14 * m1._41 + m2._24 * m1._42 + m2._34 * m1._43 + m2._44 * m1._44
		);
}

#ifndef WIN64

// Divides a Matrix through a float
CoreMatrix4x4 CoreMatrix4x4DivFloat_3DNow(CoreMatrix4x4& mIn, float fIn)
{
	CoreMatrix4x4 mOut;
	_asm
	{
		femms

		mov eax, mIn
		mov ebx, fIn
		lea ecx, mOut

		prefetch [eax]
		
		movd mm1, [ebx]
		punpckldq mm1, mm1
		pfrcp mm2, mm1;
		pfrcpit1 mm1, mm2;
		pfrcpit2 mm1, mm2;
		
		movq mm0, [eax]
		pfmul mm0, mm1
		movq [ecx], mm0

		movq mm0, [eax+8]
		pfmul mm0, mm1
		movq [ecx+8], mm0

		movq mm0, [eax+16]
		pfmul mm0, mm1
		movq [ecx+16], mm0

		movq mm0, [eax+24]
		pfmul mm0, mm1
		movq [ecx+24], mm0

		movq mm0, [eax+32]
		pfmul mm0, mm1
		movq [ecx+32], mm0

		movq mm0, [eax+40]
		pfmul mm0, mm1
		movq [ecx+40], mm0

		movq mm0, [eax+48]
		pfmul mm0, mm1
		movq [ecx+48], mm0

		movq mm0, [eax+56]
		pfmul mm0, mm1
		movq [ecx+56], mm0
		
		femms
	}
	return mOut;
}
#endif

// Divides a Matrix with a float
CoreMatrix4x4 CoreMatrix4x4DivFloat_Normal(CoreMatrix4x4& mIn, float fIn)
{
	return CoreMatrix4x4(mIn._11 / fIn, mIn._12 / fIn, mIn._13 / fIn, mIn._14 / fIn,
					mIn._21 / fIn, mIn._22 / fIn, mIn._23 / fIn, mIn._24 / fIn,
					mIn._31 / fIn, mIn._32 / fIn, mIn._33 / fIn, mIn._34 / fIn,
					mIn._41 / fIn, mIn._42 / fIn, mIn._43 / fIn, mIn._44 / fIn);
}


// Translation
CoreMatrix4x4 CoreMatrix4x4Translation(CoreVector3& vIn)
{
	return CoreMatrix4x4(1,   0,   0,   0,
					0,   1,   0,   0,
					0,   0,   1,   0,
					vIn.x, vIn.y, vIn.z, 1);
}

// Scaling
CoreMatrix4x4 CoreMatrix4x4Scale(CoreVector3& vIn)
{
	return CoreMatrix4x4(vIn.x,   0,   0,   0,
					0,   vIn.y,   0,   0,
					0,   0,   vIn.z,   0,
					0,   0,     0,   1);
}


// Rotation X
CoreMatrix4x4 CoreMatrix4x4RotationX(float alpha)
{
	return CoreMatrix4x4(1,             0,              0,   0,
					0,    cosf(alpha),   sinf(alpha),   0,
					0,   -sinf(alpha),   cosf(alpha),   0,
					0,              0,             0,   1);
}

// Rotation Y
CoreMatrix4x4 CoreMatrix4x4RotationY(float alpha)
{
	return CoreMatrix4x4(cosf(alpha),   0,    -sinf(alpha),   0,
					0,             1,               0,   0,
					sinf(alpha),   0,     cosf(alpha),   0,
					0,             0,             0,     1);
}

// Rotation Z
CoreMatrix4x4 CoreMatrix4x4RotationZ(float alpha)
{
	return CoreMatrix4x4(cosf(alpha),  sinf(alpha),    0,   0,
					-sinf(alpha),  cosf(alpha),    0,   0,
					 0,             0,             1,   0,
					 0,             0,             0,   1);
}


// Rotation
CoreMatrix4x4 CoreMatrix4x4Rotation(CoreVector3& alpha)
{
	return CoreMatrix4x4RotationZ(alpha.z) * CoreMatrix4x4RotationX(alpha.x) * CoreMatrix4x4RotationY(alpha.y);
}

//Rotation um bestimmte Axe
CoreMatrix4x4 CoreMatrix4x4RotationAxis(CoreVector3& vIn, float fIn) 
{
	float fSin(sinf(-fIn));
	float fCos(cosf(-fIn));

	CoreVector3 vAxis(CoreVector3Normalize(vIn));

	return CoreMatrix4x4((vAxis.x * vAxis.x) * (1.0f - fCos) + fCos,
		            (vAxis.x * vAxis.y) * (1.0f - fCos) - (vAxis.z * fSin),
				    (vAxis.x * vAxis.z) * (1.0f - fCos) + (vAxis.y * fSin),
					0.0f,
					(vAxis.y * vAxis.x) * (1.0f - fCos) + (vAxis.z * fSin),
					(vAxis.y * vAxis.y) * (1.0f - fCos) + fCos,
					(vAxis.y * vAxis.z) * (1.0f - fCos) - (vAxis.x * fSin),
					0.0f,
					(vAxis.z * vAxis.x) * (1.0f - fCos) - (vAxis.y * fSin),
					(vAxis.z * vAxis.y) * (1.0f - fCos) + (vAxis.x * fSin),
					(vAxis.z * vAxis.z) * (1.0f - fCos) + fCos,
					0.0f,
					0.0f,
					0.0f,
					0.0f,
					1.0f);
}

// Axis Matrix
CoreMatrix4x4 CoreMatrix4x4Axes(CoreVector3& vXAxis, CoreVector3& vYAxis, CoreVector3& vZAxis)
{
   return CoreMatrix4x4(vXAxis.x, vYAxis.x, vZAxis.x, 0.0f,
			    	vXAxis.y, vYAxis.y, vZAxis.y, 0.0f,
			    	vXAxis.z, vYAxis.z, vZAxis.z, 0.0f,
			    	0.0f,     0.0f,     0.0f,     1.0f);
}


// Calcs the Determinante of a Matrix
float CoreMatrix4x4Determinante(CoreMatrix4x4& mIn)
{
	// Cramer's Rule
	return (mIn._11 * mIn._22 - mIn._21 * mIn._12) * (mIn._33 * mIn._44 - mIn._43 * mIn._34)	- (mIn._11 * mIn._32 - mIn._31 * mIn._12) * (mIn._23 * mIn._44 - mIn._43 * mIn._24) +
		   (mIn._11 * mIn._42 - mIn._41 * mIn._12) * (mIn._23 * mIn._34 - mIn._33 * mIn._24)	+ (mIn._21 * mIn._32 - mIn._31 * mIn._22) * (mIn._13 * mIn._44 - mIn._43 * mIn._14)	-
		   (mIn._21 * mIn._42 - mIn._41 * mIn._22) * (mIn._13 * mIn._34 - mIn._33 * mIn._14)	+ (mIn._31 * mIn._42 - mIn._41 * mIn._32) * (mIn._13 * mIn._24 - mIn._23 * mIn._14);
}


// Inverts the Matrix
CoreMatrix4x4 CoreMatrix4x4Invert(CoreMatrix4x4& mIn)
{
	// Calculation with Cramer's Rule
	
	float fInvDet(CoreMatrix4x4Determinante(mIn));
	if(fInvDet == 0.0f)
	{
		return CoreMatrix4x4();
	}
		
	fInvDet = 1.0f / fInvDet;

	CoreMatrix4x4 mOut;

	mOut._11 = fInvDet * (mIn._22 * (mIn._33 * mIn._44 - mIn._43 * mIn._34) + mIn._32 * (mIn._43 * mIn._24 - mIn._23 * mIn._44) + mIn._42 * (mIn._23 * mIn._34 - mIn._33 * mIn._24));
	mOut._21 = fInvDet * (mIn._23 * (mIn._31 * mIn._44 - mIn._41 * mIn._34) + mIn._33 * (mIn._41 * mIn._24 - mIn._21 * mIn._44) + mIn._43 * (mIn._21 * mIn._34 - mIn._31 * mIn._24));
	mOut._31 = fInvDet * (mIn._24 * (mIn._31 * mIn._42 - mIn._41 * mIn._32) + mIn._34 * (mIn._41 * mIn._22 - mIn._21 * mIn._42) + mIn._44 * (mIn._21 * mIn._32 - mIn._31 * mIn._22));
	mOut._41 = fInvDet * (mIn._21 * (mIn._42 * mIn._33 - mIn._32 * mIn._43) + mIn._31 * (mIn._22 * mIn._43 - mIn._42 * mIn._23) + mIn._41 * (mIn._32 * mIn._23 - mIn._22 * mIn._33));
	mOut._12 = fInvDet * (mIn._32 * (mIn._13 * mIn._44 - mIn._43 * mIn._14) + mIn._42 * (mIn._33 * mIn._14 - mIn._13 * mIn._34) + mIn._12 * (mIn._43 * mIn._34 - mIn._33 * mIn._44));
	mOut._22 = fInvDet * (mIn._33 * (mIn._11 * mIn._44 - mIn._41 * mIn._14) + mIn._43 * (mIn._31 * mIn._14 - mIn._11 * mIn._34) + mIn._13 * (mIn._41 * mIn._34 - mIn._31 * mIn._44));
	mOut._32 = fInvDet * (mIn._34 * (mIn._11 * mIn._42 - mIn._41 * mIn._12) + mIn._44 * (mIn._31 * mIn._12 - mIn._11 * mIn._32) + mIn._14 * (mIn._41 * mIn._32 - mIn._31 * mIn._42));
	mOut._42 = fInvDet * (mIn._31 * (mIn._42 * mIn._13 - mIn._12 * mIn._43) + mIn._41 * (mIn._12 * mIn._33 - mIn._32 * mIn._13) + mIn._11 * (mIn._32 * mIn._43 - mIn._42 * mIn._33));
	mOut._13 = fInvDet * (mIn._42 * (mIn._13 * mIn._24 - mIn._23 * mIn._14) + mIn._12 * (mIn._23 * mIn._44 - mIn._43 * mIn._24) + mIn._22 * (mIn._43 * mIn._14 - mIn._13 * mIn._44));
	mOut._23 = fInvDet * (mIn._43 * (mIn._11 * mIn._24 - mIn._21 * mIn._14) + mIn._13 * (mIn._21 * mIn._44 - mIn._41 * mIn._24) + mIn._23 * (mIn._41 * mIn._14 - mIn._11 * mIn._44));
	mOut._33 = fInvDet * (mIn._44 * (mIn._11 * mIn._22 - mIn._21 * mIn._12) + mIn._14 * (mIn._21 * mIn._42 - mIn._41 * mIn._22) + mIn._24 * (mIn._41 * mIn._12 - mIn._11 * mIn._42));
	mOut._43 = fInvDet * (mIn._41 * (mIn._22 * mIn._13 - mIn._12 * mIn._23) + mIn._11 * (mIn._42 * mIn._23 - mIn._22 * mIn._43) + mIn._21 * (mIn._12 * mIn._43 - mIn._42 * mIn._13));
	mOut._14 = fInvDet * (mIn._12 * (mIn._33 * mIn._24 - mIn._23 * mIn._34) + mIn._22 * (mIn._13 * mIn._34 - mIn._33 * mIn._14) + mIn._32 * (mIn._23 * mIn._14 - mIn._13 * mIn._24));
	mOut._24 = fInvDet * (mIn._13 * (mIn._31 * mIn._24 - mIn._21 * mIn._34) + mIn._23 * (mIn._11 * mIn._34 - mIn._31 * mIn._14) + mIn._33 * (mIn._21 * mIn._14 - mIn._11 * mIn._24));
	mOut._34 = fInvDet * (mIn._14 * (mIn._31 * mIn._22 - mIn._21 * mIn._32) + mIn._24 * (mIn._11 * mIn._32 - mIn._31 * mIn._12) + mIn._34 * (mIn._21 * mIn._12 - mIn._11 * mIn._22));
	mOut._44 = fInvDet * (mIn._11 * (mIn._22 * mIn._33 - mIn._32 * mIn._23) + mIn._21 * (mIn._32 * mIn._13 - mIn._12 * mIn._33) + mIn._31 * (mIn._12 * mIn._23 - mIn._22 * mIn._13));

	return mOut;
}

// Transposes the Matrix
CoreMatrix4x4 CoreMatrix4x4Transpose(CoreMatrix4x4& m)
{
	return CoreMatrix4x4(m._11, m._21, m._31, m._41,
					m._12, m._22, m._32, m._42,
					m._13, m._23, m._33, m._43,
					m._14, m._24, m._34, m._44);
}

// Negates the Matrix
CoreMatrix4x4 CoreMatrix4x4Negate(CoreMatrix4x4& mIn)
{
	return CoreMatrix4x4(-mIn._11, -mIn._12, -mIn._13, -mIn._14, 
		             -mIn._21, -mIn._22, -mIn._23, -mIn._24, 
					 -mIn._31, -mIn._32, -mIn._33, -mIn._34, 
					 -mIn._41, -mIn._42, -mIn._43, -mIn._44);	
}

// Returns the Trace of the Matrix
float CoreMatrix4x4Trace(CoreMatrix4x4& mIn)
{
	return mIn._11 + mIn._22 + mIn._33 + mIn._44;
}

