#include "Core.h"

// Init function pointers
float (*CoreVector3Length)(CoreVector3& vIn) = &CoreVector3Length_Normal;
float (*CoreVector3LengthSq)(CoreVector3& vIn) = &CoreVector3LengthSq_Normal;
CoreVector3 (*CoreVector3Cross)(CoreVector3& v1, CoreVector3& v2) = &CoreVector3Cross_Normal;
float (*CoreVector3Dot)(CoreVector3& v1, CoreVector3& v2) = &CoreVector3Dot_Normal;
CoreVector3 (*CoreVector3Normalize)(CoreVector3& vIn) = &CoreVector3Normalize_Normal;
CoreVector3 (*CoreVector3TransformCoords)(CoreVector3& vIn, CoreMatrix4x4& mIn) = &CoreVector3TransformCoords_Normal;
CoreVector3 (*CoreVector3TransformNormal)(CoreVector3& vIn, CoreMatrix4x4& mIn) = &CoreVector3TransformNormal_Normal;


// Adds 2 vectors
CoreVector3 CoreVector3Add(CoreVector3& v1, CoreVector3& v2)
{
	return CoreVector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

// Subtracts 2 vectors
CoreVector3 CoreVector3Sub(CoreVector3& v1, CoreVector3& v2)
{
	return CoreVector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

// Multiplies 2 vectors
CoreVector3 CoreVector3Mul(CoreVector3& v1, CoreVector3& v2)
{
	return CoreVector3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

// Multiplies a vector with a float
CoreVector3 CoreVector3Mul(CoreVector3& vIn, float fIn)
{
	return CoreVector3(vIn.x * fIn, vIn.y * fIn, vIn.z * fIn);
}

// Divides 2 vectors
CoreVector3 CoreVector3Div(CoreVector3& v1, CoreVector3& v2)
{
	return CoreVector3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

// Divides a vector through a float
CoreVector3 CoreVector3Div(CoreVector3& vIn, float fIn)
{
	return CoreVector3(vIn.x / fIn, vIn.y / fIn, vIn.z / fIn);
}

// Negates a vector
CoreVector3 CoreVector3Negate(CoreVector3& vIn)
{
	return CoreVector3(-vIn.x, -vIn.y, -vIn.z);
}

#ifndef WIN64
// Transform a position-vector
CoreVector3 CoreVector3TransformCoords_3DNow(CoreVector3& vIn, CoreMatrix4x4& mIn)
{
   CoreVector3 vOut;
  _asm
  {
    femms

    // init
    mov ebx, vIn
    mov ecx, mIn
    prefetch [ebx]
    prefetchw [ecx]

    movq mm6, [ebx]                 // X,Y multiplication
    movd mm7,[ebx+8]

    movd mm0, [ecx]
    movd mm1, [ecx+16]
    punpckldq mm0, mm1
    pfmul mm0, mm6

    movd mm1, [ecx+4]
    movd mm2, [ecx+20]
    punpckldq mm1, mm2
    pfmul mm1, mm6

    movd mm2, [ecx+8]
    movd mm3, [ecx+24]
    punpckldq mm2, mm3
    pfmul mm2, mm6

    movd mm3, [ecx+12]
    movd mm4, [ecx+28]
    punpckldq mm3, mm4
    pfmul mm3, mm6

    movd mm4, [ecx+32]           // Z multiplication W addition
    pfmul mm4, mm7
    movd mm5, [ecx+48]
    punpckldq mm4, mm5

    movd mm5, [ecx+36]
    pfmul mm5, mm7
    movd mm6, [ecx+52]
    punpckldq mm5, mm6

    pfacc mm0, mm4
    pfacc mm1, mm5

    movd mm6, [ecx+40]
    pfmul mm6, mm7
    movd mm4, [ecx+56]
    punpckldq mm6, mm4

    movd mm4, [ecx+44]
    pfmul mm4, mm7
    movd mm5, [ecx+60]
    punpckldq mm4, mm5

    pfacc mm2, mm6
    pfacc mm3, mm4
    pfacc mm0, mm0
    pfacc mm1, mm1
    pfacc mm2, mm2
    pfacc mm3, mm3
	
    
	movd eax,mm3				// round
	cmp eax, 0x3f800000			// 1.0f
	je _end
	pfrcp mm4, mm3				// Reciprocal  24 Bit divide
	pfrcpit1 mm3, mm4
    pfrcpit2 mm3, mm4
	pfmul mm0, mm3
	pfmul mm1, mm3
	pfmul mm2, mm3

_end:
	lea eax, vOut
    movd [eax], mm0           // write back
    movd [eax+4], mm1
    movd [eax+8], mm2
	

    femms
  }
  return vOut;
}
#endif
// Transform a position-vector
CoreVector3 CoreVector3TransformCoords_Normal(CoreVector3& vIn, CoreMatrix4x4& mIn) 
{
	 CoreVector3 vOut(	vIn.x * mIn._11 + vIn.y * mIn._21 + vIn.z * mIn._31 + mIn._41,
						vIn.x * mIn._12 + vIn.y * mIn._22 + vIn.z * mIn._32 + mIn._42,
						vIn.x * mIn._13 + vIn.y * mIn._23 + vIn.z * mIn._33 + mIn._43);
	 float w = vIn.x * mIn._14 + vIn.y * mIn._24 + vIn.z * mIn._34 + mIn._44; 
	 if(fabs(w - 1.0f) > 0.0001f)
		 vOut /= w;
	return vOut;
	
}

#ifndef WIN64
// Transform a direction-vector
CoreVector3 CoreVector3TransformNormal_3DNow(CoreVector3& vIn, CoreMatrix4x4& mIn)
{
  CoreVector3 vOut;
  _asm
  {
    femms

    lea eax, vOut                 // init
    mov ebx, vIn
    mov ecx, mIn
    prefetch [ecx]
    prefetchw [eax]

    movq mm6, [ebx]                 // X,Y multiplication
    movd mm7,[ebx+8]

    movd mm0, [ecx]
    movd mm1, [ecx+16]
    punpckldq mm0, mm1
    pfmul mm0, mm6

    movd mm1, [ecx+4]
    movd mm2, [ecx+20]
    punpckldq mm1, mm2
    pfmul mm1, mm6

    movd mm2, [ecx+8]
    movd mm3, [ecx+24]
    punpckldq mm2, mm3
    pfmul mm2, mm6


    movd mm3, [ecx+32]           // Z multiplication W addition
    pfmul mm3, mm7

    movd mm4, [ecx+36]
    pfmul mm4, mm7

    movd mm5, [ecx+40]
    pfmul mm5, mm7


	pfacc mm0, mm3
    pfacc mm1, mm4
    pfacc mm2, mm5
    pfacc mm0, mm0
    pfacc mm1, mm1
    pfacc mm2, mm2

    movd [eax], mm0           // write back
    movd [eax+4], mm1
    movd [eax+8], mm2
	

    femms
  }
}

#endif
// Transform a direction-vector
CoreVector3 CoreVector3TransformNormal_Normal(CoreVector3& vIn, CoreMatrix4x4& mIn) 
{
	 CoreVector3 vOut(	vIn.x * mIn._11 + vIn.y * mIn._21 + vIn.z * mIn._31,		// w = 0
						vIn.x * mIn._12 + vIn.y * mIn._22 + vIn.z * mIn._32,
						vIn.x * mIn._13 + vIn.y * mIn._23 + vIn.z * mIn._33);

	return vOut;
}


#ifndef WIN64
// Length of a vector
float CoreVector3Length_3DNow(CoreVector3& vIn)						
{
	float fOut;
	_asm
	{
      femms
      mov eax, vIn

      movq mm0, [eax]			// multiply X,Y
      pfmul mm0, mm0
	  pfacc mm0, mm0

      movd mm1, [eax+8]			// multiply Z
      pfmul mm1, mm1
	  pfadd mm0, mm1
	
    
      pfrsqrt mm1, mm0				// sqrt
      pfmul mm0, mm1

      movd fOut, mm0

      femms	
	}
	return fOut;
}


#endif
// Length of a vector
float CoreVector3Length_Normal(CoreVector3& vIn)
{
	return sqrtf(vIn.x * vIn.x + vIn.y * vIn.y + vIn.z * vIn.z);
}

#ifndef WIN64
// Length of a vector Squared
float CoreVector3LengthSq_3DNow(CoreVector3& vIn)						
{
	float fOut;
	_asm
	{
      femms
      mov eax, vIn

	  movq mm0, [eax]				// multiply X,Y
	  pfmul mm0, mm0
      pfacc mm0,mm0


	  movd mm1, [eax+8]			// multiply Z
	  pfmul mm1, mm1
      pfadd mm0,mm1

      movd fOut, mm0

      femms
	}
	return fOut;
}
#endif

// Length of a Vector Squared
float CoreVector3LengthSq_Normal(CoreVector3& vIn)
{
	return vIn.x * vIn.x + vIn.y * vIn.y + vIn.z * vIn.z;
}

#ifndef WIN64
// Crossproduct of 2 vectors
CoreVector3 CoreVector3Cross_3DNow(CoreVector3& v1, CoreVector3& v2)	
{
	CoreVector3 vOut;
	_asm
	{
      	femms

		mov eax, v1
		mov ebx, v2
		lea ecx, vOut


		movq mm0, [eax+4]				// v1.y * v2.z - v1.z * v2.y
		movq mm1, [ebx+4]

		pswapd mm1, mm1

		pfmul mm0,mm1
		pfnacc mm0,mm0

		movd mm1, [eax+8]				// v1.z * v2.x - v1.x * v2.z
		movd mm2, [eax]
		punpckldq mm1,mm2

		movd mm2, [ebx]
		movd mm3, [ebx+8]
		punpckldq mm2,mm3

		pfmul mm1,mm2
		pfnacc mm1,mm1

		movq mm2, [eax]				// v1.x * v2.y - v1.y * v2.x
		movq mm3, [ebx]
		pswapd mm3, mm3
		pfmul mm2, mm3
		pfnacc mm2, mm2

		movd [ecx], mm0
		movd [ecx+4], mm1
		movd [ecx+8], mm2

		femms
	}
	return vOut;
}
#endif

// Crossproduct of 2 vectors
CoreVector3 CoreVector3Cross_Normal(CoreVector3& v1, CoreVector3& v2)	
{
	return CoreVector3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

#ifndef WIN64
// Dotproduct of 2 Vectors
float CoreVector3Dot_3DNow(CoreVector3& v1, CoreVector3& v2)
{
	float fOut;
	_asm
	{
		femms
		
		mov eax, v1
		mov ebx, v2

		movq mm0, [eax]				//X,Y
		movq mm1, [ebx]
		
		pfmul mm0,mm1

		movd mm2, [eax+8]		//Z
		movd mm3, [ebx+8]

		pfmul mm2, mm3

		pfadd mm0, mm2
		pfacc mm0, mm0
		movd fOut, mm0

		femms
	}
	return fOut;
}
#endif
// Dotproduct of 2 vectors
float CoreVector3Dot_Normal(CoreVector3& v1, CoreVector3& v2)	
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

#ifndef WIN64
// Normalizes the vector
CoreVector3 CoreVector3Normalize_3DNow(CoreVector3& vIn)
{
	CoreVector3 vOut;
	_asm
	{
		femms
		
		mov eax, vIn
		lea ebx, vOut

		movq mm0, [eax]		// Calc length
		pfmul mm0, mm0
		pfacc mm0, mm0

		movd mm1, [eax+8]
		pfmul mm1, mm1
		pfadd mm0, mm1

		pfrsqrt mm1, mm0
		pfmul mm0, mm1

		movq mm3, [eax]
		movd mm4, [eax+8]

		pfrcp mm1, mm0				// Divide
		pfmul mm3, mm1
		pfmul mm4, mm1

		movq [ebx], mm3
		movd [ebx+8], mm4

		femms
	}
	return vOut;
}
#endif
// Normalizes the vector
CoreVector3 CoreVector3Normalize_Normal(CoreVector3& vIn)
{																				
	return vIn / sqrtf(vIn.x * vIn.x + vIn.y * vIn.y + vIn.z * vIn.z);
}