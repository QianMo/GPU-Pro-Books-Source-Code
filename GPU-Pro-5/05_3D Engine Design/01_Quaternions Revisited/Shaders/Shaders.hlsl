/****************************************************************************

  GPU Pro 5 : Quaternions revisited - sample code
  All sample code written from scratch by Sergey Makeev specially for article.

  Copyright (c) 2013, Sergey Makeev

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.

  2. If you use this software in a non-commercial product, an acknowledgment
     in the product documentation would be appreciated but is not required.

  3. If you use this software in a commercial product you requried to credit
     the author.

  4. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

  5. This notice may not be removed or altered from any source distribution.
    

   Please let me know if you use the code in your products or have any questions or suggestions.

   e-mail: sergeymakeev@inbox.ru
   http://linkedin.com/in/sergeymakeev/

****************************************************************************/

float4x4 viewProj;

sampler albedo : register(s1);
sampler normalMap : register(s2);
sampler floorMap : register(s3);
samplerCUBE ambientCube : register(s4);

#if !defined(UNPACKED_TBN) && !defined(PACKED_TBN) && !defined(UNPACKED_QUATERNIONS) && !defined(PACKED_QUATERNIONS)
#define PACKED_QUATERNIONS (1)
#endif



sampler animationData : register(s0);

//xy = uv in animation data texture, zw = one texel size
float4 animationDataParameters;

// selectorConstant must be an identity matrix packed into four vectors.
// Note: if this selectorConstant is made static const, the hlsl compiler fails.
float4x4 selectorConstant;

//light direction in world space
float4 lightDirection;

#ifdef UNPACKED_TBN
struct MeshVSInput
{
	float3 p : POSITION;
	float3 n : NORMAL;
	float3 t : TANGENT;
	float3 b : BINORMAL;
	float2 uv : TEXCOORD0;
	float4 blendWeights : BLENDWEIGHT;
	float4 blendIndices : BLENDINDICES;

	//-------------------- instance data ---------------------------
	float4 instance_Quaternion : TEXCOORD1;
	float4 instance_PositionAndScaleX : TEXCOORD2;
	float2 instance_ScaleYZ : TEXCOORD3;
	float4 instance_ColorAndID : COLOR0;
};
#endif

#ifdef PACKED_TBN
struct MeshVSInput
{
	float3 p : POSITION;
	float4 packedNormal : NORMAL;
	float4 packedTangent : TANGENT;
	float4 packedBinormal : BINORMAL;
	float2 uv : TEXCOORD0;
	float4 blendWeights : BLENDWEIGHT;
	float4 blendIndices : BLENDINDICES;

	//-------------------- instance data ---------------------------
	float4 instance_Quaternion : TEXCOORD1;
	float4 instance_PositionAndScaleX : TEXCOORD2;
	float2 instance_ScaleYZ : TEXCOORD3;
	float4 instance_ColorAndID : COLOR0;
};
#endif

#ifdef UNPACKED_QUATERNIONS
struct MeshVSInput
{
	float3 p : POSITION;
	float4 quaternionTBN : TEXCOORD4;
	float2 uv : TEXCOORD0;
	float4 blendWeights : BLENDWEIGHT;
	float4 blendIndices : BLENDINDICES;

	//-------------------- instance data ---------------------------
	float4 instance_Quaternion : TEXCOORD1;
	float4 instance_PositionAndScaleX : TEXCOORD2;
	float2 instance_ScaleYZ : TEXCOORD3;
	float4 instance_ColorAndID : COLOR0;
};
#endif

#ifdef PACKED_QUATERNIONS
struct MeshVSInput
{
	float3 p : POSITION;
	float4 packedQuaternionTBN : TEXCOORD4;
	float2 uv : TEXCOORD0;
	float4 blendWeights : BLENDWEIGHT;
	float4 blendIndices : BLENDINDICES;

	//-------------------- instance data ---------------------------
	float4 instance_Quaternion : TEXCOORD1;
	float4 instance_PositionAndScaleX : TEXCOORD2;
	float2 instance_ScaleYZ : TEXCOORD3;
	float4 instance_ColorAndID : COLOR0;
};
#endif

struct MeshVSOutput
{
	float4 p : POSITION;
	float2 uv : TEXCOORD0;

#if defined(UNPACKED_QUATERNIONS) || defined(PACKED_QUATERNIONS)
	float4 quadTBN : TEXCOORD1;
	float handednessTBN : TEXCOORD2;
#else
	float3 n : TEXCOORD1;
	float3 t : TEXCOORD2;
	float3 b : TEXCOORD3;
#endif
};

struct MeshPSInput
{
	float2 uv : TEXCOORD0;

#if defined(UNPACKED_QUATERNIONS) || defined(PACKED_QUATERNIONS)
	float4 quadTBN : TEXCOORD1;
	float handednessTBN : TEXCOORD2;
#else
	float3 n : TEXCOORD1;
	float3 t : TEXCOORD2;
	float3 b : TEXCOORD3;
#endif
};


float4 UnpackHighBitFromByte( float4 argument )
{
	float v = (argument * 255.0 - 127.5);
	return saturate(v * 100.0);
}

float4 UnpackLast7BitsFromByte( float4 argument, float4 highBit )
{
	return (argument * 255.0 - 128.0 * highBit) / 127.0;
}

float2 UnpackUV(float2 uv)
{
	return uv * 8.0 / 32767.0;
}

float4 UnpackWeights(float4 w)
{
	float4 r = (w * (0.5 / 32767.0)) + 0.5;
	return r;
}

float4 MulQuaternions( float4 q1, float4 q2 )
{
	float4 res;
	res.w = q1.w * q2.w - dot( q1.xyz, q2.xyz );
	res.xyz = q1.w * q2.xyz + q2.w * q1.xyz + cross( q1.xyz, q2.xyz );
	return res;
}

float3 MulQuaternionVector( in float4 q, in float3 v )
{
	float3 t = 2.0 * cross( q.xyz, v );
	return v + q.w * t + cross( q.xyz, t );
}

float3 TransformPosition( in float3 v, in float4 q, in float3 t )
{
	return MulQuaternionVector( q, v ) + t.xyz;
}

MeshVSOutput MeshVS (MeshVSInput v)
{
	float4 rotate = v.instance_Quaternion;
	float3 translate = v.instance_PositionAndScaleX.xyz;
	float3 scale = float3 (v.instance_PositionAndScaleX.w, v.instance_ScaleYZ.x, v.instance_ScaleYZ.y);

	float3 instanceColor = v.instance_ColorAndID.rgb;
	float instanceID = v.instance_ColorAndID.w * 255.0f;

	MeshVSOutput o;
	float3 pos = v.p;

#if defined(UNPACKED_QUATERNIONS) || defined(PACKED_QUATERNIONS)

#ifdef UNPACKED_QUATERNIONS
	float4 quatTBN = v.quaternionTBN;
	float4 highBit = UnpackHighBitFromByte( quatTBN.wwww );
	quatTBN.w = UnpackLast7BitsFromByte( quatTBN.wwww, highBit ).x * 2.0 - 1.0;
#else
	float4 quatTBN = v.packedQuaternionTBN;
	float4 highBit = UnpackHighBitFromByte( quatTBN.wwww );
	quatTBN.w = UnpackLast7BitsFromByte( quatTBN.wwww, highBit ).x;
	quatTBN = quatTBN * 2.0 - 1.0;
#endif
	float4 weightedBonesQuat = float4( 0, 0, 0, 0 );
	
#else

	float3 skinnedT = float3( 0, 0, 0 );
	float3 skinnedB = float3( 0, 0, 0 );
	float3 skinnedN = float3( 0, 0, 0 );

#ifdef UNPACKED_TBN
	float3 inputT = v.t;
	float3 inputB = v.b;
	float3 inputN = v.n;
#else
	float3 inputT = v.packedTangent.xyz * 2.0 - 1.0;
	float3 inputB = v.packedBinormal.xyz * 2.0 - 1.0;
	float3 inputN = v.packedNormal.xyz * 2.0 - 1.0;
#endif

#endif

	float3 finalVertexPos = float3( 0, 0, 0 );


#ifndef DISABLE_SKINING

	//enabled skining codepath
	////////////////////////////////////////////////////////////////////////////////////

	float4 blendIndices = v.blendIndices;

#ifdef UNPACKED_TBN
	float4 blendWeights = v.blendWeights;
#else
	float4 blendWeights = UnpackWeights(v.blendWeights);
#endif

	for (int i = 0; i < 4; i++)
	{
		// cannot simply index the 'indices' or 'weights', because only constant
		// registers (and not vertex attributes) can be indexed through the address register
		float blendIndice = dot( blendIndices, selectorConstant[i] );  
		float blendWeight = dot( blendWeights, selectorConstant[i] );

		float4 boneT = (animationDataParameters.xy + animationDataParameters.zw * float2(blendIndice * 2 + 0, instanceID)).xyyy;
		float4 boneQ = (animationDataParameters.xy + animationDataParameters.zw * float2(blendIndice * 2 + 1, instanceID)).xyyy;

		float4 t = tex2Dlod(animationData, boneT);
		float4 q = tex2Dlod(animationData, boneQ);

		finalVertexPos += TransformPosition( pos.xyz, q, t.xyz ) * blendWeight;

#if defined(UNPACKED_QUATERNIONS) || defined(PACKED_QUATERNIONS)
		weightedBonesQuat += q * blendWeight;
#else
		skinnedT += MulQuaternionVector(q, inputT)* blendWeight;
		skinnedB += MulQuaternionVector(q, inputB)* blendWeight;
		skinnedN += MulQuaternionVector(q, inputN)* blendWeight;
#endif
	}

#if defined(UNPACKED_QUATERNIONS) || defined(PACKED_QUATERNIONS)
	float4 skinnedQuatTBN = MulQuaternions( normalize(weightedBonesQuat), quatTBN );
	o.quadTBN = normalize( skinnedQuatTBN );
	o.handednessTBN = highBit.x * 2.0 - 1.0;
#else
	o.t = skinnedT;
	o.b = skinnedB;
	o.n = skinnedN;
#endif

#else
	//disabled skining codepath
	////////////////////////////////////////////////////////////////////////////////////

	finalVertexPos = pos.xyz;

#if defined(UNPACKED_QUATERNIONS) || defined(PACKED_QUATERNIONS)
	o.quadTBN = quatTBN;
	o.handednessTBN = highBit.x * 2.0 - 1.0;
#else
	o.t = inputT;
	o.b = inputB;
	o.n = inputN;
#endif


#endif



#ifdef UNPACKED_TBN
	o.uv = v.uv;
#else
	o.uv = UnpackUV(v.uv);
#endif

	float3 skinnedWorldPos = MulQuaternionVector( rotate, finalVertexPos * scale ) + translate;
	o.p = mul(float4(skinnedWorldPos, 1.0), viewProj);

	return o;
};

float4 MeshPS (MeshPSInput p) : COLOR 
{
	float2 uv = p.uv;
	float4 texelAlbedo = tex2D(albedo, uv);
	float3 texelNormalMap = tex2D(normalMap, uv).rgb;
	float3 tangentNormal = texelNormalMap * 2.0 - 1.0;

#if defined(UNPACKED_QUATERNIONS) || defined(PACKED_QUATERNIONS)
	float handednessTBN = p.handednessTBN;
	float4 quadTBN = p.quadTBN;
	tangentNormal.y *= handednessTBN;
	float3 worldNormal = MulQuaternionVector( normalize(quadTBN), tangentNormal );
#else
	float3 tangent = normalize(p.t);
	float3 binormal = normalize(p.b);
	float3 normal = normalize(p.n);

	float3 worldNormal = (tangentNormal.x * tangent ) + (tangentNormal.y * binormal) + (tangentNormal.z * normal);
#endif

	worldNormal = normalize(worldNormal);

#ifdef SHOW_NORMALS
	return float4(worldNormal.xyz * 0.5 + 0.5, 1.0);
#endif

	float diffuse = saturate(dot(worldNormal, -lightDirection.xyz));
	float3 ambient = texCUBE(ambientCube, worldNormal).rgb * 0.85;
	float3 l = ambient + diffuse;
	return float4 (l * texelAlbedo.xyz, texelAlbedo.w);
}



struct FloorVSInput
{
	float3 p : POSITION;
};

struct FloorVSOutput
{
	float4 p : POSITION;
	float2 uv : TEXCOORD0;
	float3 vertexPos : TEXCOORD1;

};

struct FloorPSInput
{
	float2 uv : TEXCOORD0;
	float3 vertexPos : TEXCOORD1;
};

FloorVSOutput FloorVS (FloorVSInput v)
{
	const float floorScale = 13.0;
	FloorVSOutput o;
	float3 p = v.p * floorScale;
	float4 pos = float4(p, 1.0);
	o.p = mul(pos, viewProj);
	o.uv = p.xz;
	o.vertexPos = p;
	return o;
};

float4 FloorPS (FloorPSInput p) : COLOR 
{
	float3 texelFloor = tex2D(floorMap, p.uv).rgb;
	return texelFloor.xyzz;
}

