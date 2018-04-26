//------------------------------------------------
// CloudGrid.hlsl
//
// Shader to render cloud grid
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//------------------------------------------------


//------------------------------------------------
// input of the vertex shader
//------------------------------------------------
struct SVSInput {
	float4 vPos : POSITION0;
};

//------------------------------------------------
// output of the vertex shader
//------------------------------------------------
struct SVSOutput {
	float2 vTex      : TEXCOORD0;
	float4 vPos      : POSITION;
};

//------------------------------------------------
// Vertex shader parameters
//------------------------------------------------
float4x4 mW2C;   // world to projection matrix
float4 vXZParam; // scale and offset for x and z position
float2 vHeight;  // height parameter
float3 vEye;     // view position
float4 vUVParam; // uv scale and offset


//------------------------------------------------
// Vertex shader
//------------------------------------------------
void mainVS(out SVSOutput _Output, in SVSInput _Input)
{
	// compute world position
	float4 vWorldPos;
	vWorldPos.xz = _Input.vPos.xy * vXZParam.xy + vXZParam.zw;
	// height is propotional to the square distance in horizontal direction.
	float2 vDir = vEye.xz - vWorldPos.xz;
	float fSqDistance = dot( vDir, vDir );
	vWorldPos.y = fSqDistance * vHeight.x + vHeight.y;
	vWorldPos.w = 1.0f;
	
	// transform and projection
	_Output.vPos = mul( vWorldPos, mW2C);
	
	// texture coordinate 
	_Output.vTex = _Input.vPos.zw * vUVParam.xy + vUVParam.zw;
}


//------------------------------------------------
// input of the pixel shader
//------------------------------------------------
struct SPSInput {
	float2 vTex      : TEXCOORD0;
};

//------------------------------------------------
// pixel shader parameters
//------------------------------------------------
sampler sCloud : register(s0);   // cloud texture 
float fCloudCover;               // cloud cover

//------------------------------------------------
// pixel shader
//------------------------------------------------
float4 mainPS(in SPSInput _Input) : COLOR0
{
	float4 clTex = tex2D( sCloud, _Input.vTex );
	
	// blend 4 channel of the cloud texture according to cloud cover 
	float4 vDensity;
	vDensity = abs( fCloudCover - float4( 0.25f, 0.5f, 0.75f, 1.0f ) ) / 0.25f;
	vDensity = saturate( 1.0f - vDensity );
	float _fDensity = dot( clTex, vDensity );
	return _fDensity;
}

