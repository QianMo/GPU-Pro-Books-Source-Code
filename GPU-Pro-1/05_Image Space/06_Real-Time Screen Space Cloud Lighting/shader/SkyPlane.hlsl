//------------------------------------------------
// SkyPlane.hlsl
//
// Shader to render sky plane
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//------------------------------------------------


//------------------------------------------------
// vertex shader parameters
//------------------------------------------------

float4x4 mC2W;   // inverse matrix of the transform world to projection 
float3 vEye;     // view position
float3 litDir;   // light direction


//------------------------------------------------
// vertex shader input 
//------------------------------------------------
struct SVSInput {
	float4 vPosC : POSITION0;
};


//------------------------------------------------
// vertex shader output
//------------------------------------------------
struct SVSOutput {
	float4 vWorldPos : TEXCOORD0;
	float4 vPos : POSITION0;
};


//------------------------------------------------
// vertex shader
//------------------------------------------------
void mainVS( out SVSOutput _Output, in SVSInput _Input )
{
	_Output.vPos = _Input.vPosC;
	_Output.vWorldPos = mul( _Input.vPosC, mC2W );

}


//------------------------------------------------
// pixel shader parameters
//------------------------------------------------
float4 scat[5];  // scattering paraemters


//------------------------------------------------
// pixel shader input 
//------------------------------------------------

struct SPSInput {
	float4 vWorldPos : TEXCOORD0;
	float2 vScreenPos : VPOS;
};


//------------------------------------------------
// pixel shader
//------------------------------------------------

float4 mainPS(in SPSInput _Input) : COLOR0
{
	// ray direction 
	float3 _vWorldPos = _Input.vWorldPos.xyz/_Input.vWorldPos.w;
	float3 _vRay = _vWorldPos - vEye;	
	float _fSqDistance = dot( _vRay, _vRay );
	_vRay = normalize( _vRay );

	// calcurating in-scattering 	
	float _fVL = dot( litDir, -_vRay );
	float fG = scat[1].w + scat[0].w * _fVL;
	fG = rsqrt( fG );
	fG = fG*fG*fG;
	float3 _vMie = scat[1].rgb * fG;
	float3 _vRayleigh = scat[0].rgb*(1.0f + _fVL*_fVL);
	float3 _vInscattering = scat[2] * (_vMie + _vRayleigh) + scat[4].rgb;
	
	// compute distance the light passes through the atmosphere
	float _fSin = _vRay.y;
	float _fRSin = scat[2].w * _fSin;
	float _fDistance = sqrt( _fRSin * _fRSin + scat[3].w ) - _fRSin;
	
	float3 fRatio = exp( -scat[3].rgb * _fDistance );	
	float4 _color;
	_color.rgb = (1.0f-fRatio) *_vInscattering;
	_color.a = 1.0f;

	return _color;
}


