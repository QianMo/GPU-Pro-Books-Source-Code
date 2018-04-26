//------------------------------------------------
// CloudPlane.hlsl
//
// Shader to render screen cloud
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//------------------------------------------------



//------------------------------------------------
// vertex shader parameters
//------------------------------------------------

float4x4 mC2W;    // inverse matrix of the transform world to projection 
float3   vEye;    // view point
float3   litDir;  // ligtht direction

//------------------------------------------------
// vertex shader input 
//------------------------------------------------
struct SVSInput {
	float4 vPosC : POSITION0;
	float2 vTex      : TEXCOORD0;
};

//------------------------------------------------
// vertex shader output
//------------------------------------------------
struct SVSOutput {
	float4 vWorldPos : TEXCOORD0;
	float2 vTex      : TEXCOORD2;
	float4 vPos : POSITION0;
};


//------------------------------------------------
// vertex shader
//------------------------------------------------

void mainVS( out SVSOutput _Output, in SVSInput _Input )
{
	// Output projection position
	_Output.vPos = _Input.vPosC;
	
	// world positon
	_Output.vWorldPos = mul( _Input.vPosC, mC2W );

	// uv 
	_Output.vTex = _Input.vTex;
}


//------------------------------------------------
// pixel shader input 
//------------------------------------------------

struct SPSInput {
	float4 vWorldPos : TEXCOORD0;
	float2 vTex      : TEXCOORD2;
	float2 vScreenPos : VPOS;
};


//------------------------------------------------
// pixel shader parameters
//------------------------------------------------

sampler sDensity : register(s0);  // density map
sampler sLit : register(s1);      // blurred densitymap

float4 scat[5];   // scattering parameters
float2 vDistance; // parameter for computing distance to the cloud
float3 cLit;      // light color
float3 cAmb;      // ambient light


//------------------------------------------------
// pixel shader
//------------------------------------------------

float3 ApplyScattering( float3 _clInput, float3 _vRay )
{
	// calcurating in-scattering 	
	float _fVL = dot( litDir, -_vRay );
	float fG = scat[1].w + scat[0].w * _fVL;
	fG = rsqrt( fG );
	fG = fG*fG*fG;
	float3 _vMie = scat[1].rgb * fG;
	float3 _vRayleigh = scat[0].rgb*(1.0f + _fVL*_fVL);
	float3 _vInscattering = scat[2] * (_vMie + _vRayleigh) + scat[4].rgb;
	
	// compute distance to the cloud
	float _fSin = _vRay.y;
	float _fRSin = vDistance.x * _fSin;
	float _fDistance = sqrt( _fRSin * _fRSin + vDistance.y ) - _fRSin;
			
	float3 fRatio = exp( -scat[3].rgb * _fDistance );		
	return lerp( _vInscattering, _clInput, fRatio );
}


float4 mainPS(in SPSInput _Input) : COLOR0
{
	float4 _clDensity = tex2D( sDensity, _Input.vTex.xy );
	float4 _clLit     = 1.0f - tex2D( sLit, _Input.vTex.xy );

	// light cloud
	float3 _clCloud = cAmb + cLit * _clLit.r;
	
	// compute ray direction	
	float3 _vWorldPos = _Input.vWorldPos.xyz/_Input.vWorldPos.w;
	float3 _vRay = _vWorldPos - vEye;	
	float _fSqDistance = dot( _vRay, _vRay );
	_vRay = normalize( _vRay );
		
	// apply scattering 
	float4 _color;
	_color.rgb = ApplyScattering( _clCloud, _vRay );
	_color.rgb = saturate(_color.rgb);
	_color.a = _clDensity.a;

	return _color;
}


