//------------------------------------------------
// CloudBlur.hlsl
//
// Shader to blur cloud density
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//------------------------------------------------

//------------------------------------------------
// vertex shader parameters
//------------------------------------------------
float4x4 mC2W;  // inverse matrix of world to projection matrix
float2   vPix;  // 0.5f/texture size
float4   vOff;  // parameter to compute blur direction



//------------------------------------------------
// vertex shader input 
//------------------------------------------------
struct SVSInput {
	float4 vPosC : POSITION0;
	float2 vTex : TEXCOORD0;
};

//------------------------------------------------
// vertex shader output
//------------------------------------------------
struct SVSOutput {
	float4 vWorldPos : TEXCOORD0;
	float4 vTex : TEXCOORD1;
	float4 vPos : POSITION0;
};


//------------------------------------------------
// vertex shader
//------------------------------------------------

void mainVS( out SVSOutput _Output, in SVSInput _Input )
{
	// projection position
	_Output.vPos = float4( _Input.vPosC.xy, 0.5f, 1 );

	// transform projection space to world space 	
	_Output.vWorldPos = mul( _Input.vPosC, mC2W );

	// uv 
	_Output.vTex.xy = _Input.vTex + vPix;

	// compute blur direction 
	_Output.vTex.zw = _Input.vTex * vOff.xy + vOff.zw;

}


//------------------------------------------------
// pixel shader parameters
//------------------------------------------------

#define FILTER_WIDTH 16

sampler sDensity : register(s0);  // density map

float3  vParam;   // 
float4  vFallOff; // fall off parameters
float2  invMax;   // inverse of the maximum length of the blur vector
float3  vEye;     // view position


//------------------------------------------------
// pixel shader input 
//------------------------------------------------

struct SPSInput {
	float4 vWorldPos : TEXCOORD0;
	float4 vTex : TEXCOORD1;
	float2 vScreenPos : VPOS;
};


//------------------------------------------------
// pixel shader
//------------------------------------------------

float4 mainPS(in SPSInput _Input) : COLOR0
{
	// compute ray direction 
	float3 _vWorldPos = _Input.vWorldPos.xyz/_Input.vWorldPos.w;
	float3 _vRay = _vWorldPos - vEye;	
	float _fSqDistance = dot( _vRay, _vRay );
	_vRay = normalize( _vRay );

	// compute distance the light passes through the atmosphere.	
	float _fSin = _vRay.y;
	float _fRSin = vParam.x * _fSin;
	float _fDistance = sqrt( _fRSin * _fRSin + vParam.y ) - _fRSin;

	// Compute UV offset.
	float2 _vUVOffset = _Input.vTex.zw / FILTER_WIDTH * (vParam.z / _fDistance);

	// limit blur vector
	float2 _len = abs( _vUVOffset * invMax );
	float _over = max( _len.x, _len.y );
	float _scale = _over > 1.0f ? 1.0f/_over : 1.0f;
	_vUVOffset.xy *= _scale;

	// scale parameter of exponential weight
	float4 _distance;
	_distance = dot( _vUVOffset.xy, _vUVOffset.xy );
	_distance *= vFallOff;

	// blur 
	float2 _uv = _Input.vTex.xy;
	float4 _clOut = tex2D( sDensity, _uv );
	float4 _fWeightSum = 1.0f;
	for ( int i = 1; i < FILTER_WIDTH; ++i ) {
		float4 _weight = exp( _distance * i );
		_fWeightSum += _weight;
		
		float2 _vMove = _vUVOffset * i;		
		float4 _clDensity = tex2D( sDensity, _uv + _vMove );
		_clOut += _weight * _clDensity;
	}
	_clOut /= _fWeightSum;

	return _clOut;
}


