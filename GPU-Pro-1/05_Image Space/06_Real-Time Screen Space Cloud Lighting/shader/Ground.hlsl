//------------------------------------------------
// Ground.hlsl
//
// Shader to render a uniform grid terrain
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//------------------------------------------------


//------------------------------------------------
// input of the vertex shader
//------------------------------------------------
struct SVSInput {
	float4 vPos    : POSITION0;
	float3 vNormal : NORMAL0;
	float4 vTex    : TEXCOORD0;
};

//------------------------------------------------
// output of the vertex shader
//------------------------------------------------
struct SVSOutput {
	float4 vTex         : TEXCOORD0;
	float3 vWorldNormal : TEXCOORD1;
	float3 vWorldPos    : TEXCOORD2;
	float4 vShadowPos   : TEXCOORD3;
	float4 vPos         : POSITION;
};

//------------------------------------------------
// vertex shader parameters
//------------------------------------------------
float4x4 mL2C;  // local to projection matrix
float4x4 mL2W;  // local to world transform
float4x4 mL2S;  // local to shadowmap texture 

//------------------------------------------------
// vertex shader
//------------------------------------------------
void mainVS(out SVSOutput _Output, in SVSInput _Input)
{
	// projection
	_Output.vPos = mul( _Input.vPos, mL2C);
	
	// compute world normal vector
	_Output.vWorldNormal = (float3)mul( _Input.vNormal, mL2W );
	// compute world position
	_Output.vWorldPos = (float3)mul( _Input.vPos, mL2W );
	
	// shadowmap position
	_Output.vShadowPos = mul( _Input.vPos, mL2S);
	
	// copy uvs
	_Output.vTex = _Input.vTex;
}



//------------------------------------------------
// pixel shader input 
//------------------------------------------------
struct SPSInput {
	float4 vTex         : TEXCOORD0;
	float3 vWorldNormal : TEXCOORD1;
	float3 vWorldPos    : TEXCOORD2;
	float4 vShadowPos   : TEXCOORD3;
};



//------------------------------------------------
// pixel shader parameters
//------------------------------------------------
sampler sGroundBlend : register(s0);  // blend texture
sampler sGround0 : register(s1);      // ground textures
sampler sGround1 : register(s2);
sampler sGround2 : register(s3);
sampler sGround3 : register(s4);
sampler sGround4 : register(s5);
sampler sShadow : register(s6);       // cloud shadowmap

float3 vEye;    // view position
float3 litDir;  // light direction
float3 litCol;  // light color
float3 litAmb;  // ambient light
float4 scat[5]; // scattering parameters
float4 mDif;    // diffuse reflection  
float4 mSpc;    // specular reflection



//------------------------------------------------
// pixel shader 
//------------------------------------------------


//------------------------------------------------
// Sample and blend ground textures 
//------------------------------------------------
float4 GetGroundTexture(float4 vTex)
{
	// Sample blend texture 
	// Each channel of the blend texture means the density of each ground textures.
	float4 _clGroundBlend = tex2D( sGroundBlend, vTex.zw );
	// sample ground textures 
	float4 _clGroundTex[4];
	_clGroundTex[0] = tex2D( sGround1, vTex.xy );
	_clGroundTex[1] = tex2D( sGround2, vTex.xy );
	_clGroundTex[2] = tex2D( sGround3, vTex.xy );
	_clGroundTex[3] = tex2D( sGround3, vTex.xy );
	// blend ground textures
	float4 _clGround = tex2D( sGround0, vTex.xy );
	_clGround = lerp( _clGround, _clGroundTex[0], _clGroundBlend.r );
	_clGround = lerp( _clGround, _clGroundTex[1], _clGroundBlend.g );
	_clGround = lerp( _clGround, _clGroundTex[2], _clGroundBlend.b );
	_clGround = lerp( _clGround, _clGroundTex[3], _clGroundBlend.a );	
	
	return _clGround;
}


float4 mainPS(in SPSInput _Input) : COLOR0
{
	// Sample shadow map 
	// Cloud shadow is not depth but transparency of shadow.
	float4 _clShadow = tex2D( sShadow, _Input.vShadowPos.xy/_Input.vShadowPos.w );

	float3 _vNormal = normalize( _Input.vWorldNormal );
	
	float3 _vRay = _Input.vWorldPos - vEye;
	float _fSqDistance = dot( _vRay, _vRay );
	_vRay *= rsqrt(_fSqDistance);
	
	// apply light
	float3 _clDiffuse = litAmb;
	float3 _clSpecular = 0;
	float3 _fDot = dot( -litDir, _vNormal );	
	float3 _vHalf = -normalize( litDir.xyz + _vRay );
	float4 _lit = lit( _fDot, dot(_vHalf, _vNormal ), mSpc.w );
	float3 _litCol = _clShadow * litCol;
	_clDiffuse += _litCol * _lit.y;
	_clSpecular += _litCol * _lit.z;	

	// blend ground textures
	// alpha channel is used as a mask of specular
	float4 _clGround = GetGroundTexture( _Input.vTex );
	
	// apply material
	float4 _color;
	_color.rgb = mDif.rgb * _clDiffuse * _clGround.rgb;
	_color.a   = mDif.a;
	_color.rgb += _clSpecular * mSpc.rgb * _clGround.a;

	// apply scattering	
	
	// compute in-scattering term 	
	float _fVL = dot( litDir, -_vRay );
	float fG = scat[1].w + scat[0].w * _fVL;
	fG = rsqrt( fG );
	fG = fG*fG*fG;
	float3 _vMie = scat[1].rgb * fG;
	float3 _vRayleigh = scat[0].rgb*(1.0f + _fVL*_fVL);
	float3 _vInscattering = scat[2] * (_vMie + _vRayleigh) + scat[4].rgb;

	// ratio of scattering 
	float fDistance = sqrt( _fSqDistance );
	float3 fRatio = exp( -scat[3].rgb * fDistance );
	
	_color.rgb = lerp( _vInscattering, _color.rgb, fRatio );
	return _color;
}

