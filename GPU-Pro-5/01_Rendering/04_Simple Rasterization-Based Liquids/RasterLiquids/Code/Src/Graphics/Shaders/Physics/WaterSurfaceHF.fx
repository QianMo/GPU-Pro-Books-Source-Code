
Texture2D		txDiffuse	: register( t0 );

///< Terrain
Texture2D		txDisplacement : register( t1 );
Texture2D		txTerrain : register( t2 );

#include "..\\CommonConstants.hlsl"

struct VS_INPUT
{
    float4 Pos : POSITION;
    float3 Normal : NORMAL;
    float2 UV : TEXCOORD0;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float3 Normal : NORMAL;
    float2 UV : TEXCOORD0;
    float3 WorldPos : TEXCOORD1;
};

///< Vertex Shader
PS_INPUT PhongVS(VS_INPUT _input)
{
	PS_INPUT output;
		
	output.WorldPos = mul( _input.Pos, World ).xyz;
	output.Pos		= mul( _input.Pos, World );
	output.Normal	= mul( _input.Normal, (float3x3)World );
	
    output.Pos = mul( output.Pos, View );
    output.Pos = mul( Proj, output.Pos );
    
    output.UV = _input.UV;
	
    return output;    
}

struct VS_DISPLACEMENT_INPUT
{
    float4 Pos : POSITION;
    float2 UV : TEXCOORD0;
};

float GetHeight(float2 _UV)
{
	return (txDisplacement.SampleLevel(samLinear,_UV,0).w) + 30.0f*txTerrain.SampleLevel(samLinear,_UV.xy,0).x; 	
}

float GetWaveHeight(float2 _UV)
{
	return 2.0f*txDisplacement.SampleLevel(samLinear,_UV,0).w;
}

float3 GetNormal(float2 _UV)
{
	float step=1.0/256.0;
	
	float dx = GetWaveHeight(_UV + float2(step,0))-GetWaveHeight(_UV - float2(step,0));
	float dy = GetWaveHeight(_UV + float2(0,step))-GetWaveHeight(_UV - float2(0,step));
	
	float2 nTemp = normalize(float2(dx,dy));
	
	return normalize(float3(nTemp.x,14.1f, nTemp.y));
}

///< Vertex Shader
PS_INPUT PhongDisplacementVS(VS_DISPLACEMENT_INPUT _input)
{
	PS_INPUT output;	
	 
	float4 Pos = float4(_input.Pos.xyz,1);
	Pos.y = GetHeight(_input.UV);
	
	output.Pos		= mul( Pos, World);
	output.WorldPos = output.Pos;	
	
	output.Normal	= mul( GetNormal(_input.UV), (float3x3)World );
	
    output.Pos = mul( output.Pos, View );
    output.Pos = mul( Proj, output.Pos );
    
    output.UV = _input.UV;
	
    return output;    
}

///< Pixel Shader
float4 PhongPS( PS_INPUT _input) : SV_Target
{	
	float3 LightPos = float3(-150,75,-150);
		
	float3 lightDir = normalize(_input.WorldPos.xyz-LightPos);
	float3 normal = -normalize(_input.Normal);
	float3 viewDir = normalize(_input.WorldPos.xyz-CamPos.xyz);
	
	float3 h = normalize(lightDir + viewDir);
	
	float4 litV = lit(dot(lightDir,normal),dot(h,normal),55.0f);
	
    float diffuse = litV.y;
    float3 spec = litV.y * litV.z * 0.9f;
    
    float d=max((sqrt(min(txDiffuse.Sample(samLinear,_input.UV).w,8))/2.0f-0.3f),0);
    
    float3 brownSand=float3(1.0f, 1.0f, 1.0f);
    float3 blueSea = float3(0, 0.3f, 0.8f);
    
	float3 diffuseAmb = 1.5f*d*diffuse*blueSea + (1.0f-d)*brownSand; 
	
    return float4( d*spec + diffuseAmb, 1.0f);
}