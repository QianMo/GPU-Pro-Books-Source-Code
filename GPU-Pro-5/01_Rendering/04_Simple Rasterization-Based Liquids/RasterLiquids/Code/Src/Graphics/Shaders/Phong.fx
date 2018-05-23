
Texture2D		txDiffuse	: register( t0 );

///< Terrain
Texture2D		txDisplacement : register( t1 );

#include "CommonConstants.hlsl"

cbuffer cbPhongParams : register( b2 )
{
	float4 LightPos1;
	float4 LightPos2;		
}

cbuffer cbMeshParams : register( b4 )
{
	float4 DefaultColor;	
}

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

struct VS_DISPLACEMENT_INPUT
{
    float4 Pos : POSITION;
    float2 UV : TEXCOORD0;
};

float3 GetNormal(float2 _UV)
{
	float step=1.0/256.0;
	
	float dx = txDisplacement.SampleLevel(samLinear,_UV + float2(step,0) ,0).x-txDisplacement.SampleLevel(samLinear,_UV - float2(step,0) ,0).x;
	float dy = txDisplacement.SampleLevel(samLinear,_UV + float2(0,step) ,0).x-txDisplacement.SampleLevel(samLinear,_UV - float2(0,step) ,0).x;
	
	float2 nTemp = normalize(float2(dx,dy));
	
	return normalize(float3(nTemp.x,-1.0, nTemp.y));
}

///< Vertex Shader
PS_INPUT PhongDisplacementVS(VS_DISPLACEMENT_INPUT _input)
{
	PS_INPUT output;
	
	float2 UV=_input.UV;
	float factor=30;

	float4 Pos = float4(_input.Pos.xyz,1);
	Pos.y = txDisplacement.SampleLevel(samLinear,UV,0).x*factor;
	
	output.Pos		=  mul( float4(Pos.xyz,1), World);
	output.WorldPos =	output.Pos.xyz;	
		
	output.Normal	= mul( GetNormal(_input.UV), (float3x3)World );
	
    output.Pos = mul( output.Pos, View );
    output.Pos = mul( Proj, output.Pos );
    
    output.UV = _input.UV;
	
    return output;    
}


///< Vertex Shader
PS_INPUT PhongVS(VS_INPUT _input)
{
	PS_INPUT output;
	
	output.Pos		= mul(float4(_input.Pos.xyz,1),  World);
	output.WorldPos = output.Pos.xyz;
	
	output.Normal	= mul(_input.Normal,(float3x3)World );
	
    output.Pos = mul( output.Pos, View );
    output.Pos = mul( Proj, output.Pos );
    
    output.UV = _input.UV;	
    return output;    
}

///<
float4 LightContribution(float3 _d, float3 _lightDir, float3 _viewDir, float3 _n)
{
	float3 h = normalize(_lightDir + _viewDir);	

	float s = clamp(45.0f-_d, 0, 45)/45.0f;

	return lit(dot(_lightDir,_n),dot(h,_n),55.0f);
}

///<
float3 PhongGeneric(float3 _worldX, float3 _camX, float3 _n, float3 _diffuse)
{
	float3 lightDir1 = (_worldX-LightPos1.xyz);
	float3 lightDir2 = (_worldX-LightPos2.xyz);
	float3 viewDir = normalize(_worldX-_camX);
	_n=-_n;

	float4 litV1 =  LightContribution(length(lightDir1), normalize(lightDir1), viewDir, _n); 
	float4 litV2 =  LightContribution(length(lightDir2), normalize(lightDir2), viewDir, _n); 
	float4 litV = 1.1f*(litV1 + litV2);
			
    float3 diffuse	= litV.y*_diffuse;
    float3 spec		= litV.y * litV.z *0.55f;
   
	return (spec + diffuse + float3(0.1,0.1,0.1));
}

///< Pixel Shader
float4 PhongPS( PS_INPUT _input) : SV_Target
{	
	float3 diff=txDiffuse.Sample(samLinear,_input.UV).xyz;
	return float4(PhongGeneric(_input.WorldPos.xyz, CamPos.xyz, normalize(_input.Normal), diff),1);   
}


