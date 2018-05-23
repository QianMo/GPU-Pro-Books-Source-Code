

Texture2D txDepth : register( t0 );
Texture2D txLightDepth : register( t1 );

#include "CommonConstants.hlsl"

cbuffer cbParams : register( b0 )
{
	///< Screen Resolution for post process!
    float4 FrameRes;  
    
    ///< 0=No SSOA
	///< 1=Min Distance
	///< 2=Max Distance
    float4 SSAO;
};

struct VS_INPUT
{
    float4 Pos : POSITION;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
};

///< Vertex Shader
PS_INPUT SSAO_VS(VS_INPUT _x)
{
	PS_INPUT output = (PS_INPUT)0;
	output.Pos = _x.Pos;
	
    return output;    
}

float GetDepth(float2 _UV)
{
	float depth = txDepth.Sample(samLinear, _UV); 
	return depth;
}

float AO(float2 _UV)
{
	float dp = GetDepth(_UV);
	float cd = 0;
	float dr = 1.0f;
	for(float i=-2; i<2; i=i+1)
	{
		for(float j=-2; j<2; j=j+1)
		{
			float d = GetDepth(_UV + (float2(i,j)*FrameRes*dr));
			float dist = length(d-dp);
			if (dist > 1.0*SSAO.y)// && dist < 0.5*SSAO.z)
				cd += 0.1f;
		}
	}

	return cd;
}

///< Pixel Shader
float4 SSAO_PS(PS_INPUT _x) : SV_Target
{

	float2 UV = _x.Pos.xy*FrameRes;
	float ao = AO(UV);
	//if (UV.x>0.2f)
	{		
		if (SSAO.x>0)
			return float4(ao,ao,ao,0);  
		else
			return float4(0,0,0,0); 
	} 
	
	return float4(0,0,0,0); 
		
}

 
float4 ShadowMap_PS(PS_INPUT _x) : SV_Target
{
	float2 UV = _x.Pos.xy*FrameRes;

	float depth1 = txDepth.Sample(samLinear, UV); 
	UV.x=1.0f-UV.x;
	float depth2 = txLightDepth.Sample(samLinear, UV);
	
	if (depth2>depth1)
		return float4(0.9f,0.9f,0.9f,0);
	else
		return float4(0,0,0,0); 		
}