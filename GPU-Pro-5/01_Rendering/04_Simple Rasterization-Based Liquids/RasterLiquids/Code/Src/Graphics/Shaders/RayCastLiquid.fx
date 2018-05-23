
Texture2D		txRayDirections	: register( t0 );
Texture3D		txDensityVolume	: register( t1 );
Texture2D		txDepth			: register( t2 );
Texture2D		txEnvMap		: register( t3 );

#include "Physics\\SPHParams.hlsl"
#include "CommonConstants.hlsl"

struct VS_INPUT
{
    float4 Pos : POSITION;
    float3 UVW: TEXCOORD0;

};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    
    float3 UVW : TEXCOORD0;
    float3 WorldPos : TEXCOORD1;
};

///< Vertex Shader
PS_INPUT VS_DrawPosition(VS_INPUT _input)
{
	PS_INPUT output;
	
	output.UVW = _input.UVW;
	
	output.Pos = mul( float4(_input.Pos.xyz,1), World );
	output.WorldPos = output.Pos;
		
	output.Pos = mul( output.Pos, View );
    output.Pos = mul( Proj, output.Pos );
    
    return output;    
}

///<
float4 PS_DrawBackFaces( PS_INPUT _input) : SV_Target
{
	return float4(_input.UVW,0);
}

///<
float4 PS_DrawFrontFaces( PS_INPUT _input) : SV_Target
{
	return float4(-_input.UVW,0);
}

///<
float3 ComputeNormal(float3 _x, float _step)
{
	float nx = txDensityVolume.SampleLevel(samLinear, _x+float3(_step,0,0), 0).w-txDensityVolume.SampleLevel(samLinear, _x-float3(_step,0,0), 0).w;
	float ny = txDensityVolume.SampleLevel(samLinear, _x+float3(0,_step,0), 0).w-txDensityVolume.SampleLevel(samLinear, _x-float3(0,_step,0), 0).w;
	float nz = txDensityVolume.SampleLevel(samLinear, _x+float3(0,0,_step), 0).w-txDensityVolume.SampleLevel(samLinear, _x-float3(0,0,_step), 0).w;
	
	return normalize(float3(nx,ny,nz));
}

///<
float4 ComputeShading(float3 _x, float3 _nDir, float3 _n, float4 _envMapColor)
{
					
	float3 lightDir = normalize(float3(1,0,1));	
	float3 lightDir2 = normalize(float3(-1,0,1));

	float lightDiffuse = saturate(dot(lightDir,_n)) + saturate(dot(lightDir2,_n));
	
	float3 h = normalize(lightDir + _nDir);

	float3 spec = pow(saturate(dot(_n, h)), 30.5)*0.1f; 

	float3 color = float3(1,1,1)*0.1f;
	
	return float4(spec +  lightDiffuse*_envMapColor.xyz,1);	
}

///<
float4 PS_RayCastVolume( PS_INPUT _input) : SV_Target
{
	float2 UV = _input.Pos.xy/ScreenDimensions.xy;
	
	float dRay = 1.0/32.0;
		
	float3 backFace = txRayDirections.Sample(samPoint, UV);
	
	float3 dir = backFace-(_input.UVW);
	
	float IntDen=1;	
	float3 x=_input.UVW;
	
	float l=length(dir);
	float3 nDir = normalize(dir);
	
	float3 viewDir = normalize(_input.WorldPos-CamPos);
	
	float3 Z = float3(View._12, View._22, View._32);
	float3 Pv = normalize(Z-dot(Z,viewDir)*viewDir);
	float3 Ph = cross(Pv, viewDir);	
	
	
	float depth = txDepth.Sample(samPoint, UV);
	
	int steps = l/dRay;	
	for (float i=0; i<steps; ++i)
	{
		x+=dRay*nDir;

		float density = txDensityVolume.SampleLevel(samLinear, x, 0).w;
	
		if (density>0.8)
		{
			///< scale of the box is 20, [-1; 1].
			float3 worldPos = _input.WorldPos + 20.0f*viewDir*i*dRay;
			float4 projPos = mul( Proj, mul(float4(worldPos,1), View) );	
			float xz=projPos.z/(projPos.w);
			
			float3 n = ComputeNormal(x,dRay);
			
			float2 envUV =  (UV-0.5f)*2.0f + 0.5f - 0.10f*float2(dot(n, Ph), dot(n, Pv));
			float4 envMapColor = float4(envUV, 0.2f, 1);

			if (xz < depth)
				return ComputeShading(x, nDir, n, envMapColor);
			else
				return float4(0,0,0,0);				
						
		}		
	}
	
	return float4(0,0,0,0);

}

///< Pixel Shader
float4 PS_RayCastSmoke( PS_INPUT _input) : SV_Target
{
	return float4(1,0,0,1);
}