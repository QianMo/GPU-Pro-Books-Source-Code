matrix WorldViewProj;
matrix World;

matrix WorldViewProjLight;

float3 LightPos;
float3 CameraPos;

Texture2D DeferedPosInWorld;
Texture2D DeferedPosInLight;
Texture2D DeferedNormal;
Texture2D DeferedColor;
Texture2D FilterPassPrecalcDepth;
Texture2D Texture;

float2 FrameBufferDimension;

float3 Color;

SamplerState SamPoint
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
    AddressW = Clamp;
};

SamplerState SamLinear
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
    AddressW = Wrap;
};

#define AA_WEIGHT 0.5f

#include "DeepShadowMapGlobal.hlsl"

float CalculateKajiyaKay(float3 tangent, float3 posInWorld)
{
	float3 eye = normalize(CameraPos - posInWorld);
	float3 light = normalize(LightPos - posInWorld);
	
	float diffuse = sin(acos(dot(tangent, light)));
	float specular = pow(dot(tangent, light) * dot(tangent, eye) + sin(acos(dot(light, tangent))) * sin(acos(dot(tangent, eye))), 6.0f);
	
	return (max(min(diffuse, 1.0f), 0.0f) * 0.7f + max(min(specular, 1.0f), 0.0f) * 0.8f) * 0.90f; // scale this thing a bit
}

float CalculateSpecularColor(float3 norm, float3 posInWorld)
{
	float3 light = normalize(LightPos - posInWorld);
	float nDotL  = clamp(dot(light, norm), 0.0f, 1.0f);
			return nDotL;				
	float3 view = CameraPos - posInWorld;
	float3 halfAngle = normalize(2 * dot(view, norm) * norm - view);
	float rDotL = saturate(dot(halfAngle, light));
	return clamp(nDotL * pow(rDotL, 60.0f), 0.0f, 1.0f); 
}

void defaultShadowMapCompare(inout LinkedListEntryWithPrev entry, inout LinkedListEntryNeighbors entryNeighbors, float z, out float outShading)
{
	LinkedListEntryWithPrev tempEntry;
	
	if(entry.depth < z)
		outShading = entry.shading;
	else
		outShading = 1.0f;
}

// ----------------------------------------------

struct VS_IN
{
	float4 pos : POSITION;
	float2 texcoord : TEXCOORD;
};

struct PS_IN
{
	float4 pos : SV_POSITION;
	float2 texcoord : TEXCOORD0;
};

PS_IN vs_main(VS_IN input)
{
	PS_IN output = (PS_IN)0;
	
	output.pos = input.pos;
	output.texcoord = input.texcoord;
	return output;
}

float4 ps_main(PS_IN input) : SV_Target
{
	// HAIR: norm is actually the tangent
	float3 norm = DeferedNormal.Sample(SamPoint, input.texcoord);

	if(norm.x == 0 && norm.y == 0 && norm.z == 0)
		return float4(1.0f, 1.0f, 1.0f, 1.0f);
		
	float3 posInWorld = DeferedPosInWorld.Sample(SamPoint, input.texcoord);
	float3 posInLight = DeferedPosInLight.Sample(SamPoint, input.texcoord);
	
	int xLight = posInLight.x;
	int yLight = posInLight.y;
		
	float3 finalColor;
	float4 objectColor = DeferedColor.Sample(SamPoint, input.texcoord);
	if(objectColor.a == 0.0f) // means do KajiyaKay shading
		finalColor = objectColor.xyz * (CalculateKajiyaKay(normalize(norm), posInWorld) + 0.09f);
	else
		finalColor = objectColor.xyz * (CalculateSpecularColor(normalize(norm), posInWorld) + 0.09f);
	
	float shading = 0.0f;

	LinkedListEntryWithPrev currentXEntry;
	LinkedListEntryWithPrev currentYEntry;
	LinkedListEntryNeighbors currentXEntryNeighbors;
	LinkedListEntryNeighbors currentYEntryNeighbors;
	int currentX = xLight - FILTER_SIZE;
	bool noXLink = true;
	
	for(int x = 0; x < FILTER_SIZE * 2 + 1; x++)
	{
		bool noYLink = false;
		int currentY = yLight - FILTER_SIZE;

		if(noXLink && !(currentX < 0 || currentY < 0 || currentX >= Dimension || currentY >= Dimension))
		{
			int start = StartElementBufRO[currentY * Dimension + currentX].start;
			if(start != -1)
			{
				currentXEntry = LinkedListBufWPRO[start];
				currentXEntryNeighbors = NeighborsBufRO[start];
				noXLink = false;
			}
		}

		if(noXLink)
			noYLink = true;
		else
		{
			currentYEntry = currentXEntry;
			currentYEntryNeighbors = currentXEntryNeighbors;
		}

		for(int y = 0; y < FILTER_SIZE * 2 + 1; y++)
		{
			if(currentX < 0 || currentY < 0 || currentX >= Dimension || currentY >= Dimension)
			{
				shading += 1.0f;
				currentY++;
				continue;
			}
			
			if(noYLink)
			{
				int start = StartElementBufRO[currentY * Dimension + currentX].start;
				if(start == -1)
				{
					shading += 1.0f;
					currentY++;
					continue;
				}

				noYLink = false;
				currentYEntry = LinkedListBufWPRO[start];
				currentYEntryNeighbors = NeighborsBufRO[start];
			}
			
			float localShading;
			
			defaultShadowMapCompare(currentYEntry, currentYEntryNeighbors, posInLight.z, localShading);
			shading += localShading;

			currentY++;

			if(currentYEntryNeighbors.top != -1 && !noYLink)
			{
				currentYEntry = LinkedListBufWPRO[currentYEntryNeighbors.top];
				currentYEntryNeighbors = NeighborsBufRO[currentYEntryNeighbors.top];
			}
			else
				noYLink = true;
		}

		currentX++;

		if(currentXEntryNeighbors.right != -1 && !noXLink)
		{
			currentXEntry = LinkedListBufWPRO[currentXEntryNeighbors.right];
			currentXEntryNeighbors = NeighborsBufRO[currentXEntryNeighbors.right];
		}
		else
			noXLink = true;
	}

	shading /= FILTER_AREA;
	
	return float4(finalColor * clamp(shading, 0.1f, 1.0f), 1.0f);
}

float4 ps_main_aa(PS_IN input) : SV_Target
{
	const float2 delta[8] =
	{
		float2(-1, 1), float2(1, -1), float2(-1, 1), float2(1, 1),
		float2(-1, 0), float2(1, 0), float2(0, -1), float2(0, 1)
	};
	
	float3 tex = DeferedNormal.Sample(SamPoint, input.texcoord);
	float factor = 0.0f;
	
	for(int i = 0; i < 4 ; i++)
	{
		float3 t = DeferedNormal.Sample(SamPoint, input.texcoord + delta[i] / FrameBufferDimension);
		
		t -= tex;
		factor += dot(t, t);
	}
	
	factor = min(1.0, factor) * AA_WEIGHT;
	
	float4 color = float4(0.0,0.0,0.0,0.0);
	
	for(int i = 0 ; i < 8 ; i++)
		color += DeferedColor.Sample(SamPoint, input.texcoord + delta[i] / FrameBufferDimension * AA_WEIGHT);
	
	color += 2.0f * DeferedColor.Sample(SamPoint, input.texcoord);
	
	return color * 1.0f / 10.0f;
}

RasterizerState DisableCulling
{
    CullMode = NONE;
	DepthBias = 0;
	SlopeScaledDepthBias = 0.0f;
};

struct VS_IN_DEFERED
{
	float4 pos : POSITION;
	float3 norm : NORMAL;
};

struct PS_IN_DEFERED
{
	float4 pos : SV_POSITION;
	float3 norm : NORMAL;
	float3 posInWorld : TEXCOORD0;
	float4 posInLight : TEXCOORD1;
};

struct PS_OUT_DEFERED
{
	float3 norm : SV_Target0;
	float3 posInWorld : SV_Target1;
	float3 posInLight : SV_Target2;
	float4 color : SV_Target3;
};

PS_IN_DEFERED vs_main_defered(VS_IN_DEFERED input)
{
	PS_IN_DEFERED output = (PS_IN_DEFERED)0;
	output.pos = mul(input.pos, WorldViewProj);
	output.norm = mul(input.norm, (float3x3)World);
	output.posInWorld = mul(input.pos, World);
	output.posInLight = mul(input.pos, WorldViewProjLight);
	
	return output;
}

PS_OUT_DEFERED ps_main_defered(PS_IN_DEFERED input)
{
	PS_OUT_DEFERED output = (PS_OUT_DEFERED)0;
	input.posInLight.xyz /= input.posInLight.w;
	input.posInLight.xy = input.posInLight.xy / 2 + 0.5f;
	input.posInLight.y = 1.0f - input.posInLight.y;
	input.posInLight.xy *= Dimension;
	
	output.norm = input.norm;
	output.posInLight.xyz = input.posInLight.xyz;
	output.posInWorld = input.posInWorld;
	output.color = float4(Color.xyz, 0);
	return output;
}

struct VS_IN_DEFERED_MODEL
{
	float4 pos : POSITION;
	float3 norm : NORMAL;
	float2 texcoord : TEXCOORD0;
};

struct PS_IN_DEFERED_MODEL
{
	float4 pos : SV_POSITION;
	float3 norm : NORMAL;
	float3 posInWorld : TEXCOORD0;
	float4 posInLight : TEXCOORD1;
	float2 texcoord : TEXCOORD2;
};

PS_IN_DEFERED_MODEL vs_main_defered_model(VS_IN_DEFERED_MODEL input)
{
	PS_IN_DEFERED_MODEL output = (PS_IN_DEFERED_MODEL)0;
	output.pos = mul(input.pos, WorldViewProj);
	output.norm = mul(input.norm, (float3x3)World);
	output.posInWorld = mul(input.pos, World);
	output.posInLight = mul(input.pos, WorldViewProjLight);
	output.texcoord = input.texcoord;
	
	return output;
}

PS_OUT_DEFERED ps_main_defered_model(PS_IN_DEFERED_MODEL input)
{
	PS_OUT_DEFERED output = (PS_OUT_DEFERED)0;
	input.posInLight.xyz /= input.posInLight.w;
	input.posInLight.xy = input.posInLight.xy / 2 + 0.5f;
	input.posInLight.y = 1.0f - input.posInLight.y;
	input.posInLight.xy *= Dimension;
	
	output.norm = input.norm;
	output.posInLight.xyz = input.posInLight.xyz;
	output.posInWorld = input.posInWorld;
	output.color = float4(Texture.Sample(SamLinear, input.texcoord).rgb, 1);  // alpha == 1 -> shader type 1
	return output;
}

technique11 Render
{
	pass P0
	{
		SetRasterizerState(DisableCulling);  
		SetVertexShader(CompileShader(vs_5_0, vs_main()));
		SetPixelShader (CompileShader(ps_5_0, ps_main()));
	}
}

technique11 RenderDefered
{
	pass P0
	{
		SetRasterizerState(DisableCulling);  
		SetVertexShader(CompileShader(vs_5_0, vs_main_defered()));
		SetPixelShader (CompileShader(ps_5_0, ps_main_defered()));
	}
}

technique11 RenderDeferedModel
{
	pass P0
	{
		SetRasterizerState(DisableCulling);  
		SetVertexShader(CompileShader(vs_5_0, vs_main_defered_model()));
		SetPixelShader (CompileShader(ps_5_0, ps_main_defered_model()));
	}
}

technique11 RenderAA
{
	pass P0
	{
		SetRasterizerState(DisableCulling);  
		SetVertexShader(CompileShader(vs_5_0, vs_main()));
		SetPixelShader (CompileShader(ps_5_0, ps_main_aa()));
	}
}