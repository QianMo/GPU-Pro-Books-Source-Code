
#include "std_cbuffer.h"
#include "light_definitions.h"
#include "illum.h"


//-----------------------------------------------------------------------------------------
// Textures and Samplers
//-----------------------------------------------------------------------------------------

Texture2D g_norm_tex;


Buffer<uint4> g_vLightList;
StructuredBuffer<SFiniteLightData> g_vLightData;

SamplerState g_samWrap;
SamplerState g_samClamp;
SamplerComparisonState g_samShadow;

//--------------------------------------------------------------------------------------
// shader input/output structure
//--------------------------------------------------------------------------------------
struct VS_INPUT
{
    float3 Position     : POSITION;
    float3 Normal       : NORMAL;
    float2 TextureUV    : TEXCOORD0;
    float4 Tangent		: TEXCOORD1;
    float4 BiTangent    : TEXCOORD2;
};

struct VS_OUTPUT
{
    float4 Position     : SV_POSITION;
    float4 Diffuse      : COLOR0;
    float2 TextureUV    : TEXCOORD0;
	float3 norm			: TEXCOORD1;
    float3 tang			: TEXCOORD2;
    float3 bitang		: TEXCOORD3;
};


VS_OUTPUT RenderSceneVS( VS_INPUT input)
{
	VS_OUTPUT Output;
	float3 vNormalWorldSpace;

	// Transform the position from object space to homogeneous projection space
	Output.Position = mul( float4(input.Position.xyz,1), g_mWorldViewProjection );

	// position & normal
	Output.norm = mul(input.Normal.xyz, (float3x3) g_mLocToView);
	Output.tang = mul(input.Tangent.xyz, (float3x3) g_mLocToView);
	Output.bitang = mul(input.BiTangent.xyz, (float3x3) g_mLocToView);
	
	// tile the texture
	Output.TextureUV = 60*float2(input.TextureUV.x,1-input.TextureUV.y); 

	return Output;    
}


float3 ExecuteLightList(const uint4 uList, const float3 vV, const float3 vVPos, const float3 vN, const float3 unit_normal);
uint GetSubListLightCount(uint4 uList);


float4 RenderScenePS( VS_OUTPUT In ) : SV_TARGET0
{ 	
	float4 v4ScrPos = float4(In.Position.xyz, 1);
	float4 v4ViewPos = mul(v4ScrPos, g_mScrToView);
	float3 vVPos = v4ViewPos.xyz / v4ViewPos.w;
	float3 vV = normalize(-vVPos);
		
	float3 unit_normal = normalize(In.norm);

	float3 vNt = 2*g_norm_tex.Sample( g_samWrap, In.TextureUV.xy ).xyz-1;
	vNt.xy *= 0.3;		// bump scale
	float3 vN = normalize(In.tang*vNt.x + In.bitang*vNt.y + In.norm*vNt.z);

	// find our tile from the pixel coordinate (tiled forward lighting)
	uint2 uTile = ((uint2) In.Position.xy) / 16;
	int NrTilesX = (g_iWidth+15)/16;
	const int offs = uTile.y*NrTilesX+uTile.x;
	

	// execute lighting
	float3 ints = 0;

	uint4 uListA = g_vLightList[2*offs+0];		// lower 12 lights
	ints += ExecuteLightList( uListA, vV, vVPos, vN, unit_normal );
	uint4 uListB = g_vLightList[2*offs+1];		// upper 12 lights
	ints += ExecuteLightList( uListB, vV, vVPos, vN, unit_normal );

	float3 res = ints;


	// debug heat vision
	if(g_iMode == 0)		// debug view on
	{
		const float4 kRadarColors[12] = 
		{
			float4(0.0,0.0,0.0,0.0),   // black
			float4(0.0,0.0,0.6,0.5),   // dark blue
			float4(0.0,0.0,0.9,0.5),   // blue
			float4(0.0,0.6,0.9,0.5),   // light blue
			float4(0.0,0.9,0.9,0.5),   // cyan
			float4(0.0,0.9,0.6,0.5),   // blueish green
			float4(0.0,0.9,0.0,0.5),   // green
			float4(0.6,0.9,0.0,0.5),   // yellowish green
			float4(0.9,0.9,0.0,0.5),   // yellow
			float4(0.9,0.6,0.0,0.5),   // orange
			float4(0.9,0.0,0.0,0.5),   // red
			float4(1.0,0.0,0.0,0.9)    // strong red
		};

		uint uNrLights = GetSubListLightCount(uListA)+GetSubListLightCount(uListB);

		float fMaxNrLightsPerTile = 24;
		
		// change of base
		// logb(x) = log2(x) / log2(b)
		int nColorIndex = uNrLights==0 ? 0 : (1 + (int) floor(10 * (log2((float)uNrLights) / log2(fMaxNrLightsPerTile))) );
		nColorIndex = nColorIndex<0 ? 0 : nColorIndex;
		float4 col = nColorIndex>11 ? float4(1.0,1.0,1.0,1.0) : kRadarColors[nColorIndex];
		col.xyz = pow(col.xyz, 2.2);

		res = uNrLights==0 ? 0 : (res*(1-col.w) + 0.15*col*col.w);
	}

	return float4(res,1);
}


uint FetchIndex(uint4 uEntries, uint l)
{
	uint uEntry = l<3 ? l : (l<6 ? (l-3) : (l<9 ? (l-6) : (l-9)));
	uint uIndex = l<3 ? uEntries.x : (l<6 ? uEntries.y : (l<9 ? uEntries.z : uEntries.w));
	uIndex = (uIndex>>(uEntry*10)) & 0x3ff;

	return uIndex;
}

uint GetSubListLightCount(uint4 uList)
{
	return ((uList.x>>30)&0x3)+((uList.y>>30)&0x3)+((uList.z>>30)&0x3)+((uList.w>>30)&0x3);
}


float3 ExecuteLightList(const uint4 uList, const float3 vV, const float3 vVPos, const float3 vN, const float3 unit_normal)
{
	float3 ints = 0;


	// hack since we're not actually using imported values for this demo
	float3 vMatColDiff = 0.26*float3(1,1,1);
	float3 vMatColSpec = 0.5*0.3;
	float fSpecPow = 12;



	uint uNrLights = GetSubListLightCount(uList);

	uint l=0;

	// we need this outer loop for when we cannot assume a wavefront is 64 wide
	// since in this case we cannot assume the lights will remain sorted by type
	// during processing in lightlist_cs.hlsl
#if !defined(XBONE) && !defined(PLAYSTATION4)
	while(l<uNrLights)
#endif
	{
		uint uIndex = l<uNrLights ? FetchIndex(uList, l) : 0;
		uint uLgtType = l<uNrLights ? g_vLightData[uIndex].uLightType : 0;

		// specialized loop for spot lights
		while(l<uNrLights && (uLgtType==SPOT_CIRCULAR_LIGHT || uLgtType==WEDGE_LIGHT))
		{
			SFiniteLightData lgtDat = g_vLightData[uIndex];	
			float3 vLp = lgtDat.vLpos.xyz;
																											// nuts but vLdir is the X axis
			if(uLgtType==WEDGE_LIGHT) vLp += clamp(dot(vVPos-vLp, lgtDat.vLdir.xyz), 0, lgtDat.fSegLength) * lgtDat.vLdir.xyz;	// wedge light

			float3 toLight  = vLp - vVPos;
			float dist = length(toLight);

			float3 vL = toLight / dist;
			float fAttLook = saturate(dist * lgtDat.fInvRange + lgtDat.fNearRadiusOverRange_LP0);

			
			float fAngFade = saturate((lgtDat.fPenumbra + dot(lgtDat.vBoxAxisX.xyz, vL)) * lgtDat.fInvUmbraDelta);	// nuts but entry vBoxAxisX is the spot light dir
			fAngFade = fAngFade*fAngFade*(3-2*fAngFade);    // smooth curve
			fAngFade *= fAngFade;                           // apply an additional squaring curve
			fAttLook *= fAngFade;                           // finally apply this to the dist att.

			ints += lgtDat.vCol*fAttLook*BRDF2_ts_nphong_nofr(vN, unit_normal, vL, vV, vMatColDiff, vMatColSpec, fSpecPow);


			++l; uIndex = l<uNrLights ? FetchIndex(uList, l) : 0;
			uLgtType = l<uNrLights ? g_vLightData[uIndex].uLightType : 0;
		}
		
		// specialized loop for sphere lights
		while(l<uNrLights && (uLgtType==SPHERE_LIGHT || uLgtType==CAPSULE_LIGHT))
		{
			SFiniteLightData lgtDat = g_vLightData[uIndex];	
			float3 vLp = lgtDat.vLpos.xyz;

			if(uLgtType==CAPSULE_LIGHT) vLp += clamp(dot(vVPos-vLp, lgtDat.vLdir.xyz), 0, lgtDat.fSegLength) * lgtDat.vLdir.xyz;		// capsule light

			float3 toLight  = vLp - vVPos; 
			float dist = length(toLight);

			float3 vL = toLight / dist;
			float fAttLook = saturate(dist * lgtDat.fInvRange + lgtDat.fNearRadiusOverRange_LP0);

			ints += lgtDat.vCol*fAttLook*BRDF2_ts_nphong_nofr(vN, unit_normal, vL, vV, vMatColDiff, vMatColSpec, fSpecPow);


			++l; uIndex = l<uNrLights ? FetchIndex(uList, l) : 0;
			uLgtType = l<uNrLights ? g_vLightData[uIndex].uLightType : 0;
		}

		// specialized loop for box lights
		while(l<uNrLights && uLgtType==BOX_LIGHT)
		{
			SFiniteLightData lgtDat = g_vLightData[uIndex];	
		
			float3 vBoxAxisY = lgtDat.vLdir.xyz;
			float3 toLight  = lgtDat.vLpos.xyz - vVPos;

			float3 dist = float3( dot(toLight, lgtDat.vBoxAxisX), dot(toLight, vBoxAxisY), dot(toLight, lgtDat.vBoxAxisZ) );
			dist = abs(dist) - lgtDat.vBoxInnerDist;
			dist = saturate(dist * lgtDat.vBoxInvRange);

			float3 vL = normalize(toLight);
			float fAttLook = max(max(dist.x, dist.y), dist.z);

			// arb ramp
			fAttLook = 1-fAttLook;
			fAttLook = fAttLook*fAttLook*(3-2*fAttLook);
			

			ints += lgtDat.vCol*fAttLook*BRDF2_ts_nphong_nofr(vN, unit_normal, vL, vV, vMatColDiff, vMatColSpec, fSpecPow);


			++l; uIndex = l<uNrLights ? FetchIndex(uList, l) : 0;
			uLgtType = l<uNrLights ? g_vLightData[uIndex].uLightType : 0;
		}

#if !defined(XBONE) && !defined(PLAYSTATION4)
		if(uLgtType>=MAX_TYPES) ++l;
#endif
	}

	return ints;
}