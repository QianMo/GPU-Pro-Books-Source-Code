#include "ML.fxh"

struct VSIn
{
	float4 pos 		: POSITION;
	float3 norm		: NORMAL;
#ifdef MD_RENDER_TO_CUBEMAP
	uint instID		: SV_InstanceID;
#endif
};

struct PSIn
{
	float4 		pos 	: SV_Position;
	float3		view_os	: VIEW_OS;
	float3		norm_os	: NORMAL_OS;
#if MD_NUM_POINT_LIGHTS
	float3		pos_os	: POSITION_OS;
#endif
};

struct VSOut
{
	PSIn i;
#ifdef MD_RENDER_TO_CUBEMAP
	uint instID	: INSTANCE;
#endif
};


#ifdef MD_RENDER_TO_CUBEMAP
typedef VSOut GSIn;
struct GSOut
{
	PSIn i;
	uint vpID : SV_RenderTargetArrayIndex;
};
#endif

VSOut main_vs( VSIn i )
{
	VSOut o;

	float4x4 matTrans;

#ifdef MD_RENDER_TO_CUBEMAP
	matTrans = matCubeMapWVP[i.instID];
#else
	matTrans = matWVP;
#endif

	float3 spos = i.pos * gScale;

	o.i.pos 	= mul(i.pos, matTrans);

	o.i.view_os	= gCamPos_OS - spos;
	o.i.norm_os	= i.norm * gInvScale;
#if MD_NUM_POINT_LIGHTS
	o.i.pos_os	= spos;
#endif

#ifdef MD_RENDER_TO_CUBEMAP
	o.instID		= cubeFaceIndices[i.instID];
#endif
	
	return o;	
};

#ifdef MD_RENDER_TO_CUBEMAP
[maxvertexcount(3)]
void main_gs( triangle GSIn ia[3], inout TriangleStream<GSOut> os )
{
	[unroll]
	for(int i = 0; i < 3; i ++ )
	{
		GSOut o;
		o.i 	= ia[i].i;
		o.vpID	= ia[i].instID;
		os.Append( o );
	}
	os.RestartStrip();
}

#endif

SamplerState defSampler
{
	Filter 			= ANISOTROPIC;
	MaxAnisotropy 	= 8;
};

SamplerState cubemapSampler
{
	Filter 			= MIN_MAG_MIP_LINEAR;
};


float4 main_ps( PSIn i ) : SV_Target0
{

	float3 diffColr		= gDiffColor;
	float3 norm 		= normalize( i.norm_os );
	float3 view			= normalize( i.view_os );


	float3 refColr = 0;

#ifdef MD_DO_REFLECTION
	float3 refVec_OS = reflect( -view, norm );

	float3 refVec = mul( refVec_OS, (float3x3)matW_NS );

	refColr = ReflectTex.Sample( cubemapSampler, refVec ) * gReflection;
#endif

	LightComponents lc;
	Init( lc );

	int ii = 0;

	[unroll]
	for( ; ii < MD_NUM_DIR_LIGHTS; ii ++ )
	{
		CalculateLight( lc, GetDirLight( ii ), norm, view, gSpecPower );
	}

#if MD_NUM_POINT_LIGHTS
	[unroll]
	for( ; ii < MD_NUM_DIR_LIGHTS + MD_NUM_POINT_LIGHTS; ii ++ )
	{
		CalculateLight( lc, GetPointLight( ii , i.pos_os ), norm, view, gSpecPower );
	}
#endif

#ifdef MD_SELF_ILLUMINATE
	float3 finalDiffuse = diffColr * lerp ( lc.diffuse + gAmbient, 1, gSelfIlluminate ) + lc.specular;
#else
	float3 finalDiffuse = diffColr * ( lc.diffuse + gAmbient) + lc.specular + refColr;
#endif

	return float4( finalDiffuse, gDiffColor.a );

}


RasterizerState MSEnabled
{
	MultiSampleEnable = TRUE;
};

RasterizerState Default
{
};


technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs()));
#ifdef MD_RENDER_TO_CUBEMAP
		SetGeometryShader( CompileShader( gs_4_0, main_gs()));
#else
		SetGeometryShader( NULL );
#endif
		SetPixelShader( CompileShader( ps_4_0, main_ps()));

		SetRasterizerState(MSEnabled);
	}

	pass
	{
		SetRasterizerState(Default);
	}
}