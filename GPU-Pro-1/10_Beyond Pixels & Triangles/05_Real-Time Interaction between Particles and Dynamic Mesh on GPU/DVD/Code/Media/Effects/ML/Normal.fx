#include "ML.fxh"

#if defined(MD_RENDER_TO_CUBEMAP) && defined(MD_BONE_TRANSFORM)
#error 1 pass render to cubemap and boned transforms are incompatible ATM
#endif

#if defined(MD_BONE_TRANSFORM) && defined(MD_VTEXT)
#error MD_BONE_TRANSFORM and MD_VTEXT can't be used togather
#endif

#if defined(MD_RENDER_TO_CUBEMAP) && defined(MD_VTEXT)
#error MD_RENDER_TO_CUBEMAP and MD_VTEXT can't be used togather
#endif

#ifdef MD_BONE_TRANSFORM
#include "Transform.fxh"
#endif

#ifdef MD_VTEXT

struct VSIn
{
	uint bufIdx : VERTEX_IDX;
	uint letIdx : SYMBOL_IDX;
};

#else

struct VSIn
{
	float4 pos 		: POSITION;
	float3 norm		: NORMAL;
	float3 tang 	: TANGENT;
	float2 texc		: TEXCOORD;

	#ifdef MD_BONE_TRANSFORM
	uint4 indices	: BLENDINDICES;
	float4 weights	: BLENDWEIGHT;
	#endif

	#ifdef MD_RENDER_TO_CUBEMAP
	uint instID		: SV_InstanceID;
	#endif
};

#endif

struct PSIn
{
	float4 		pos 	: SV_Position;
	float3x3    TBN		: TBN;
	float3		view_os	: VIEW_OS;
#if MD_NUM_POINT_LIGHTS
	float3		pos_os	: POSITION_OS;
#endif
	float2 		texc	: TEXCOORD;
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


#if defined( MD_VTEXT )

	uint vidx = i.bufIdx;

	float4 position	= FontBuf.Load( vidx++ );

	position.xyz 	*= 	letterPoses[ i.letIdx ].z;
	position.xy 	+= 	letterPoses[ i.letIdx ].xy;

	float3 normal	= FontBuf.Load( vidx++ ).xyz;
	float3 tangent	= FontBuf.Load( vidx++ ).xyz;
	o.i.texc		= FontBuf.Load( vidx ).xy;

#else
	o.i.texc  	= i.texc;

	float4 position = i.pos;
	float3 tangent 	= i.tang;
	float3 normal 	= i.norm;

	#ifdef MD_BONE_TRANSFORM
	BoneTransform( position, tangent, normal, i.indices, i.weights, MD_NUM_WEIGHTS );
	#endif

#endif

#ifdef MD_RENDER_TO_CUBEMAP
	matTrans = matCubeMapWVP[i.instID];
#else
	matTrans = matWVP;
#endif

	o.i.pos 	= mul(position, matTrans);

	float3 spos = position * gScale;

	float3x3 toTBN 	= float3x3( tangent, cross(tangent, normal), normal );

	// NOTE : we're wrighting inverse transformation
	o.i.TBN			= toTBN;


	// we tranform normal to objects space ( to handle multiple lights optimally )
	// then we fix for possible non-uniform scaling

	o.i.TBN._11_21_31 	*= gInvScale.x;
	o.i.TBN._12_22_32 	*= gInvScale.y;
	o.i.TBN._13_23_33 	*= gInvScale.z;
                    
	o.i.view_os		= gCamPos_OS - spos;

#if MD_NUM_POINT_LIGHTS
	o.i.pos_os		= spos;
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
	MaxAnisotropy 	= 4;
};


SamplerState cubemapSampler
{
	Filter 			= MIN_MAG_MIP_LINEAR;
};


SamplerState tileSampler
{
	Filter 			= ANISOTROPIC;
	MaxAnisotropy 	= 4;
	AddressU		= WRAP;
	AddressV		= WRAP;
};

#ifdef MD_TERRAIN
#define MD_SAMPLER tileSampler
#else
#define MD_SAMPLER defSampler
#endif

float4 main_ps( PSIn i ) : SV_Target0
{

	float light_map = 1;

#ifdef MD_TERRAIN
	float sample_w;
	float2( sample_w, light_map ) = WeightTex.Sample( defSampler, i.texc );
	float2 texc = i.texc * 32;
	float2 texc1 = i.texc * 8;
#else
	float2 texc = i.texc;
#endif

	float4 diffColr = DiffuseTex.Sample( MD_SAMPLER, texc ) * light_map;

#ifdef MD_TERRAIN
	diffColr = lerp( diffColr, DiffuseTex1.Sample( MD_SAMPLER, texc1 ), sample_w );
#endif


	float specPowerRaw 	= gSpecPower;
	float3 reflection 	= gReflection;

	float specPower = specPowerRaw;

#ifdef MD_BAKED_SPECULAR
	float3 bakedSpecular;
	float4( bakedSpecular, specPowerRaw ) = SpecularTex.Sample( MD_SAMPLER, texc );

#ifdef MD_TERRAIN
	{
		float3 bakedSpecular1;
		float specPowerRaw1;
		float4( bakedSpecular1, specPowerRaw1 ) = SpecularTex1.Sample( MD_SAMPLER, texc1 );
		bakedSpecular = lerp( bakedSpecular, bakedSpecular1, sample_w );
		specPowerRaw = lerp( specPowerRaw, specPowerRaw1, sample_w );
	}
#endif


#ifdef MD_REFLECTION_MASK
	reflection	= saturate( bakedSpecular - diffColr.a );
	diffColr.a	= 1;
#else
	reflection 	= saturate( bakedSpecular - 0.25 );
#endif

	specPower = specPowerRaw * 255.;

#define UPDATE_LIGHT(l) UpdateLight( l, bakedSpecular )
#else
#define UPDATE_LIGHT(l) (l)
#endif

	float3 norm_ts = NormalTex.Sample( MD_SAMPLER, texc ).gaa*(2).xxx-(1).xxx;

#ifdef MD_TERRAIN
	norm_ts.xy = lerp( norm_ts.xy, NormalTex1.Sample( MD_SAMPLER, texc1 ).ga*(2).xx-(1).xx, sample_w );
#endif
	
	norm_ts.z = sqrt( saturate( 1 - dot( norm_ts.xy, norm_ts.xy ) ) );


	// have to normalize after transfrom cause there might be scale in there
	float3 norm = normalize( mul( norm_ts, i.TBN ) );

	float3 view = normalize( i.view_os );


	float3 refColr = 0;

#ifdef MD_DO_REFLECTION
	float3 refVec_OS = reflect( -view, norm );

	float3 refVec = mul( refVec_OS, (float3x3)matW_NS );

#ifdef MD_PS_4_1
	float LOD = ReflectTex.CalculateLevelOfDetailUnclamped( cubemapSampler, refVec );
#define SAMPLE SampleLevel( cubemapSampler, refVec, LOD )
#else
	float3 dx = ddx( refVec );
	float3 dy = ddy( refVec );

#define SAMPLE SampleGrad( cubemapSampler, refVec, dx, dy )
#endif
	[branch]
	if( asint( dot( reflection, 1 ) ) )
	{
		float refBlur;

#ifdef MD_BLUR_REFLECTION
	#ifdef MD_BAKED_SPECULAR
		refBlur 	= 1 + 16 * saturate( 0.5 - specPowerRaw );
	#else
		refBlur 	= gRefBlur;	
	#endif

	#ifdef MD_PS_4_1
		LOD += refBlur;
	#else
		dx *= refBlur;
		dy *= refBlur;
	#endif
#endif
		refColr = ReflectTex.SAMPLE * reflection;
	}
#endif

	LightComponents lc;
	Init( lc );

	int ii = 0;

	[unroll]
	for( ; ii < MD_NUM_DIR_LIGHTS; ii ++ )
	{
		CalculateLight( lc, UPDATE_LIGHT( GetDirLight( ii ) ), norm, view, specPower );
	}

#if MD_NUM_POINT_LIGHTS
	[unroll]
	for( ; ii < MD_NUM_DIR_LIGHTS + MD_NUM_POINT_LIGHTS; ii ++ )
	{
		CalculateLight( lc, UPDATE_LIGHT( GetPointLight( ii, i.pos_os ) ), norm, view, specPower );
	}
#endif

#ifdef MD_TERRAIN
	float overbright = dot(lc.specular,0.5);
#else
	float overbright = dot(lc.specular,2);
#endif

	return float4( diffColr * ( lc.diffuse + gAmbient) + lc.specular * light_map + refColr, overbright );
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