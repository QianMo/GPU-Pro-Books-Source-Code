#include "ML.fxh"

struct VSIn
{
	float4 pos		: POSITION;
	float3 norm		: NORMAL;
	float3 tang		: TANGENT;
	float2 texc		: TEXCOORD;
};

struct VSOut
{
	float4 		pos 	: SV_Position;
	float3x3    TBN		: TBN;
	float3		view_os	: VIEW_OS;
#if MD_NUM_POINT_LIGHTS
	float3		pos_os	: POSITION_OS;
#endif
	float2 		texc	: TEXCOORD;
	float4		screen	: SCREEN;
};

typedef VSOut PSIn;

// simplified refraction routing( returns unnormalized v )
float3 Refract( float3 v, float3 n, float k )
{
	float dpres = dot( v, n );
	float3 pv =  dpres * n;
	float3 d = pv - v;

	[flatten]
	if( dpres < -0.0001 )
		return v + k * d;
	else       
		return v;
}

float3 GetRefrDsp( float3 v, float3 n )
{
	float dpres = dot( v, n );
	float3 pv =  dpres * n;
	return pv - v;
}



VSOut main_vs( VSIn i )
{
	VSOut o;

	o.texc  	= i.texc;

	float4 position = i.pos;
	float3 tangent 	= i.tang;
	float3 normal 	= i.norm;

	o.pos 	= mul( position, matWVP );

	float3 spos = position * gScale;

	float3x3 toTBN 	= float3x3( tangent, cross(tangent, normal), normal );

	// NOTE : we're wrighting inverse transformation
	o.TBN			= toTBN;


	// we tranform normal to objects space ( to handle multiple lights optimally )
	// then we fix for possible non-uniform scaling

	o.TBN._11_21_31 	*= gInvScale.x;
	o.TBN._12_22_32 	*= gInvScale.y;
	o.TBN._13_23_33 	*= gInvScale.z;
                    
	o.view_os		= gCamPos_OS - spos;

#if MD_NUM_POINT_LIGHTS
	o.pos_os		= spos;
#endif
	o.screen		= float4( o.pos.x + o.pos.w, -o.pos.y + o.pos.w, o.pos.z, o.pos.w * 2 );

	return o;	
};

SamplerState defSampler
{
	Filter 			= ANISOTROPIC;
	MaxAnisotropy 	= 4;
};


SamplerState linearSampler
{
	Filter 			= MIN_MAG_MIP_LINEAR;
	AddressU		= CLAMP;
	AddressV		= CLAMP;
};


SamplerState tileSampler
{
	Filter 			= ANISOTROPIC;
	MaxAnisotropy 	= 4;
	AddressU		= WRAP;
	AddressV		= WRAP;
};

float4 main_ps( PSIn i ) : SV_Target0
{
	float2 sceneTexc = i.screen.xy / i.screen.w;

	float2 texc = i.texc;

	float4 diffColr = gDiffColor;

	float specPowerRaw 	= gSpecPower;
	float3 reflection 	= 1;

	float specPower = specPowerRaw;

	float lightness;
	float weight;

	float2(weight,lightness) = WeightTex.Sample( defSampler, texc );

	float2 scale0, scale1;

	float2 dsp0, dsp1;

	float shore = 1 - saturate( weight * 8 );

	[flatten]
	if( weight < 0.5 )
	{
		weight *= 2;
		dsp0 = float2( 0, 0 );
		dsp1 = float2( 0, gFlowTime*0.5 );

		scale0 = float2( 24, 24 );
		scale1 = float2( 16, 8 );
	}
	else
	{
		weight = (weight - 0.5)*2;
		dsp0 = float2( 0, gFlowTime*0.5 );
		dsp1 = float2( 0, gFlowTime );

		scale0 = float2( 16, 8 );
		scale1 = float2( 16, 4 );

	}

	float3 norm_ts0 = NormalTex.SampleLevel( tileSampler, texc * scale0 + dsp0, 0 ).gaa*(2).xxx-(1).xxx;
	norm_ts0.z = sqrt( saturate( 1 - dot( norm_ts0.xy, norm_ts0.xy ) ) );


	float3 norm_ts1 = NormalTex.SampleLevel( tileSampler, texc * scale1 + dsp1, 0 ).gaa*(2).xxx-(1).xxx;
	norm_ts1.z = sqrt( saturate( 1 - dot( norm_ts1.xy, norm_ts1.xy ) ) );

	float3 norm_ts = lerp( norm_ts0, norm_ts1, weight );

	// have to normalize after transfrom cause there might be scale in there
	float3 norm = normalize( mul( norm_ts, i.TBN ) );

	float3 view = normalize( i.view_os );

	float3 reflVec_OS = reflect( -view, norm );

	float3 reflVec = mul( reflVec_OS, (float3x3)matW_NS );

	float3 reflColr = ReflectTex.Sample( linearSampler, reflVec ) * reflection;

#ifdef MD_CUBEMAPREFRACTION
	float3 refrVec_OS =  Refract( normalize(-view), norm, 0.05 );
	float3 refrVec = mul( refrVec_OS, (float3x3)matW_NS );

	float3 refrColr = ReflectTex.SampleLevel( linearSampler, refrVec, 0 );
#else
	float3 refrVec_OS =  GetRefrDsp( normalize(-view), norm );
	float2 refrVec = mul( refrVec_OS, (float3x3)matWV_NS );

	float3 refrColr = RefractTex.SampleLevel( linearSampler, sceneTexc + float2(refrVec.x, -refrVec.y)*lerp(0.03125,0,shore), 0 );
#endif

	float fres = saturate( dot( view, norm ) + 0.33 );

	float3 waterColor = lerp( reflColr*lightness, refrColr, lerp( fres, 1, shore ) ) * lerp( diffColr, 1, shore );

	LightComponents lc;
	Init( lc );

	int ii = 0;

	[unroll]
	for( ; ii < MD_NUM_DIR_LIGHTS; ii ++ )
	{
		CalculateLight( lc, GetDirLight( ii ), norm, view, specPower );
	}

#if MD_NUM_POINT_LIGHTS
	[unroll]
	for( ; ii < MD_NUM_DIR_LIGHTS + MD_NUM_POINT_LIGHTS; ii ++ )
	{
		CalculateLight( lc, GetPointLight( ii, i.pos_os ), norm, view, specPower );
	}
#endif

	lc.specular = lerp( lc.specular, 0, shore );

	return float4( waterColor + lc.specular, dot(lc.specular,1) );
}

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs()));
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps()));
	}
}