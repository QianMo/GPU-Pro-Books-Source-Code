cbuffer perFrame
{
#ifdef MD_RENDER_TO_CUBEMAP
	float4x4 matCubeMapWVP[6];
	uint cubeFaceIndices[6];
#else
	float4x4 matWVP;
#endif

	float3	objLightDir = float3(0,0,1);
	float3 	objCamPos;
	float	specPower 	= 15;
	float4	ambient		= float4( (0.25).xxx, 1 );
	float4	color		= float4(0.7, 0.7, 0.7, 1);
	float4	specColor	= float4(2,0.5,0.5,1);
};


struct VSInput
{
	float4 pos 		: POSITION;
	float3 norm		: NORMAL;
#ifdef MD_RENDER_TO_CUBEMAP
	uint instID		: SV_InstanceID;
#endif
};

struct PSIn
{
	float4 pos 		: SV_Position;
	float3 norm		: NORMAL;
	float3 view 	: VIEW;
	float3 refl		: REFL;
};

struct VSOutput
{
	PSIn i;
#ifdef MD_RENDER_TO_CUBEMAP
	uint instID	: INSTANCE;
#endif
};

#ifdef MD_RENDER_TO_CUBEMAP
typedef VSOutput GSIn;
struct GSOut
{
	PSIn i;
	uint vpID : SV_RenderTargetArrayIndex;
};
#endif


VSOutput main_vs( VSInput i )
{
	VSOutput o;

#ifdef MD_RENDER_TO_CUBEMAP
	float4x4 matWVP = matCubeMapWVP[i.instID];
	o.instID = cubeFaceIndices[i.instID];
#endif

	o.i.pos 		= mul(i.pos, matWVP);

	o.i.norm		= i.norm;
	o.i.view		= objCamPos - i.pos;
	o.i.refl		= reflect( -objLightDir, i.norm );
	
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


float4 main_ps( PSIn i) : SV_Target0
{
	float d = dot( normalize( i.norm ), objLightDir );
	float s = dot( normalize( i.refl ), normalize( i.view ) ) * (d > 0);

	return ambient + color*saturate( d ) + specColor*pow( saturate( s ), specPower );
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