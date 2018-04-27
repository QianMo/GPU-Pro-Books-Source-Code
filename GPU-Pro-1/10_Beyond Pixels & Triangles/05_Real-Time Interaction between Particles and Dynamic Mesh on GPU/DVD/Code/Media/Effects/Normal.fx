Texture2D NormalTex;
Texture2D DiffuseTex;

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
	float	ambient		= 0.25;
	float4	specColor	= float4(2,0.5,0.5,1);
};


struct VSIn
{
	float4 pos 		: POSITION;
	float3 norm		: NORMAL;
	float3 tang 	: TANGENT;
	float2 texc		: TEXCOORD;
#ifdef MD_RENDER_TO_CUBEMAP
	uint instID		: SV_InstanceID;
#endif
};

struct PSIn
{
	float4 	pos 	: SV_Position;
	float3 	tlight	: LIGHT;		// tangent space light
	float3 	thalf	: HALF;			// tangent space half-vec
	float2 	texc	: TEXCOORD;
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

	o.i.pos 	= mul(i.pos, matTrans);
	o.i.texc  	= i.texc;

	float3x3 toTBN 	= float3x3( i.tang, cross(i.tang, i.norm), i.norm );
	o.i.tlight		= normalize(mul(toTBN, objLightDir));
	o.i.thalf		= normalize(mul(toTBN, objLightDir+normalize(objCamPos-i.pos)));

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


float4 main_ps( PSIn i) :SV_Target0
{


#ifdef MD_PACKED_NORMALS
#define MD_MASK gaa	
#else
#define MD_MASK rgb
#endif

	float3 norm = NormalTex.Sample(defSampler, i.texc).MD_MASK*(2).xxx-(1).xxx;


	norm.z = sqrt( saturate( 1 - dot( norm.xy, norm.xy ) ) );

	float lit	= saturate(dot(i.tlight, norm));
	float spec	= pow(saturate(dot(normalize(i.thalf), norm)), specPower);

	if(lit==0) spec = 0;

	float4 colr = DiffuseTex.Sample(defSampler, i.texc)*0.85;
	float4 litColr = (lit.xxxx*colr+float4(specColor*spec.xxx,0));
	return lerp(litColr, colr, ambient);
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