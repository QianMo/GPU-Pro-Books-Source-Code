// ------------------------------------------------------------------------------------------------
// State Objects
// ------------------------------------------------------------------------------------------------
DepthStencilState dssDisableDepth
{
	DepthEnable	= FALSE;
};

BlendState bsDisableBlending
{
	BlendEnable[ 0 ]	= FALSE;
};

BlendState bsAdditiveBlending
{
	BlendEnable[ 0 ]	= TRUE;
	SrcBlend			= ONE;
	DestBlend			= ONE;
	SrcBlendAlpha		= ZERO;
	DestBlendAlpha		= ZERO;
	BlendOp				= ADD;
	BlendOpAlpha		= MIN;
	RenderTargetWriteMask[ 0 ]	= 0x00000001;
};

SamplerState ssPointSampler
{
	AddressU		= CLAMP;
	AddressV		= CLAMP;
	BorderColor		= float4( 0.5, 0.5, 0.5, 0.5 );
	Filter			= MIN_MAG_MIP_POINT;
};

// ------------------------------------------------------------------------------------------------
// Texture Objects
// ------------------------------------------------------------------------------------------------
Texture2D<half2> tInputTexture;


// ------------------------------------------------------------------------------------------------
// Constant Buffers
// ------------------------------------------------------------------------------------------------
cbuffer cbImageDimensions
{
	float	fInvHeight;
};

// ------------------------------------------------------------------------------------------------
// Vertex Declrations
// ------------------------------------------------------------------------------------------------
struct VS_R2VB
{
	float	pixel		: POSITION0;
	uint	vertexID	: SV_VERTEXID;
};

struct PS_R2VB
{
	float4 position		: SV_POSITION;
	float texcoord		: TEXCOORD0;
};

struct VS_SCATTER
{
	float2 position	: POSITION0;	
};

struct PS_SCATTER
{
	float4 position	: SV_POSITION;
};


struct VS_SCATTER3 {
	uint2	pixelIndex : POSITION0;
};

// ------------------------------------------------------------------------------------------------
// Shader
// ------------------------------------------------------------------------------------------------
PS_R2VB vsR2VB( in VS_R2VB vsIn )
{
	PS_R2VB vsOut	= (PS_R2VB)0;
	vsOut.texcoord	= vsIn.pixel;
	
	if( vsIn.vertexID == 0 )
		vsOut.position = float4( -1, 0, 0, 1 );
	else
		vsOut.position = float4( 1, 0, 0, 1 );
		
	return vsOut;
}

half2 psR2VB( in PS_R2VB psIn ) : SV_TARGET0
{
	float2 uv = float2( frac( psIn.texcoord ), ( 0.5f + floor( psIn.texcoord ) ) * fInvHeight );
	return tInputTexture.SampleLevel( ssPointSampler, uv, 0.f );
}



PS_SCATTER vsScatter2( in uint VertexID : SV_VERTEXID ) {
	uint w,h;
	tInputTexture.GetDimensions( w, h );
	
	uint x = VertexID % w;
	uint y = VertexID / w;
	
	half2 tmp = tInputTexture.Load( int3( x,y,0 ) );
	
	PS_SCATTER vsOut = (PS_SCATTER)0;
	vsOut.position = float4( tmp, 0, 1 );
	return vsOut;
}

PS_SCATTER vsScatter3( in VS_SCATTER3 vsIn ) {
	//uint x = vsIn.pixelIndex & 0xFFFF;
	//uint y = vsIn.pixelIndex >> 16;
	
	half2 tmp = tInputTexture.Load( int3( vsIn.pixelIndex, 0 ) );

	PS_SCATTER vsOut = (PS_SCATTER)0;
	vsOut.position = float4( tmp, 0, 1 );
	return vsOut;
	
}

PS_SCATTER vsScatter( in VS_SCATTER vsIn )
{
	PS_SCATTER vsOut = (PS_SCATTER)0;
	
	vsOut.position	= float4( vsIn.position.xy, 0, 1 );
	
	return vsOut;
}



float psScatter( in PS_SCATTER psIn ) : SV_TARGET0
{
	return 1;
}

// ------------------------------------------------------------------------------------------------
// Techniques
// ------------------------------------------------------------------------------------------------
technique10 tCopy2VertexBuffer
{
	pass p0
	{
		SetVertexShader( CompileShader( vs_4_0, vsR2VB() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, psR2VB() ) );
		
		
		SetDepthStencilState( dssDisableDepth, 0 );
		SetBlendState( bsDisableBlending, float4( 0, 0, 0, 0 ), 0xFFFFFFFF );
	}
}

technique10 tScatterToBuckets
{
	pass p0
	{
		SetVertexShader( CompileShader( vs_4_0, vsScatter() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, psScatter() ) );
		
		SetDepthStencilState( dssDisableDepth, 0 );
		SetBlendState( bsAdditiveBlending, float4( 0, 0, 0, 0 ), 0xFFFFFFFF );
	}
}

technique10 tScatterToBuckets2
{
	pass p0
	{
		SetVertexShader( CompileShader( vs_4_0, vsScatter2() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, psScatter() ) );
		
		SetDepthStencilState( dssDisableDepth, 0 );
		SetBlendState( bsAdditiveBlending, float4( 0, 0, 0, 0 ), 0xFFFFFFFF );
	}
}

technique10 tScatterToBuckets3
{
	pass p0
	{
		SetVertexShader( CompileShader( vs_4_0, vsScatter3() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, psScatter() ) );
		
		SetDepthStencilState( dssDisableDepth, 0 );
		SetBlendState( bsAdditiveBlending, float4( 0, 0, 0, 0 ), 0xFFFFFFFF );
	}
}