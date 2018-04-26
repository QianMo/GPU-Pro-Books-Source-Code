#define SHD_VOL_EXTRUDE 150.f
#define SHD_VOL_BIAS 0.000f

//-----------------------------------------------------------------------------
// zfill states
//-----------------------------------------------------------------------------
RasterizerState rsZFILL {
	CULLMODE				= BACK;
	FILLMODE				= SOLID;
	FRONTCOUNTERCLOCKWISE	= FALSE;
	DEPTHBIAS				= 0;
	DEPTHBIASCLAMP			= 0;
	SLOPESCALEDDEPTHBIAS	= 0;
	DEPTHCLIPENABLE			= TRUE;
	SCISSORENABLE			= FALSE;
	MULTISAMPLEENABLE		= FALSE;
	ANTIALIASEDLINEENABLE	= FALSE;
};

DepthStencilState dssZFILL {
	DEPTHENABLE			= TRUE;
	DEPTHWRITEMASK		= ALL;
	DEPTHFUNC			= LESS;
	STENCILENABLE		= FALSE;
};

BlendState bsZFILL {
	ALPHATOCOVERAGEENABLE	= FALSE;
	BLENDENABLE[0]			= FALSE;
	RENDERTARGETWRITEMASK[0]= 0x0;
};

//-----------------------------------------------------------------------------
// volume pass states
//-----------------------------------------------------------------------------
RasterizerState rsVolumeStencilPass {
	CULLMODE				= NONE;
	FILLMODE				= SOLID;
	FRONTCOUNTERCLOCKWISE	= FALSE;
	DEPTHBIAS				= 0;
	DEPTHBIASCLAMP			= 0;
	SLOPESCALEDDEPTHBIAS	= 0;
	DEPTHCLIPENABLE			= TRUE;
	SCISSORENABLE			= FALSE;
	MULTISAMPLEENABLE		= FALSE;
	ANTIALIASEDLINEENABLE	= FALSE;
};


DepthStencilState dssVolumeStencilPass
{
	DEPTHENABLE			= TRUE;
    DEPTHWRITEMASK		= ZERO;
    DEPTHFUNC			= LESS;
    
    // Setup stencil states
    STENCILENABLE		= TRUE;
    STENCILREADMASK		= 0xFFFFFFFF;
    STENCILWRITEMASK	= 0xFFFFFFFF;
    
    // back facing stencil operations
    BACKFACESTENCILFUNC			= ALWAYS;		//always pass stencil test
    BACKFACESTENCILDEPTHFAIL	= INCR;			//increment stencil if back facing triangles fail the depth test
    BACKFACESTENCILPASS			= KEEP;			//don't modify the stencil when stencil testing passes
    BACKFACESTENCILFAIL			= Keep;			//don't modify the stencil when stencil testing fails
    
    FRONTFACESTENCILFUNC		= ALWAYS;		//always pass stencil test
    FRONTFACESTENCILDEPTHFAIL	= DECR;			//decrement stencil if back facing triangles fail the depth test
    FRONTFACESTENCILPASS		= KEEP;			//don't modify the stencil when stencil testing passes
    FRONTFACESTENCILFAIL		= KEEP;			//don't modify the stencil when stencil testing fails
};

//-----------------------------------------------------------------------------
// light pass states
//-----------------------------------------------------------------------------
RasterizerState rsLightPass {
	CULLMODE				= BACK;
	FILLMODE				= SOLID;
	FRONTCOUNTERCLOCKWISE	= FALSE;
	DEPTHBIAS				= 0;
	DEPTHBIASCLAMP			= 0;
	SLOPESCALEDDEPTHBIAS	= 0;
	DEPTHCLIPENABLE			= TRUE;
	SCISSORENABLE			= FALSE;
	MULTISAMPLEENABLE		= FALSE;
	ANTIALIASEDLINEENABLE	= FALSE;
};

DepthStencilState dssLightPass {
	DEPTHENABLE			= TRUE;
	DEPTHWRITEMASK		= ZERO;
	DEPTHFUNC			= LESS_EQUAL;

	StencilEnable		= TRUE;
    StencilReadMask		= 0xFFFFFFFF;
    StencilWriteMask	= 0x0;
    
    
	FRONTFACESTENCILFUNC= EQUAL;
	FRONTFACESTENCILPASS= KEEP;
	FRONTFACESTENCILFAIL= ZERO;
	
	BackFaceStencilFunc = Never;
    BackFaceStencilPass = Zero;
    BackFaceStencilFail = Zero;
};

BlendState bsLightPass {
	ALPHATOCOVERAGEENABLE	= FALSE;
	BLENDENABLE[0]			= TRUE;
	RENDERTARGETWRITEMASK[0]= 0xF;
	SRCBLEND				= ONE;
	DESTBLEND				= ONE;
	BLENDOP					= ADD;
};

DepthStencilState RenderShadows
{
    DepthEnable = true;
    DepthWriteMask = ZERO;
    DepthFunc = Less_Equal;
    
    StencilEnable		= true;
    StencilReadMask		= 0xFFFFFFFF;
    StencilWriteMask	= 0x0;
    
    FrontFaceStencilFunc = Not_equal;
    FrontFaceStencilPass = Keep;
    FrontFaceStencilFail = Zero;
    
    BackFaceStencilFunc = Never;
    BackFaceStencilPass = Zero;
    BackFaceStencilFail = Zero;
};


//-----------------------------------------------------------------------------
// ambient pass states
//-----------------------------------------------------------------------------
DepthStencilState dssAmbientPass {
	DEPTHENABLE			= TRUE;
	DEPTHWRITEMASK		= ZERO;
	DEPTHFUNC			= LESS_EQUAL;
	StencilEnable		= FALSE;
};

BlendState bsAmbientPass {
	ALPHATOCOVERAGEENABLE	= FALSE;
	BLENDENABLE[0]			= FALSE;
	RENDERTARGETWRITEMASK[0]= 0xF;
	
};


Texture2D<float4> diffTexture;
Texture2D<float> itemHistogram;

sampler DiffuseSampler
{
	//Filter		= MIN_LINEAR_MAG_POINT_MIP_LINEAR;
	Filter		= ANISOTROPIC;
	MaxAnisotropy=16;
	AddressU	= WRAP;
	AddressV	= WRAP;
	MipLODBias	= 0;
	MinLOD		= 0;
	MaxLOD		= 1024;
};


DepthStencilState dssVolumeVisualize {
	DEPTHENABLE			= FALSE;
	DEPTHWRITEMASK		= 0;
	DEPTHFUNC			= LESS;
	STENCILENABLE		= FALSE;
};



//-----------------------------------------------------------------------------
// constant buffers
//-----------------------------------------------------------------------------
cbuffer cbPerObject 
{
	float4x4 mObject2WorldBox;
	float4x4 mNormal2WorldBox;
};

cbuffer cbPerCasterObject2World {
	float4x4 mObject2WorldCaster[ 512 ];
};

cbuffer cbPerCasterNormal2World {
	float4x4 mNormal2WorldCaster[ 512 ];
};


cbuffer cbCamera
{
	float4x4 mViewProjMat;
};

cbuffer cbLight
{
	float4 vLightPos;
	float4 vLightIntensity;
	float4 vEyePos;
};

cbuffer cbHistogramDimensions
{
	float4 vHistogramDimension;
};

//-----------------------------------------------------------------------------
// vertex declarations for ZFILL
//-----------------------------------------------------------------------------
struct VS_ZFILL
{
	float4 Position : POSITION0;
};

struct PS_ZFILL
{
	float4 Position : SV_POSITION;
};


//-----------------------------------------------------------------------------
// vertex declarations for the volume 2 stencil pass
//-----------------------------------------------------------------------------
struct VS_VOLUMETOSTENCIL
{
	float4 Position	: POSITION0;
	uint InstanceID	: SV_INSTANCEID;
};

struct GS_VOLUMETOSTENCIL
{
	float4 Position : POSITION0;
	uint InstanceID	: TEXCOORD0;
};

struct PS_VOLUMETOSTENCIL
{
	float4 Position : SV_POSITION;
	bool frontFace	: SV_IsFrontFace;
};

//-----------------------------------------------------------------------------
// vertex declarations for light pass (wooden box)
//-----------------------------------------------------------------------------
struct VS_LIGHT
{
	float4	Position	: POSITION0;
	float4	Normal		: NORMAL0;
	float2	Texcoord	: TEXCOORD0;
};

struct PS_LIGHT 
{
	float4 Position		: SV_POSITION;
	float4 WorldPos		: POSITION0;
	float4 Normal		: NORMAL0;
	float2 Texcoord		: TEXCOORD0;
};

//-----------------------------------------------------------------------------
// vertex declarations for light pass (shadow caster)
//-----------------------------------------------------------------------------
struct VS_LIGHT_CASTER 
{
	float4	Position	: POSITION0;
	float4	Normal		: NORMAL0;
};

struct PS_LIGHT_CASTER
{
	float4	Position	: SV_POSITION;
	float4	WorldPos	: POSITION0;
	float4	Normal		: NORMAL0;
};

//-----------------------------------------------------------------------------
// vertex declarations for item buffer build pass
//-----------------------------------------------------------------------------
struct VS_ITEM_BUFFER_CASTER 
{
	float4 Position	: POSITION0;	
};

struct PS_ITEM_BUFFER_CASTER
{
	float4 Position						: SV_POSITION;
	nointerpolation float2 HistogramBin : TEXCOORD0;
};


//-----------------------------------------------------------------------------
// zfill shaders
//-----------------------------------------------------------------------------
PS_ZFILL vsZFILLBox( in VS_ZFILL vsIn ) {
	PS_ZFILL vsOut = (PS_ZFILL)0;
	
	vsOut.Position	= mul( vsIn.Position, mObject2WorldBox );
	vsOut.Position	= mul( vsOut.Position, mViewProjMat );
	
	return vsOut;
}

float4 psZFILLBox( in PS_ZFILL psIn ) : SV_TARGET0 {
	return 1;
}

PS_ZFILL vsZFILLCaster( in VS_ZFILL vsIn, in uint instanceID : SV_INSTANCEID ) {
	PS_ZFILL vsOut = (PS_ZFILL)0;
	
	vsOut.Position	= mul( vsIn.Position, mObject2WorldCaster[instanceID] );
	vsOut.Position	= mul( vsOut.Position, mViewProjMat );
	
	return vsOut;
}

float4 psZFILLCaster( in PS_ZFILL vsIn ) : SV_TARGET0 {
	return 1;
}

//-----------------------------------------------------------------------------
// volume to stencil shaders
//-----------------------------------------------------------------------------
void createShadowGeometry( 
	in float4 v0, 
	in float4 v1, 
	in float4 adj, 
	inout TriangleStream<PS_VOLUMETOSTENCIL> shadowVolume )
{
	// normal of adjacent triangle
	//float3 adjNormal	= cross( adj.xyz - v0.xyz, v0.xyz - v1.xyz );
	
	
	float3 adjNormal	= cross( v1.xyz - adj.xyz, v0.xyz - adj.xyz );
	
	float3 adj2Light	= ( v0.xyz - vLightPos.xyz );
	
	if( dot( adjNormal, adj2Light ) < 0 )
		return;
	
	
		
	float3 d0	= normalize( v0.xyz - vLightPos.xyz );
	float3 d1	= normalize( v1.xyz - vLightPos.xyz );
	
	
	float4 tmp[4];
	tmp[0]		= v0 + SHD_VOL_BIAS * float4(d0,0);
	tmp[1]		= v0 + SHD_VOL_EXTRUDE * float4( d0, 0 );
	tmp[2]		= v1 + SHD_VOL_BIAS * float4(d1,0);
	tmp[3]		= v1 + SHD_VOL_EXTRUDE * float4( d1, 0 );
	
	
	[unroll]
	for( int i = 0; i < 4; ++i ) {
		PS_VOLUMETOSTENCIL v2s	= (PS_VOLUMETOSTENCIL)0;
		v2s.Position			= mul( tmp[i], mViewProjMat );
		shadowVolume.Append( v2s );
	}
	shadowVolume.RestartStrip();
}

void createShadowVolume( 
	triangleadj GS_VOLUMETOSTENCIL gsIn[6],
	inout TriangleStream<PS_VOLUMETOSTENCIL> shadowVolume )
{

	createShadowGeometry( gsIn[0].Position, gsIn[2].Position, gsIn[1].Position, shadowVolume );
	createShadowGeometry( gsIn[2].Position, gsIn[4].Position, gsIn[3].Position, shadowVolume );
	createShadowGeometry( gsIn[4].Position, gsIn[0].Position, gsIn[5].Position, shadowVolume );
	
	 //near cap
    PS_VOLUMETOSTENCIL	gsOut = (PS_VOLUMETOSTENCIL)0;
    for( int v0 = 0; v0 < 6; v0 += 2 )
    {
		float3 extrude	= normalize( gsIn[v0].Position.xyz - vLightPos.xyz );
		float4 pos		= gsIn[v0].Position + SHD_VOL_BIAS * float4(extrude, 0);
		gsOut.Position	= mul( pos, mViewProjMat );
		shadowVolume.Append( gsOut );
    }
    shadowVolume.RestartStrip();
    
    
    
    //far cap (reverse the order)
    for( int v1 = 4; v1 >= 0; v1 -= 2 )
    {
        float3 extrude = normalize( gsIn[v1].Position.xyz - vLightPos.xyz );
    
        float4 pos		= gsIn[v1].Position + SHD_VOL_EXTRUDE * float4(extrude, 0);
        gsOut.Position	= mul( pos, mViewProjMat );
        shadowVolume.Append( gsOut );
    }
    shadowVolume.RestartStrip();
    
}

GS_VOLUMETOSTENCIL vsVolumeToStencilBox( in VS_VOLUMETOSTENCIL vsIn ) {
	GS_VOLUMETOSTENCIL vsOut = (GS_VOLUMETOSTENCIL)0;
	
	vsOut.Position	= mul( float4( vsIn.Position.xyz, 1 ), mObject2WorldBox );
	return vsOut;
}

GS_VOLUMETOSTENCIL vsVolumeToStencilCaster( in VS_VOLUMETOSTENCIL vsIn, in uint InstanceID : SV_INSTANCEID ) {
	GS_VOLUMETOSTENCIL vsOut = (GS_VOLUMETOSTENCIL)0;
	
	vsOut.Position	= mul( float4( vsIn.Position.xyz, 1 ), mObject2WorldCaster[InstanceID] );
	vsOut.InstanceID= InstanceID;
	return vsOut;
}


[maxvertexcount(18)]
void gsVolumeToStencil( triangleadj GS_VOLUMETOSTENCIL gsIn[6], inout TriangleStream<PS_VOLUMETOSTENCIL> gsOut, in uint pid : SV_PRIMITIVEID )
{
	float3 normal	= cross( gsIn[2].Position.xyz - gsIn[0].Position.xyz, gsIn[4].Position.xyz - gsIn[0].Position.xyz );
	
	float3 tri2Light= vLightPos.xyz - gsIn[0].Position.xyz;
	
	if( dot( normal, tri2Light ) > 0 ) {
		createShadowVolume( gsIn, gsOut );
	}
	else {
		return;
	}
	
}

[maxvertexcount(18)]
void gsVolumeToStencilCasterVisibility( triangleadj GS_VOLUMETOSTENCIL gsIn[6], inout TriangleStream<PS_VOLUMETOSTENCIL> gsOut, in uint pid : SV_PRIMITIVEID )
{
	// compute index into histogram
	uint width, height, levels;
	itemHistogram.GetDimensions( 0, width, height, levels );
	
	uint hx	= gsIn[0].InstanceID % width;
	uint hy = gsIn[0].InstanceID / width;
	
	float V = itemHistogram.Load( int3(hx,hy,0) );
	if( V == 0.f )
		return;
	
	float3 normal	= cross( gsIn[2].Position.xyz - gsIn[0].Position.xyz, gsIn[4].Position.xyz - gsIn[0].Position.xyz );
	
	float3 tri2Light= vLightPos.xyz - gsIn[0].Position.xyz;
	
	if( dot( normal, tri2Light ) > 0 ) {
		createShadowVolume( gsIn, gsOut );
	}
	else {
		return;
	}
	
}


float4 psVolumeToStencil( in PS_VOLUMETOSTENCIL vsIn ) : SV_TARGET0 {
	//return float4(0.0045, 0.003, 0.0, 1.f);
	return float4( 0, 0.005, 0, 1 );
}

//-----------------------------------------------------------------------------
// light pass shaders (wood box)
//-----------------------------------------------------------------------------
PS_LIGHT vsLightBox( in VS_LIGHT vsIn ) {
	PS_LIGHT vsOut = (PS_LIGHT)0;
	
	vsOut.WorldPos	= mul( float4( vsIn.Position.xyz, 1 ), mObject2WorldBox );
	vsOut.Position	= mul( vsOut.WorldPos, mViewProjMat );
	vsOut.Normal	= mul( float4( vsIn.Normal.xyz, 0 ), mNormal2WorldBox );
	vsOut.Texcoord	= vsIn.Texcoord;
	
	return vsOut;
}

float4 psLightWoodBox( in PS_LIGHT psIn ) : SV_TARGET0 {
	float4 color= diffTexture.Sample( DiffuseSampler, psIn.Texcoord );
	float3 n	= normalize( psIn.Normal.xyz );
	float3 p	= psIn.WorldPos.xyz;
	float3 l	= normalize( vLightPos.xyz - p );
	float  NdL	= max( 0.f, dot( n, l ) );



	return float4( color.xyz * NdL , 1 );
}

//-----------------------------------------------------------------------------
// light pass shaders (shadow caster box)
//-----------------------------------------------------------------------------
PS_LIGHT_CASTER vsLightCaster( in VS_LIGHT_CASTER vsIn, in uint InstanceID : SV_INSTANCEID ) {
	PS_LIGHT_CASTER vsOut = (PS_LIGHT_CASTER)0;
	
	vsOut.WorldPos	= mul( float4( vsIn.Position.xyz, 1 ), mObject2WorldCaster[InstanceID] );
	vsOut.Position	= mul( vsOut.WorldPos, mViewProjMat );
	vsOut.Normal	= mul( float4( vsIn.Normal.xyz, 0 ), mNormal2WorldCaster[InstanceID] );
	
	return vsOut;
}

float4 psLightCaster( in PS_LIGHT_CASTER psIn ) : SV_TARGET0 {

	float3 n	= normalize( psIn.Normal.xyz );
	float3 p	= psIn.WorldPos.xyz;
	float3 l	= normalize( vLightPos.xyz - p );
	float3 e	= normalize( vEyePos.xyz - p );
	float3 h	= normalize( l + e );
	float NdL	= max( 0.f, dot( n, l ) );
	float NdH	= 0.333f * pow( max( 0.f, dot( n, h ) ), 40 );
	
	return float4(NdH,NdL + NdH,NdH,1);
}

float4 psAmbientCaster( in PS_LIGHT_CASTER psIn ) : SV_TARGET0 {
	return float4(0,0.1,0,1);
}
//-----------------------------------------------------------------------------
// ambient pass shaders
//-----------------------------------------------------------------------------
PS_LIGHT vsAmbientBox( in VS_LIGHT vsIn ) {
	PS_LIGHT vsOut	= (PS_LIGHT)0;
	
	vsOut.WorldPos	= mul( float4( vsIn.Position.xyz, 1 ), mObject2WorldBox );
	vsOut.Position	= mul( vsOut.WorldPos, mViewProjMat );
	vsOut.Normal	= mul( float4( vsIn.Normal.xyz, 0 ), mNormal2WorldBox );
	vsOut.Texcoord	= vsIn.Texcoord;
	
	return vsOut;
}

float4 psAmbientWoodBox( in PS_LIGHT psIn ) : SV_TARGET0 {
	float4 color = 0.2f * diffTexture.Sample( DiffuseSampler, psIn.Texcoord );
	
	float x = saturate( 1.0f - ( psIn.WorldPos.z * 0.4f + psIn.WorldPos.x ) * 0.05f );
	x		= smoothstep( 0, 1, x ) * 0.5f + 0.5f;
	
	return float4(color.xyz * x, 1 );
}

//-----------------------------------------------------------------------------
// item buffer shader
//-----------------------------------------------------------------------------
PS_ITEM_BUFFER_CASTER vsItemBufferCaster( in VS_ITEM_BUFFER_CASTER vsIn, uint InstanceID : SV_INSTANCEID ) {
	PS_ITEM_BUFFER_CASTER vsOut = (PS_ITEM_BUFFER_CASTER)0;
	
	vsOut.Position	= mul( vsIn.Position, mObject2WorldCaster[InstanceID] );
	vsOut.Position	= mul( vsOut.Position, mViewProjMat );
	
	// instead of computing and storing an ID we
	// directly compute the histogram bin
	
	float iid			= (float)InstanceID;
	float tmp			= iid * vHistogramDimension.z;	// tmp = InstanceID / histoWidth
	float x				= frac( tmp );
	float y				= floor( tmp ) * vHistogramDimension.w;	// tmp = InstanceID / histoHeight
	vsOut.HistogramBin	= -1 + 2.f * float2(x,y) + vHistogramDimension.zw;
	
	return vsOut;
}

half2 psItemBufferCaster( in PS_ITEM_BUFFER_CASTER psIn ) : SV_TARGET0 {
	return half2( psIn.HistogramBin.x,-psIn.HistogramBin.y );
}

//-----------------------------------------------------------------------------
// techniques
//-----------------------------------------------------------------------------

BlendState bsVolumeTest {
	BlendEnable[0]	= TRUE;
	SrcBlend		= ONE;
	DestBlend		= ONE;
	BlendOp			= ADD;
};
technique10 tZFILLBox {
	pass p0 {
		SetVertexShader( CompileShader( vs_4_0, vsZFILLBox() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, psZFILLBox() ) );
		
		SetRasterizerState( rsZFILL );
		SetDepthStencilState( dssZFILL, 0 );
		SetBlendState( bsZFILL, float4(0,0,0,0), 0xFFFFFFFF );
	}
}

technique10 tZFILLCaster {
	pass p0 {
		SetVertexShader( CompileShader( vs_4_0, vsZFILLCaster() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, psZFILLCaster() ) );
		
		SetRasterizerState( rsZFILL );
		SetDepthStencilState( dssZFILL, 0 );
		SetBlendState( bsZFILL, float4(0,0,0,0), 0xFFFFFFFF );
	}
}


technique10 tVolumesToStencilBox {
	pass p0 {
	
		SetVertexShader( CompileShader( vs_4_0, vsVolumeToStencilBox() ) );
		SetGeometryShader( CompileShader( gs_4_0, gsVolumeToStencil() ) );
		SetPixelShader( CompileShader( ps_4_0, psVolumeToStencil() ) );
	
		SetRasterizerState( rsVolumeStencilPass );
		SetDepthStencilState( dssVolumeStencilPass, 1 );
		SetBlendState( bsZFILL, float4(0,0,0,0), 0xFFFFFFFF );
	}
}

technique10 tVolumesToStencilCaster {
	pass p0 {
	
		SetVertexShader( CompileShader( vs_4_0, vsVolumeToStencilCaster() ) );
		SetGeometryShader( CompileShader( gs_4_0, gsVolumeToStencil() ) );
		SetPixelShader( CompileShader( ps_4_0, psVolumeToStencil() ) );
	
		SetRasterizerState( rsVolumeStencilPass );
		SetDepthStencilState( dssVolumeStencilPass, 1 );
		SetBlendState( bsZFILL, float4(0,0,0,0), 0xFFFFFFFF );
	}
}

technique10 tVolumesToStencilCasterVisibility {
	pass p0 {
	
		SetVertexShader( CompileShader( vs_4_0, vsVolumeToStencilCaster() ) );
		SetGeometryShader( CompileShader( gs_4_0, gsVolumeToStencilCasterVisibility() ) );
		SetPixelShader( CompileShader( ps_4_0, psVolumeToStencil() ) );
	
		SetRasterizerState( rsVolumeStencilPass );
		SetDepthStencilState( dssVolumeStencilPass, 1 );
		SetBlendState( bsZFILL, float4(0,0,0,0), 0xFFFFFFFF );
	}
}


technique10 tLightBox {
	pass p0 {
		SetVertexShader( CompileShader( vs_4_0, vsLightBox() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, psLightWoodBox() ) );
		
		SetRasterizerState( rsLightPass );
		SetDepthStencilState( dssLightPass, 0 );
		//SetDepthStencilState( RenderShadows, 0 );
		SetBlendState( bsLightPass, float4(0,0,0,0), 0xFFFFFFFF );
	}
}

technique10 tLightCaster {
	pass p0 {
		SetVertexShader( CompileShader( vs_4_0, vsLightCaster() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, psLightCaster() ) );
		
		SetRasterizerState( rsLightPass );
		SetDepthStencilState( dssLightPass, 0 );
		SetBlendState( bsLightPass, float4(0,0,0,0), 0xFFFFFFFF );
	}
}

technique10 tAmbientBox {
	pass p0 {
		SetVertexShader( CompileShader( vs_4_0, vsAmbientBox()	) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, psAmbientWoodBox() ) );
		
		SetRasterizerState( rsLightPass );
		SetDepthStencilState( dssAmbientPass, 0 );
		SetBlendState( bsAmbientPass, float4(0,0,0,0), 0xFFFFFFFF );
	}
}

technique10 tAmbientCaster {
	pass p0 {
		SetVertexShader( CompileShader( vs_4_0, vsLightCaster() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, psAmbientCaster() ) );
		
		SetRasterizerState( rsLightPass );
		SetDepthStencilState( dssAmbientPass, 0 );
		SetBlendState( bsAmbientPass, float4(0,0,0,0), 0xFFFFFFFF );

	}
}

technique10 tRenderVolumes {
	pass p0 {
		SetVertexShader( CompileShader( vs_4_0, vsVolumeToStencilCaster() ) );
		SetGeometryShader( CompileShader( gs_4_0, gsVolumeToStencil() ) );
		SetPixelShader( CompileShader( ps_4_0, psVolumeToStencil() ) );
		
		SetRasterizerState( rsVolumeStencilPass );
		//SetDepthStencilState( dssZFILL, 0 );
		SetDepthStencilState( dssVolumeVisualize, 0 );
		SetBlendState( bsVolumeTest, float4(0,0,0,0), 0xFFFFFFFF );
	}
}

technique10 tRenderVolumesCasterVisibility {
	pass p0 {
		SetVertexShader( CompileShader( vs_4_0, vsVolumeToStencilCaster() ) );
		SetGeometryShader( CompileShader( gs_4_0, gsVolumeToStencilCasterVisibility() ) );
		SetPixelShader( CompileShader( ps_4_0, psVolumeToStencil() ) );
	
		SetRasterizerState( rsVolumeStencilPass );
		SetDepthStencilState( dssVolumeStencilPass, 1 );
		SetBlendState( bsVolumeTest, float4(0,0,0,0), 0xFFFFFFFF );
	}
}

technique10 tBuildItemBufferCaster {
	pass p0 {
		SetVertexShader( CompileShader( vs_4_0, vsItemBufferCaster() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, psItemBufferCaster() ) );
		
		SetRasterizerState( rsZFILL );
		SetDepthStencilState( dssZFILL, 0 );
		SetBlendState( bsAmbientPass, float4(0,0,0,0), 0xFFFFFFFF );
	}
}