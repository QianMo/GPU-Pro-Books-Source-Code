
//--------------------------------------------------------------------------------------
// Constant Buffer Variables.
//--------------------------------------------------------------------------------------

///< Never changes for now.
cbuffer cbNeverChanges
{
	matrix World 			: WORLD;
	matrix View 			: VIEW;
	matrix ViewInverse 		: VIEWINVERSE;
	matrix Projection 		: PROJECTION;
	
	float3 BlowerPosition : BLOWER_POSITION<
		string UIName="Blower position";
	> = float3(0.0f, 0.0f, 0.0f);
	
	float3 BlowerVelocity : BLOWER_VELOCITY<
		string UIName="Blower velocity";
	> = float3(0.0f, 0.5f, 0.5f);


}

float3 FieldStep 	: FIELDSIZE < >		= 1.0f/32.0f;

//--------------------------------------------------------------------------------------
// Textures and Samplers.
//--------------------------------------------------------------------------------------

Texture3D FieldT : FIELD;
Texture2D HaloT : HALO;

SamplerState PointSampler
{
    AddressU  	= Clamp;
    AddressV 	= Clamp;
    AddressW 	= Clamp;
	
	Filter=MIN_MAG_MIP_POINT;
};

SamplerState LinearSampler
{
    AddressU  	= Clamp;
    AddressV 	= Clamp;
    AddressW 	= Clamp;   
	
	Filter=MIN_MAG_LINEAR_MIP_POINT;
};




///<
cbuffer cbImmutable
{
	///<
	const float3 Directions[6]
	<
		string UIWidget="None";
	>
	=
	{
		float3(1,0,0),
		float3(0,1,0),
		float3(-1,0,0),
		float3(0,-1,0),
		float3(0,0,1),
		float3(0,0,-1)	
	};
	
    const float3 Positions[4]
	<
		string UIWidget="None";
	>
	=
    {
        float3( -1, 1, 0 ),
        float3( 1, 1, 0 ),
        float3( -1, -1, 0 ),
        float3( 1, -1, 0 ),
    };
	
	const float3 Normals[4]
	<
		string UIWidget="None";
	>
	=
    {
        float3( -0.8f, 0.8f, 0.8f ),
        float3( 0.8f, 0.8f, 0.8f ),
        float3( -0.8f, -0.8f, 0.8f ),
        float3( 0.8f, -0.8f, 0.8f ),
    };
	
    const float2 TexCoords[4]
	<
		string UIWidget="None";
	>
	= 
    { 
        float2(0,1), 
        float2(1,1),
        float2(0,0),
        float2(1,0),
    };
	
	const float Radius=0.01f;
	
    //float g_fParticleRadius = 0.10;
    //float g_fParticleDrawRadius = 0.025;
	
};


const float v	: VISCOSITY
<
	string 	UIName 			= "Viscosity";
> = 0.1f;

const float dt : TIMESTEP
<
	string 	UIName 			= "Simulation Time Step";
> = 1.0f;

const float K : PRESSURESCALE
<
	string 	UIName 			= "Pressure Scale";
> = 0.1f;

const float g : GRAVITY
<
	string UIName = "Gravity Force";
> = 0.05f;

const float ParticleDX : ParticleScale
<
	string UIName = "Particle Scale";
> = 1.0f;

const float CentralScale = 1.0f/2.0f;

bool IsBoundary(float3 UV)
{
	return (UV.x<=FieldStep.x || UV.x>(1.0f-FieldStep.x) || UV.y<=FieldStep.y || UV.y>(1.0f-FieldStep.y) || UV.z<=FieldStep.z || UV.z>(1.0f-FieldStep.z) );
}

//--------------------------------------------------------------------------------------

///<
struct VS_INPUT_SCREEN
{

    float4 Pos 		: POSITION;
	float3 UV 		: TEXCOORD;		
    
};

///<
struct VS_OUTPUT_SCREEN
{
    float4 Pos		: SV_POSITION;
	float3 UV		: TEXCOORD0;	
};

struct GS_OUTPUT_SCREEN
{
    float4 Pos               : SV_Position; 
    float3 UV             	 : TEXCOORD0;  
	
    uint RTIndex             : SV_RenderTargetArrayIndex;  // used to choose the destination slice
};

//--------------------------------------------------------------------------------------
// Vertex Shader.
//--------------------------------------------------------------------------------------
VS_OUTPUT_SCREEN VS_Screen(VS_INPUT_SCREEN _Input)
{
	VS_OUTPUT_SCREEN  Out;
	Out.Pos		= float4(_Input.Pos.xyz,1);
	Out.UV		= _Input.UV;
		
	return Out;
}

//--------------------------------------------------------------------------------------
// Geometry Shader.
//--------------------------------------------------------------------------------------


[maxvertexcount (3)]
void GS_ARRAY(triangle VS_OUTPUT_SCREEN In[3], inout TriangleStream<GS_OUTPUT_SCREEN> triStream)
{
    GS_OUTPUT_SCREEN Out;
    
	// UV.z of the first vertex in the triangle determines the destination slice index.
    Out.RTIndex = In[0].UV.z/FieldStep;
    
    for(int v=0; v<3; v++)
    {
        Out.Pos     = In[v].Pos;
		Out.UV      = In[v].UV;

        triStream.Append(Out);
    }
    triStream.RestartStrip();
}

//--------------------------------------------------------------------------------------
// Pixel Shader.
//--------------------------------------------------------------------------------------

float4 PS_DrawSlice(VS_OUTPUT_SCREEN _PSInput, uniform uint _index) : SV_Target
{		
	float fIndex=_index*FieldStep.z;
	
	return FieldT.Sample(LinearSampler,float3(_PSInput.UV.xy,fIndex))+float4(0.5,0.5f,0.5f,1);
}

bool IsBlower(float3 UV)
{
	return (UV.x<BlowerPosition.x+FieldStep.x && UV.x>(BlowerPosition.x-FieldStep.x) && UV.y<=BlowerPosition.y+FieldStep.y && UV.y>(BlowerPosition.y-FieldStep.y) && UV.z<=BlowerPosition.z+FieldStep.z && UV.z>(BlowerPosition.z-FieldStep.z) );
}

float3 W(float3 UV, 
float4 UdX,
float4 UdY,
float4 UdZ)
{
	const float4 U	= FieldT.Sample(PointSampler,UV);

	float w1=UdY.x*U.z-UdZ.x*U.y;
	float w2=UdZ.y*U.x-UdY.y*U.z;
	float w3=UdX.z*U.y-UdY.z*U.x;
	
	return float3(w1,w2,w3);
}


///<
float4 PS_UpdateField(GS_OUTPUT_SCREEN PSInput) : SV_Target
{
	///< Get Texel Coordinate
	const float3 UV 			= PSInput.UV;
	
	float4 FieldCentre			= FieldT.Sample(PointSampler,UV);

	if (IsBlower(UV))
	{
		FieldCentre.xyz = BlowerVelocity;
	}	
	
	const float4 FieldRight		= FieldT.Sample(PointSampler,UV+float3(FieldStep.x,0,0));
	const float4 FieldLeft		= FieldT.Sample(PointSampler,UV-float3(FieldStep.x,0,0));
	
	const float4 FieldTop		= FieldT.Sample(PointSampler,UV+float3(0,FieldStep.y,0));
	const float4 FieldDown		= FieldT.Sample(PointSampler,UV-float3(0,FieldStep.y,0));
	
	const float4 FieldRoof		= FieldT.Sample(PointSampler,UV+float3(0,0,FieldStep.z));
	const float4 FieldBase		= FieldT.Sample(PointSampler,UV-float3(0,0,FieldStep.z));	
		
	float3x4 FieldMatPositives	= {FieldRight,FieldTop,FieldRoof};	
	float3x4 FieldMatNegatives	= {FieldLeft,FieldDown,FieldBase};	
	
	float3 Laplacian 			= mul(float3(1,1,1),(float3x3)FieldMatPositives) + mul(float3(1,1,1),(float3x3)FieldMatNegatives) - 6.0f*FieldCentre.xyz;
	
	///< du/dx, dv/dy, dw/dz
	float4 UdX=float4(FieldRight-FieldLeft)*CentralScale;
	float4 UdY=float4(FieldTop-FieldDown)*CentralScale;
	float4 UdZ=float4(FieldRoof-FieldBase)*CentralScale;
	

	float3 	DdX 		= float3(UdX.w, UdY.w, UdZ.w);	
	float 	Udiv		= UdX.x+UdY.y+UdZ.z;
	float4 	Temp 		= float4(DdX ,Udiv);
	
	FieldCentre.w 		= clamp(FieldCentre.w - dt*dot(Temp,FieldCentre),0.3f,1.7f);
	
	///< Semi-lagrangian advection.	
	//float3 Was 			= UV - dt*FieldStep*FieldCentre.xyz;		
	//FieldCentre.xyz 	= FieldT.Sample(LinearSampler,Was).xyz;	
	
	float3 Transport = -float3(dot(FieldCentre.xyz,UdX.xyz),dot(FieldCentre.xyz,UdY.xyz),dot(FieldCentre.xyz,UdZ.xyz));
	
	///< Rest of momentum conservation.
	float3 ViscosityForce 	= v*Laplacian;	
	float3 PdX 				= (K/(dt))*DdX;		
	
	//float3 w=W(UV,UdX,UdY,UdZ);
	//float3 Curl = W()
	
	float3 ExternalForces 	= 0;//float3(0,-0.001f,0.0f);//float3(0.012f,0.0f,0.0f);
	
	FieldCentre.xyz 	+= dt*(Transport + ViscosityForce - PdX + ExternalForces);	
	
	for (int i=0; i<6;++i)
	{
		if (IsBoundary(UV+(FieldStep*Directions[i]))) 
		{
			float3 SetToZero=(1-abs(Directions[i]));
			FieldCentre.xyz*=SetToZero;				
		}
	}
	
	if (IsBoundary(UV))
	{
		FieldCentre.xyz=0;
		FieldCentre.w=1;
	}

	return FieldCentre;
	
}

///<
RasterizerState PostProcess
{
	FillMode				= Solid;
	CullMode				=  Back;
	DepthClipEnable			= false;
	DepthBias				= false;	
	DepthBiasClamp			= 0;
	
	MultisampleEnable		= false;
	AntialiasedLineEnable	= false;    
};

DepthStencilState DepthDisable
{
	DepthEnable 	= FALSE;
	DepthWriteMask 	= ZERO;
	StencilEnable 	= FALSE;
};

DepthStencilState DepthEnable
{
	DepthEnable 	= TRUE;
	StencilEnable 	= FALSE;

	DepthWriteMask 	= ZERO;
	DepthFunc 		= LESS;
};

BlendState DirectAlpha
{
	AlphaToCoverageEnable = FALSE;
	
    BlendEnable[0] 	= TRUE;
	
    SrcBlend 		= SRC_ALPHA;		//SRC_COLOR;//
    DestBlend 		= INV_SRC_ALPHA;	//DEST_COLOR;//
    BlendOp 		= ADD;
	
    SrcBlendAlpha 	= ZERO;
    DestBlendAlpha 	= ZERO;
    BlendOpAlpha 	= ADD;
    RenderTargetWriteMask[0] = 1 | 2 | 4 | 8;
};

BlendState AdditiveBlending
{
    AlphaToCoverageEnable = FALSE;
	
    BlendEnable[0] 	= TRUE;
	
    SrcBlend 		= ONE;		//SRC_ALPHA; //SRC_COLOR;//
    DestBlend 		= ONE;		//INV_SRC_ALPHA;//DEST_COLOR;//
    BlendOp 		= ADD;
	
    SrcBlendAlpha 	= ZERO;
    DestBlendAlpha 	= ZERO;
    BlendOpAlpha 	= ADD;
    RenderTargetWriteMask[0] = 1 | 2 | 4 | 8;
};

BlendState DisableBlend
{
	 	AlphaToCoverageEnable 	= FALSE;
		BlendEnable[0] 			= FALSE;
};


///<
///<
///< Update and draw Particles.
///<
///<

struct VSParticleIn
{
    float4 Pos              : POSITION;         //position of the particle
};

struct VSParticleDrawOut
{
    float4 Pos 		: POSITION;
};

struct PSSceneIn
{
    float4 Pos 		: SV_Position;
    float2 UV		: TEXTURE0;
	float3 Normal : NORMAL;
};

float3 PosToUV(float3 _Pos)
{
	return (_Pos+1)/2.0f;
}

VSParticleDrawOut VS_AdvanceParticles(VSParticleIn VSInput)
{
	VSParticleDrawOut o =  VSInput;
	
	float4 Vel=FieldT.SampleLevel(LinearSampler,PosToUV(VSInput.Pos.xyz),0).xyzw;
	
	o.Pos	= float4(VSInput.Pos.xyz + ParticleDX*dt*Vel.xyz,1);
	
	return o;
}

VSParticleDrawOut VS_DrawParticles(VSParticleIn VSInput)
{
	VSParticleDrawOut o =  VSInput;
	
	o.Pos	=	VSInput.Pos;
	
	return o;
}
//
// GS for rendering point sprite particles.  Takes a point and turns it into 2 tris.
//
[maxvertexcount(4)]
void GS_PointSprite(point VSParticleDrawOut input[1], inout TriangleStream<PSSceneIn> SpriteStream)
{
    PSSceneIn Vertice;
    
    for(int i=0; i<4; i++)
    {
        float3 Pos 		= mul(Positions[i],(float3x3)ViewInverse)*Radius + input[0].Pos;		
        Vertice.Pos 	= mul(mul(float4(Pos,1.0), View),Projection);        
		Vertice.UV		= TexCoords[i];
		Vertice.Normal 	= mul(mul(float4(Normals[i],1.0), View),Projection);
		
        SpriteStream.Append(Vertice);
    }
    SpriteStream.RestartStrip();
}

//
// PS for particles
//
float3 LightView=float4(0.0,-1.0,0.0,1);
float Squared(float _x){return _x*_x;}
float4 PS_PointSprite(PSSceneIn PSInput) : SV_Target
{   
	///< 0, 1
	float Range = clamp((PSInput.Pos.z-1.5f)/(-2.5f),0,1);

	float4 C = HaloT.Sample(PointSampler,float2(Range,PSInput.UV.y))*Range;
	C.xyz	*= 2.0f;
	C.w		*= (Squared(1-Range));
	return C;
 
}

//----------------------------------------------------------------------------

VertexShader vs_AdvanceParticles = CompileShader(vs_4_0, VS_AdvanceParticles());
GeometryShader gs_StreamOut = ConstructGSWithSO(vs_AdvanceParticles, "POSITION.xyzw;");

technique10 Particle
{
	pass p0
    {
        SetVertexShader(vs_AdvanceParticles);
        SetGeometryShader(gs_StreamOut);
        SetPixelShader(NULL);
		
		SetRasterizerState(PostProcess);       
		SetDepthStencilState(DepthDisable, 0);
		SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
	
	pass p1
    {
        SetVertexShader(CompileShader(vs_4_0, VS_DrawParticles()));
        SetGeometryShader(CompileShader( gs_4_0,  GS_PointSprite()));
        SetPixelShader(CompileShader(ps_4_0, PS_PointSprite()));
		
		SetRasterizerState(PostProcess);       
		SetDepthStencilState(DepthDisable, 0);
		///AdditiveBlending
		SetBlendState(DirectAlpha, float4(0, 0, 0, 0), 0xFFFFFFFF);//1 | 2 | 4 | 8);
    }
}

//--------------------------------------------------------------------------------------

technique10 TDSolver
{

    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, VS_Screen()));
        SetGeometryShader( CompileShader( gs_4_0,  GS_ARRAY() ));
        SetPixelShader(CompileShader(ps_4_0, PS_UpdateField()));
		
		SetRasterizerState(PostProcess);       
		SetDepthStencilState(DepthDisable, 0);
		SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
    
    pass p1
    {
        SetVertexShader(CompileShader(vs_4_0, VS_Screen()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, PS_DrawSlice(15)));
		
		SetRasterizerState(PostProcess);       
		SetDepthStencilState(DepthDisable, 0);
		SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }

}

