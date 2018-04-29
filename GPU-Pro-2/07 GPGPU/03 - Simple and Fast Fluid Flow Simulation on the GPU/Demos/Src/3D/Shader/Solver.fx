


float Script : STANDARDSGLOBAL
<
    string UIWidget 		= "none";
    string ScriptClass 		= "scene";
    string ScriptOrder 		= "postprocess";
    string ScriptOutput 	= "color";
    string Script 			= "Technique=DefaultMPasses;";
> = 1.0f;

float2 ViewportSize 	: VIEWPORTPIXELSIZE 
<
    string UIName		= "Screen Size";
    string UIWidget		= "None";
> = float2(1,1);


float4 ClearColor <
    string UIWidget = "color";
    string UIName 	= "Clear Color";
> = {0,0,0,1.0};

float ClearDepth < string UIWidget = "None"; > = 1.0f;


///< Shared.
///< Transformations.
float4x4 Projection		: 	PROJECTION;

float2 Step = 1.0f/128.0f;


///< Quad Vertex and Pixel shaders.
struct Quad
{
    float4 Pos		: POSITION;
    float2 UV		: TEXCOORD0;
};

Quad ScreenQuadVS(Quad VSInput, uniform float2 Offset) 
{		
	VSInput.UV = VSInput.UV+0.5f*Offset;
    return VSInput;
}

texture2D FieldT : RENDERCOLORTARGET
<
	string 	UIName 			= "Field Texture";
    string 	UIWidget 		= "Texture";
	int 	MipLevels 		= 1;
	string 	Format 			= "A16B16G16R16F";
>;

sampler2D FieldSampler = sampler_state 
{
    Texture 	= <FieldT>;
	
	SRGBTexture = False;
	
    AddressU  	= Clamp;
    AddressV 	= Clamp;
	
	MipFilter 	= None;
    MinFilter 	= Point;
    MagFilter 	= Point;   
	
};

sampler2D FieldLinearSampler = sampler_state 
{
    Texture 	= <FieldT>;
	
	SRGBTexture = False;
	
    AddressU  	= Clamp;
    AddressV 	= Clamp;
	
	MipFilter 	= None;
    MinFilter 	= Linear;
    MagFilter 	= Linear;   
	
};

texture InitialDensity : INITIALDENSITY 
<
    string UIName 		= "Initial Density";
	
    string ResourceType = "2D";
>;

sampler2D InitialDensitySampler = sampler_state
{
    Texture = <InitialDensity>;
	
    SRGBTexture = True;
	
    AddressU  	= Clamp;
    AddressV 	= Clamp;
	
	MipFilter 	= None;
    MinFilter 	= Point;
    MagFilter 	= Point; 	
	
};  

texture InitialDomain : INITIALDOMAIN 
<
    string UIName 		= "Initial Domain";
    string ResourceType = "2D";
>;

sampler2D InitialDomainSampler = sampler_state
{
    Texture 		= <InitialDomain>;
	
    SRGBTexture		= True;
	
    AddressU  	= Clamp;
    AddressV 	= Clamp;		

	MipFilter 	= None;
    MinFilter 	= Point;
    MagFilter 	= Point; 
	
};  

texture2D DensityT : RENDERCOLORTARGET
<
	string 	UIName 			= "Density Texture!";
   	string 	UIWidget 		= "Texture";	
	int 	MipLevels 		= 1;
	string 	Format 			= "A32B32G32R32F";
>;

sampler2D DensitySampler = sampler_state 
{
    Texture 	= <DensityT>;
	
	SRGBTexture = False;
	
    AddressU  	= Clamp;
    AddressV 	= Clamp;
	
	MipFilter 	= None;
    MinFilter 	= Point;
    MagFilter 	= Point;   
	
};

sampler2D DensityLinearSampler = sampler_state 
{
    Texture 	= <DensityT>;
	
	SRGBTexture = False;
	
    AddressU  	= Clamp;
    AddressV 	= Clamp;
	
	MipFilter 	= None;
    MinFilter 	= Linear;
    MagFilter 	= Linear; 
		   
};

///
bool BInitializeDomain : INIT 
<
  	string 	UIName 			= "Initialize Domain";
> = true;


///<
float2 Directions[4] : DIRECTIONS
<
	string UIWidget="None";
>
=
{
	float2(1,0),
	float2(0,-1),
	float2(-1,0),
	float2(0,1)	
};


float v	
<
	string 	UIName 			= "Viscosity";
		
	string UIWidget = "slider";
    float UIMin = 0.03;
    float UIMax = 0.2;
    float UIStep = 0.01;
	
> = 0.05f;


float Grav	
<
	string 	UIName 			= "Gravity";
		
	string UIWidget = "slider";
    float UIMin = 0.0;
    float UIMax = 0.2;
    float UIStep = 0.01;
	
> = 0.07f;

float dt
<
	string 	UIName 			= "Delta t: Simulation Time Step";
		
	string UIWidget = "slider";
    float UIMin = 0.0;
    float UIMax = 0.3;
    float UIStep = 0.01;
	
> = 0.1f;

float K
<
	string 	UIName 			= "K";
	string UIWidget = "slider";
    float UIMin = 0.2;
    float UIMax = 0.3;
    float UIStep = 0.01;
> = 1.0f;

float Squared(float _x){return _x*_x;}

bool IsBoundary(float2 UV)
{
	return (UV.x<=Step.x || UV.x>(1.0f-Step.x) || UV.y<=Step.y || UV.y>(1.0f-Step.y));
}

///< Solver Pixel Shader.
float4 SolverPS(Quad PSInput) : COLOR0 
{	
	float CScale=1.0f/(2.0f);
	
	const float2 	UV 	= PSInput.UV;
	float4 			FC	= tex2D(FieldSampler,UV);
	
	bool isBoundary=tex2D(DensitySampler,UV).w<0.1f;
	
	if (BInitializeDomain)
	{
		FC=float4(0,0,1,0);
	}
	else
	{		
		if (!isBoundary)
		{			
			const float3 FR	= tex2D(FieldSampler,UV+float2(Step.x,0));
			const float3 FL	= tex2D(FieldSampler,UV-float2(Step.x,0));			
			const float3 FT	= tex2D(FieldSampler,UV+float2(0,Step.y));
			const float3 FD	= tex2D(FieldSampler,UV-float2(0,Step.y));
			
			float4x3 FieldMat={FR,FL,FT,FD};
	
			//du/dx,du/dy
			float3 	UdX			= float3(FieldMat[0]-FieldMat[1])*CScale;
			float3 	UdY			= float3(FieldMat[2]-FieldMat[3])*CScale;
			
			float 	Udiv		= UdX.x+UdY.y;
			float2 	DdX			= float2(UdX.z,UdY.z);			
			
			///<
			///< Solve for Velocity. 	
			///<
			float2  Laplacian 		= mul((float4)1,(float4x2)FieldMat)-4.0f*FC.xy;
			Laplacian/=1.0f;
						
			float2 	PdX 			= (K/dt)*DdX;		
			float2 	ViscosityForce 	= v*Laplacian/FC.z;
			
			///<
			///< Solve for density.
			///<
			FC.z 	-=  dt*dot(float3(DdX,Udiv),FC.xyz);
			///< See Stability Condition.
  			FC.z 	= clamp(FC.z,0.2f,3.0f);
					
			float2 	ExternalForces 	= 	0;
			if (tex2D(DensitySampler,UV).x > 0.1f)
				ExternalForces=float2(0.0f,Grav);			
				
			///< Semi-lagrangian advection.
			float2 Was 	= UV - dt*FC.xy*Step;
			FC.xy 	= tex2D(FieldLinearSampler,Was).xy;	
				
			FC.xy 	+= dt*(ViscosityForce - PdX + ExternalForces);	
			
			///< Boundary conditions.
			for (int i=0; i<4; ++i)
			{
				if (tex2D(DensitySampler,UV+Step*Directions[i]).w < 0.1f) 
				{
					float2 SetToZero=(1-abs(Directions[i]));
					FC.xy*=SetToZero;
				}
			}				
		}	
	}///If init
		
	return FC;
}

float4 UpdateDensityPS(Quad PSInput) : COLOR0
{	
	const float2 UV 	= PSInput.UV;
	float4 Density		= tex2D(DensitySampler,UV);
	
	if (!BInitializeDomain)
	{
		if (Density.w>0.1f)
		{					
			///< Semi-lagrangian advection.			
			float2 Was 	= UV - dt*tex2D(FieldLinearSampler,UV).xy*Step;
			Density.x 	= tex2D(DensityLinearSampler,Was).x;	
		}
	}
	else
	{
		Density.xyz=tex2D(InitialDensitySampler,UV).x*7.0f;
		Density.w=tex2D(InitialDomainSampler,UV).x;
	}
	
	return Density;

}

float4 ShowPS(Quad PSInput) : COLOR0
{
	float4 D=tex2D(DensityLinearSampler,PSInput.UV);	
	float c = D.x;
	return D.w*float4(c,c*2.5f,c,1) + (1.0f-D.w)*float4(0.3f,0.2f,1.0f,1);
}

/////////////////////////////////////////////////////////////////////////////////////
///
/// Techniques.
///
/// Comments
/// -At release time :
/// 	Remove the SRGBWrite and don't use the sampler 
///		state in the texture look ups.
/////////////////////////////////////////////////////////////////////////////////////


///< DX 9 
technique DefaultMPasses
< 
string Script=	"RenderColorTarget0=;"								
				"RenderDepthStencilTarget=;"
				"ClearSetColor=ClearColor;"
				"ClearSetDepth=ClearDepth;"
				"Clear=Color;"
				"Clear=Depth;"
	    		"ScriptExternal=color;"
				
				///< Solving is done in one pass.				
				"Pass=SolveVelocityField;"
				"Pass=SolveVisualisationDensityField;"
				"Pass=Show";						
> 
{
	
	
	///< Velocity Pass
	pass SolveVelocityField
	<
       	string Script= 	
						"RenderColorTarget0=FieldT;"
						"RenderDepthStencilTarget=;"
						"Draw=Buffer;";
    >
    {	
		
		
		///< Cull
		CullMode 			= None;
		
		///< Alpha
		AlphaTestEnable		= False;
		AlphaBlendEnable	= False;
		
		///< Color
		SRGBWriteenable 	= false;	
		ColorWriteEnable 	= RED|GREEN|BLUE|ALPHA;
		
		///< Z
		ZEnable 			= False;
		ZWriteEnable 		= False;
		
		VertexShader 		= compile vs_3_0 ScreenQuadVS(Step);
		PixelShader 		= compile ps_3_0 SolverPS();	
    }	
	
	///< Smoke Density Pass
	pass SolveVisualisationDensityField
	<
       	string Script= 	"RenderColorTarget0=DensityT;"
						"RenderDepthStencilTarget=;"				
						"Draw=Buffer;";
    >
	{		
		///< Cull
		CullMode 			= None;
		
		///< Alpha
		AlphaTestEnable		= False;
		AlphaBlendEnable	= False;
		
		///< Color
		SRGBWriteenable 	= False;
		ColorWriteEnable 	= RED|GREEN|BLUE|ALPHA;
		
		///< Z
		ZEnable 			= False;
		ZWriteEnable 		= False;
		
		VertexShader 		= compile vs_3_0 ScreenQuadVS(Step);
		PixelShader 		= compile ps_3_0 UpdateDensityPS();	
    }	
	
	///< Show Smoke Density
	pass Show
	<
       	string Script= 	"RenderColorTarget0=;"
						"RenderDepthStencilTarget=;"
						"ClearSetColor=ClearColor;"
						"ClearSetDepth=ClearDepth;"					
						"Draw=Buffer;";
    >
	{		
		///< Cull
		CullMode 			= None;
		
		///< Alpha
		AlphaTestEnable		= False;
		AlphaBlendEnable	= False;
		
		///< Color
		SRGBWriteenable 	= False;
		ColorWriteEnable 	= RED|GREEN|BLUE|ALPHA;
		
		///< Z
		ZEnable 			= False;
		ZWriteEnable 		= False;
		
		VertexShader 		= compile vs_3_0 ScreenQuadVS(1.0f/ViewportSize);
		PixelShader 		= compile ps_3_0 ShowPS();	
    }	
	
}
