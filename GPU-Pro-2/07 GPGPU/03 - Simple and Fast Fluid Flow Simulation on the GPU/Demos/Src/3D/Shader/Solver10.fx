
///< Fluid Flow Simulation on The GPU using DirectX 10.
///< Need to copy textures i.e., can't read and write from/to same texture with DirectX10.


float Script : STANDARDSGLOBAL
<
    string UIWidget 		= "none";
    string ScriptClass 		= "scene";
    string ScriptOrder 		= "postprocess";
    string ScriptOutput 	= "color";
    string Script 			= "Technique=DefaultMPasses10;";
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

static float2 FieldStep 		= 1.0f/128.0f;
static float2 ViewportOffset 	= float2(-0.5f,-0.5)/ViewportSize;

///< Quad Vertex and Pixel shaders.
struct Quad
{
    float4 Pos		: POSITION;
    float2 UV		: TEXCOORD0;
};

Quad ScreenQuadVS(Quad VSInput )
{		
	VSInput.UV = VSInput.UV + ViewportOffset;
    return VSInput;
}

Quad UniformQuadVS(Quad VSInput )
{		
	VSInput.UV = VSInput.UV;//-0.5f*FieldStep;
    return VSInput;
}

Texture2D Scene : RENDERCOLORTARGET 
<
    float2 	ViewPortRatio 	= {1.0,1.0};
    int 	MipLevels 		= 1;
    string 	Format 			= "X8R8G8B8";
>;

Texture2D FieldT : RENDERCOLORTARGET
<
	string 	UIName 			= "Velocity Field";
    string 	UIWidget 		= "Texture";
	int 	MipLevels 		= 0;
	string 	Format 			= "A32B32G32R32F";
>;

Texture2D FieldTCopy : RENDERCOLORTARGET
<
	string 	UIName 			= "Velocity Field Copy";
    string 	UIWidget 		= "Texture";
	int 	MipLevels 		= 0;
	string 	Format 			= "A32B32G32R32F";
>;

Texture2D DensityT : RENDERCOLORTARGET
<
	string 	UIName 			= "Smoke Density";
   	string 	UIWidget 		= "Texture";	
	int 	MipLevels 		= 0;
	string 	Format 			= "A32B32G32R32F";
>;

Texture2D DensityTCopy : RENDERCOLORTARGET
<
	string 	UIName 			= "Density Copy";
   	string 	UIWidget 		= "Texture";	
	int 	MipLevels 		= 0;
	string 	Format 			= "A32B32G32R32F";
>;

Texture2D InitialDensity : INITIALDENSITY 
<
    string UIName 			= "Initial Density";	
    string ResourceType 	= "2D";
>;

Texture2D InitialDomain : INITIALDOMAIN 
<
    string UIName 			= "Initial Domain";
    string ResourceType 	= "2D";
>;

SamplerState PointSampler  
{	
    AddressU  	= Clamp;
    AddressV 	= Clamp;
	
	Filter=MIN_MAG_MIP_POINT;
};

SamplerState LinearSampler
{
    AddressU  	= Clamp;
    AddressV 	= Clamp;
	
	Filter=MIN_MAG_LINEAR_MIP_POINT;   
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

const float CentralScale=1.0f/2.0f;
///< Solver Pixel Shader.
float4 SolverPS(Quad PSInput) : SV_Target 
{	
	const float2 	UV 		= PSInput.UV;
	float4 			FC		= FieldT.Sample(PointSampler,UV);
	
	if (BInitializeDomain)
	{
		FC=float4(0,0,1,0);
	}
	else
	{		
		///< obstacles are stored in the density texture w-component.
		if (DensityT.Sample(PointSampler,UV).w > 0.1f)
		{							
			const  float3 FR 	= FieldT.Sample(PointSampler,UV+float2(FieldStep.x,0));
			const  float3 FL	= FieldT.Sample(PointSampler,UV-float2(FieldStep.x,0));
			
			const  float3 FT	= FieldT.Sample(PointSampler,UV+float2(0,FieldStep.y));
			const  float3 FD	= FieldT.Sample(PointSampler,UV-float2(0,FieldStep.y));
	
			float4x3 FieldMat = {FR,FL,FT,FD};
				
			//du/dx,du/dy
			float3 UdX=float3(FieldMat[0]-FieldMat[1])*CentralScale;
			float3 UdY=float3(FieldMat[2]-FieldMat[3])*CentralScale;
	
			float2 DdX 					= float2(UdX.z, UdY.z);	
			float3 Temp 				= float3(DdX,UdX.x+UdY.y);
			FC.z 						= clamp(FC.z - dt*dot(Temp,FC.xyz),0.2f,3.0f);		
			
			float2	PdX 				= (K/dt)*DdX;
			float2 	Laplacian 			= mul(float4(1,1,1,1),(float4x2)FieldMat)-4.0f*FC.xy;
			float2  ViscosityForce 		= v*Laplacian;			
			
			float2 Was 	= UV - dt*FC.xy*FieldStep;
			FC.xy 		= FieldT.Sample(LinearSampler,Was).xy;		
			
			float2 	ExternalForces 		= 	0;
			if (DensityTCopy.SampleLevel(PointSampler,UV,0).x > 0.1f)
				ExternalForces=float2(0.0f,0.1f);	
				
			FC.xy += dt*(ViscosityForce - PdX + ExternalForces);	
				
			
			/*
			if (DensityT.Sample(PointSampler,UV+FieldStep*Directions[0]).w < 0.1f) 
			{				
				if(abs(Directions[0].y)>0)
				{
					float2 SetToZero=(1-abs(Directions[0]));
					FC.xy*=SetToZero;
				}					
			}
			
			if (DensityTCopy.Sample(PointSampler,UV+FieldStep*Directions[1]).w < 0.1f) 
			{				
				if(abs(Directions[1].y)>0)
				{
					float2 SetToZero=(1-abs(Directions[1]));
					FC.xy*=SetToZero;
				}					
			}
			
			
			if (DensityT.Sample(PointSampler,UV+FieldStep*Directions[2]).w < 0.1f) 
			{				
				if(abs(Directions[2].y)>0)
				{
					float2 SetToZero=(1-abs(Directions[2]));
					FC.xy*=SetToZero;
				}					
			}
			
			if (DensityT.Sample(PointSampler,UV+FieldStep*Directions[3]).w < 0.1f) 
			{				
				if(abs(Directions[3].y)>0)
				{
					float2 SetToZero=(1-abs(Directions[3]));
					FC.xy*=SetToZero;
				}					
			}*/
				
				
			for (int i=0; i<4; ++i)
			{
				
				if (DensityTCopy.SampleLevel(PointSampler,UV+FieldStep*Directions[i],0).w < 0.2f) 
				{				
					if(abs(Directions[i].y)>0)
					{
						float2 SetToZero=(1-abs(Directions[i]));
						FC.xy*=SetToZero;
					}					
				}
			}
			
		}
		else
		{
			FC.xy=0;
			FC.z=1;
		}
		
	}///If init
	
	
	return FC;
}

float4 UpdateDensityPS(Quad PSInput) : SV_Target
{	
	const float2 UV 	= PSInput.UV;
	float4 Density		= DensityT.Sample(PointSampler,UV);
	
	if (!BInitializeDomain)
	{
		if (Density.w>0.1f)
		{
			///< Semi-lagrangian advection.
			float2 Was 	= UV - dt*FieldT.Sample(PointSampler,UV).xy*FieldStep;
			Density.x 	= DensityT.Sample(LinearSampler,Was).x;
				/*		
			float4 DensityRight	=DensityT.SampleLevel(PointSampler,UV+float2(FieldStep.x,0),0).x;
			float4 DensityLeft	=DensityT.SampleLevel(PointSampler,UV-float2(FieldStep.x,0),0).x;
			
			float4 DensityTop	=DensityT.SampleLevel(PointSampler,UV+float2(0,FieldStep.y),0).x;
			float4 DensityDown	=DensityT.SampleLevel(PointSampler,UV-float2(0,FieldStep.y),0).x;						
			
			/// Visualization
			float2 DdX;
			DdX.x 	= (DensityRight.x - DensityLeft.x)*CentralScale;
			DdX.y 	= (DensityTop.x	- DensityDown.x)*CentralScale;
				
			float2 DddX;
			DddX.x 	= DensityRight.x - 2.0f*Density.x + DensityLeft.x;
			DddX.y 	= DensityTop.x - 2.0f*Density.x + DensityDown.x;			

			float Forces 	= dot(-FieldT.Sample(LinearSampler,UV).xy,DdX)+v*(DddX.x+DddX.y);
			
			Density.x += dt*Forces;			
			*/
		}
	}
	else
	{
		Density.xyz	=InitialDensity.Sample(PointSampler,UV).x*7.0f;
		Density.w	=InitialDomain.Sample(PointSampler,UV).x;
	}
	
	return Density;

}

float4 ShowPS(Quad PSInput) : SV_Target
{
	float4 D=DensityT.Sample(PointSampler,PSInput.UV);
	float c = D.x;
	return D.w*float4(c,c*2.5f,c,1) + (1.0f-D.w)*float4(0.8f,0.8f,0.0f,1);
}

float4 CopyPS(Quad PSInput, uniform Texture2D _Src) : SV_Target
{
	return _Src.Sample(PointSampler,PSInput.UV);
}

/////////////////////////////////////////////////////////////////////////////////////
///
/// Techniques.
///
/// Comments:
///
/////////////////////////////////////////////////////////////////////////////////////

RasterizerState DisableCulling
{
    CullMode = None;
};

DepthStencilState DepthEnabling
{
	DepthEnable = False;
};

BlendState DisableBlend
{
	BlendEnable[0] = False;
};

technique10 DefaultMPasses10
< 
string Script=	"RenderColorTarget0=;"								
				"RenderDepthStencilTarget=;"
				"ClearSetColor=ClearColor;"
				"ClearSetDepth=ClearDepth;"
				"Clear=Color;"
				"Clear=Depth;"
	    		"ScriptExternal=color;"
							
				"Pass=SolveVelocityField;"
				"Pass=CopyVelocity;"
				"Pass=SolveVisualisationDensityField;"
				"Pass=CopyDensity;"
				"Pass=Show";
				>
				
{
	///< Velocity Pass
	pass SolveVelocityField
	<
       	string Script= 	
						"RenderColorTarget0=FieldTCopy;"
						"RenderDepthStencilTarget=;"
						"Draw=Buffer;";
    >
    {			
		SetVertexShader(CompileShader( vs_4_0, UniformQuadVS() ));
        SetGeometryShader( NULL );
        SetPixelShader(CompileShader( ps_4_0, SolverPS() ) );
		
		SetRasterizerState(DisableCulling);       
		SetDepthStencilState(DepthEnabling, 0);
		SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }	
	
	///< DX10 detects simultanous Read/Write and sets the ressource to null.
	pass CopyVelocity
	<
		string Script= 	
					"RenderColorTarget0=FieldT;"
					"RenderDepthStencilTarget=;"
					"Draw=Buffer;";
	>
	{
		SetVertexShader(CompileShader( vs_4_0, UniformQuadVS() ));
        SetGeometryShader( NULL );
        SetPixelShader(CompileShader( ps_4_0, CopyPS(FieldTCopy) ) );
		
		SetRasterizerState(DisableCulling);       
		SetDepthStencilState(DepthEnabling, 0);
		SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
	}
	
	///< Smoke Density Pass
	pass SolveVisualisationDensityField
	<
       	string Script= 	"RenderColorTarget0=DensityTCopy;"
						"RenderDepthStencilTarget=;"				
						"Draw=Buffer;";
    >
	{		

		
		SetVertexShader(CompileShader( vs_4_0, UniformQuadVS() ));
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, UpdateDensityPS() ) );
		
		SetRasterizerState(DisableCulling);       
		SetDepthStencilState(DepthEnabling, 0);
		SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
		
    }	
	
	///< DX10 detects simultanous Read/Write and sets the ressource to null.
	pass CopyDensity
	<
		string Script= 	
					"RenderColorTarget0=DensityT;"
					"RenderDepthStencilTarget=;"
					"Draw=Buffer;";
	>
	{
		SetVertexShader(CompileShader( vs_4_0, UniformQuadVS() ));
        SetGeometryShader( NULL );
        SetPixelShader(CompileShader( ps_4_0, CopyPS(DensityTCopy) ) );
		
		SetRasterizerState(DisableCulling);       
		SetDepthStencilState(DepthEnabling, 0);
		SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
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
		SetVertexShader(CompileShader( vs_4_0, ScreenQuadVS() ));
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, ShowPS() ) );
		
		SetRasterizerState(DisableCulling);       
		SetDepthStencilState(DepthEnabling, 0);
		SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);

    }	
}

