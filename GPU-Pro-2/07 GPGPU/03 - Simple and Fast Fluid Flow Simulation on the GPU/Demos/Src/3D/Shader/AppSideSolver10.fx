
//--------------------------------------------------------------------------------------
// Constant Buffer Variables.
//--------------------------------------------------------------------------------------

///< Never changes for now.
cbuffer cbNeverChanges
{
	matrix World 			: WORLD;
	matrix View 			: VIEW;
	matrix Projection 		: PROJECTION;
	
	float3 BlowerPosition : BLOWER_POSITION
<
	string UIName="Blower position";
> = float3(0.0f, 0.0f,0);

float3 BlowerVelocity : BLOWER_VELOCITY
<
	string UIName="Blower velocity";
> = float3(0.0f, 0.0f,0);


}

//--------------------------------------------------------------------------------------
// Textures and Samplers.
//--------------------------------------------------------------------------------------

Texture2D FieldT : FIELD;

Texture2D SmokeDensityT : DENSITY;

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

float2 FieldStep 	: FIELDSIZE < > = 1.0f/64.0f;
float2 DensityStep 	: DENSITYSIZE < > = 1.0f/256.0f;

///<
const float2 Directions[4] : DIRECTIONS
<
	string UIWidget="None";
>
=
{
	float2(1,0),
	float2(0,1),
	float2(-1,0),
	float2(0,-1)	
};

const float v	: VISCOSITY
<
	string 	UIName 			= "Viscosity";
> = 0.03f;

const float dt : TIMESTEP
<
	string 	UIName 			= "Simulation Time Step";
> = 0.03f;

const float K : PRESSURESCALE
<
	string 	UIName 			= "Pressure Scale";
> = 0.15f;

const float vdt	: DENSITYSTEP
<
	string 	UIName 			= "Density time Step";	
> = 0.03f;

const float vv : DENSITYVISCOSITY
<
	string 	UIName 			= "Density Viscosity";
>	= 0.04f;

const float g : GRAVITY
<
	string UIName = "Gravity Force";
> = -0.03f;

const float dIn : SMOKESOURCE
<
	string 	UIName 			= "Smoke Source";
>	= 0.02f;

const float dOut : SMOKECOUNTER
<
	string 	UIName 			= "Smoke Counter";
>	= 0.004f;


bool IsBlower(float2 UV)
{
	float2 StepRadius = FieldStep*5.0f;	
	return (UV.x<BlowerPosition.x+StepRadius.x && UV.x>(BlowerPosition.x-StepRadius.x) && UV.y<=BlowerPosition.y+StepRadius.y && UV.y>(BlowerPosition.y-StepRadius.y) );
}

const float CentralScale = 1.0f/2.0f;

bool IsBoundary(float2 UV)
{
	return (UV.x<=FieldStep.x || UV.x>(1.0f-FieldStep.x) || UV.y<=FieldStep.y || UV.y>(1.0f-FieldStep.y));
}

//--------------------------------------------------------------------------------------

///<
struct VS_INPUT
{
	float3 UV 		: TEXCOORD;
    float4 Pos 		: POSITION;   
};

///<
struct PS_INPUT
{
	float3 UV		: TEXCOORD0;
    float4 Pos		: SV_POSITION;   
};

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
PS_INPUT VS(VS_INPUT _Input)
{
	PS_INPUT Out;
	Out.Pos		= _Input.Pos;
	Out.UV		= _Input.UV;
		
	return Out;
}

//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------

///<
float4 PS_UpdateField(PS_INPUT PSInput) : SV_Target
{
	///< Get Texel Coordinate
	const float2 UV 			= PSInput.UV.xy;
	
	float4 FieldCentre			= FieldT.Sample(PointSampler,UV);
						
	const  float3 FieldRight 	= FieldT.Sample(PointSampler,UV+float2(FieldStep.x,0));
	const  float3 FieldLeft	 	= FieldT.Sample(PointSampler,UV-float2(FieldStep.x,0));
	
	const  float3 FieldTop		= FieldT.Sample(PointSampler,UV+float2(0,FieldStep.y));
	const  float3 FieldDown		= FieldT.Sample(PointSampler,UV-float2(0,FieldStep.y));
	
	float4x3 FieldMat={FieldRight,FieldLeft,FieldTop,FieldDown};
	
	float2 Laplacian = mul(float4(1,1,1,1),(float4x2)FieldMat)-4.0f*FieldCentre.xy;
		
	//du/dx,du/dy
	float3 UdX=float3(FieldMat[0]-FieldMat[1])*CentralScale;
	float3 UdY=float3(FieldMat[2]-FieldMat[3])*CentralScale;
	
	float2 Viscosity 		= v*Laplacian;
			
	float2 ExternalForces 	= 0;		
	if (SmokeDensityT.Sample(PointSampler,UV).x > 0.4f)
	{
		ExternalForces=float2(0.0f,-g);
	}
	
	float2 DdX 					= float2(UdX.z, UdY.z);	
	float2 PdX 					= (K/dt)*DdX;
	
	float3 Temp 				= float3(DdX,UdX.x+UdY.y);
	FieldCentre.z 				= clamp(FieldCentre.z - dt*dot(Temp,FieldCentre.xyz),0.3f,1.7f);	
	
	///< Semi-Langrangian.
	float2 Was 		= UV - dt*FieldCentre.xy*FieldStep;		
	FieldCentre.xy 	= FieldT.Sample(LinearSampler,Was).xy;	
	
	FieldCentre.xy 	+= dt*(Viscosity - PdX + ExternalForces);	
			
	for (int i=0; i<4;++i)
	{
		if (IsBoundary(UV+(FieldStep*Directions[i]))) 
		{	
			float2 SetToZero=(1-abs(Directions[i]));
			FieldCentre.xy*=SetToZero;
		}
	}			
	/*
	if (IsBoundary(UV))
	{
		FieldCentre.xy=0;
		FieldCentre.z=1;
	}*/
			
	return FieldCentre;
	
}

///<
float4 PS_UpdateSmokeField(PS_INPUT PSInput) : SV_Target
{
	///< Get Texel Coordinate
	const float2 UV 	= PSInput.UV.xy;
	float2 Density		= SmokeDensityT.Sample(PointSampler,UV).xy;
	
	// float DensityRight	= x;
	// float DensityLeft	= y;
	
	// float DensityTop		= z;
	// float DensityDown	= w;
	
	float4 Neighboors   = float4(SmokeDensityT.Sample(PointSampler,UV+float2(DensityStep.x,0)).x,SmokeDensityT.Sample(PointSampler,UV-float2(DensityStep.x,0)).x,
	SmokeDensityT.Sample(PointSampler,UV+float2(0,DensityStep.y)).x,SmokeDensityT.Sample(PointSampler,UV-float2(0,DensityStep.y)).x);
	
	/// Visualization		
	float4 VisualDensityGradient  = float4(-CentralScale*(Neighboors.xz-Neighboors.yw),vv.xx);		
	float4 VisualDensityLaplacian = float4(FieldT.Sample(LinearSampler,UV).xy, Neighboors.xz + Neighboors.yw - 2.0f*Density.xx);  
			
	//float2 Was 	= UV - 1.0f*vdt*FieldT.Sample(LinearSampler,UV).xy*FieldStep;
	//Density.x 	= SmokeDensityT.Sample(LinearSampler,Was).x;		
	
	//float3 Forces=float3(0, Density.y*dIn,-dOut);		
	
	float3 Forces=float3(dot(VisualDensityGradient,VisualDensityLaplacian), Density.y*dIn,-dOut);
			
	Density.x = clamp(Density.x + vdt*dot(Forces,float3(1,1,1)),0.0f,2.0f);
	/*
	if (IsBoundary(UV))
	{
		Density.x=0;
	}*/
	
	return float4(Density,0,1);
	
}

///<
float4 PS_Draw(PS_INPUT PSInput) : SV_Target
{
    float c=SmokeDensityT.Sample(LinearSampler,PSInput.UV.xy);
	if(!IsBoundary(PSInput.UV.xy))
	{		
		return 1.0f-float4(c.x*0.6f,c.x*1.0f,c.x*2.2f,0);		
	}
	else
	{
		return float4(0.6f,0.5f,1.0f,1);
	}
}

///<
RasterizerState PostProcess
{
	FillMode				= Solid;
	CullMode				= Back;
	DepthClipEnable			= false;
	DepthBias				= false;	
	DepthBiasClamp			= 0;
	
	MultisampleEnable		= false;
	AntialiasedLineEnable	= false;    
};

DepthStencilState DepthDisable
{
	DepthEnable = FALSE;
};

BlendState DisableBlend
{
	BlendEnable[0] = FALSE;
};

//--------------------------------------------------------------------------------------

technique10 Solver
{

    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, PS_UpdateField()));
		
		SetRasterizerState(PostProcess);       
		SetDepthStencilState(DepthDisable, 0);
		SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
    
    pass p1
    {
        SetVertexShader(CompileShader(vs_4_0, VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, PS_UpdateSmokeField()));
        
        		SetRasterizerState(PostProcess);       
		SetDepthStencilState(DepthDisable, 0);
		SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
        		
    }


    pass p2
    {
        SetVertexShader(CompileShader(vs_4_0, VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, PS_Draw()));    
        
        		SetRasterizerState(PostProcess);       
		SetDepthStencilState(DepthDisable, 0);
		SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);

  
	} 
}

