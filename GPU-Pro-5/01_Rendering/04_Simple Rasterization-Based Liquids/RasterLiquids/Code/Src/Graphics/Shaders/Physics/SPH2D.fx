
Texture2D		txUp	: register( t0 );
Texture2D		txD_Up	: register( t1 );

Texture2D		txMask		: register( t2 );

Texture2D		txP			: register( t3 );
Texture2D		txDiv		: register( t4 );

#include "..\\CommonConstants.hlsl"
#include "SPHParams.hlsl"

///<
///< Quad Processing
///<

struct VS_INPUT
{
    float4 Pos : POSITION;
    float2 UV : TEXCOORD0;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float2 UV : TEXCOORD0;
};


PS_INPUT RenderQuad_VS(VS_INPUT _input)
{
	PS_INPUT output;
	  
	output.Pos = _input.Pos;  
    output.UV = _input.UV;
	
    return output;    
}

///< 
struct PARTICLE_DATA
{
    float4 Pos : POSITION;
    float4 Data: TEXCOORD0; 
};

struct GS_SPLATTING_DATA
{
	float4 Pos : SV_POSITION;
    float2 Radius : TEXCOORD0;
	float2 Velocity : TEXCOORD1;   
};

PARTICLE_DATA Splatting_VS(PARTICLE_DATA _Input)
{
	PARTICLE_DATA particle;	
	
	particle.Pos	= _Input.Pos;
	particle.Data	= _Input.Data;
		
	return particle;
}

float2 UVToWorld(float2 _UV)
{
	_UV.y=1.0f-_UV.y;
	return (_UV-0.5f)*2.0f;
}

///<
[maxvertexcount(4)]
void Splatting_GS(point PARTICLE_DATA _input[1], inout TriangleStream<GS_SPLATTING_DATA> _SpriteStream)
{

	const float2 Positions[4]=
	{
		float2( -1, 1 ),
		float2( 1, 1 ),
		float2( -1, -1 ),
		float2( 1, -1)
	}; 
        
    GS_SPLATTING_DATA Vertex;
	
	float3 x = _input[0].Pos.xyz;
	
	Vertex.Velocity = _input[0].Data.xy;
	
	float fNumPixels=3.0;
	float2 dr	= fNumPixels*GridSpacing.xy;
	
	for(int i=0; i<4; ++i)
	{
		Vertex.Pos = float4(x.xy + dr*Positions[i],0,1);
				
		float2 r	= Positions[i];
		Vertex.Pos.xy=UVToWorld(Vertex.Pos.xy);

		Vertex.Radius = r;		
		
		_SpriteStream.Append(Vertex);
	}
	_SpriteStream.RestartStrip();
}


///<
bool IsBoundary(float2 _UV)
{
	return  (_UV.x < GridSpacing.x || _UV.x>(1.0f-GridSpacing.x) || _UV.y<GridSpacing.y || _UV.y>(1.0f-GridSpacing.y) );
}

float4 Splatting_PS(GS_SPLATTING_DATA _input) : SV_Target
{
	float r = length(_input.Radius);
	r=min(r,1);
	
	float d=(exp(1.0f-r)-1.0f)/(exp(1.0f)-1.0f);	

	return float4(_input.Velocity*d, 0, d);    
}

///<
float GetDensity(float2 _UV)
{
	return txUp.Sample(samPoint,_UV).w;
}

///<
float GetWeaklyPressure(float2 _UV)
{
	float pr= (txUp.SampleLevel(samPoint,_UV,0).w)/InitialDensity.x; 
	
	return 0.0075f*pow(pr,4); 
}

float2 GetDensityGradient(float2 _UV)
{
	float dR=GetWeaklyPressure( _UV+float2(GridSpacing.x,0) );
	float dL=GetWeaklyPressure( _UV-float2(GridSpacing.x,0) );	
	
	float dT=GetWeaklyPressure( _UV+float2(0,GridSpacing.y) );
	float dD=GetWeaklyPressure( _UV-float2(0,GridSpacing.y) );

	return float2(dR-dL,dT-dD);
}

///<
float4 PS_AddDensityGradient(PS_INPUT _input) : SV_Target
{		
	const float2 UV = _input.UV;
	
	const float2 Directions[4]=
	{
		float2(1,0),
		float2(0,1),
		float2(-1,0),
		float2(0,-1)
	};

	float2 PressGrad = GetDensityGradient(UV);
	
	float3 samp		= txUp.Sample(samPoint,UV).xyw;
	float2	u=0;
	if (samp.z>0.01f)
		u = samp.xy/samp.z;
		
	float	p		= samp.z;
	float2  grav	= 0;
	
	if (p>SurfaceDensity)
		grav=float2(Gravity.xy);
						
	float2 du = u + dt*(-PressureScale*PressGrad + grav);

	for (int i=0; i<4;++i)
	{
		if (IsBoundary(UV+(GridSpacing.xy*Directions[i]))) 
		{	
			float2 SetToZero=(1-abs(Directions[i]));
			du.xy*=SetToZero;
		}
	}	
	
	return float4(du,0,p);
}

///< Transport:
PARTICLE_DATA VS_AdvanceParticles(PARTICLE_DATA _Input)
{
	PARTICLE_DATA particle;	
	
	particle.Data		= _Input.Data;
	
	float2 posX			= _Input.Pos.xy;	
	
	float3 uP			= txUp.SampleLevel(samLinear, posX, 0).xyw;
	particle.Data.w		= uP.z;
	
	float2 u=0;
	if (uP.z>0.01f)
		u = uP.xy/uP.z;
			
	float2 uCorrected	= txD_Up.SampleLevel(samLinear, posX, 0).xy;
	particle.Data.xy = PIC_FLIP*uCorrected + (1.0-PIC_FLIP)*(particle.Data.xy+(uCorrected-u)); 
	
	posX += particle.Data.xy*0.5f*dt;
	posX = clamp(posX, 1.7f*GridSpacing.xy , 1.0f-1.7f*GridSpacing.xy);

	uCorrected	= txD_Up.SampleLevel(samLinear, posX, 0).xy;
	posX += uCorrected*dt;
	posX = clamp(posX,  1.7f*GridSpacing.xy, 1.0f-1.7f*GridSpacing.xy);
	
	particle.Pos	= float4(posX,0,1);	
	
	return particle;

}

///<
float4 PS_ComputeDivergence(PS_INPUT _input) : SV_Target
{
	const float2 UV = _input.UV;

	float d = txUp.Sample(samPoint, UV).w;
	
	float dR = txUp.Sample(samPoint, UV+float2(GridSpacing.x,0) ).x;
	float dL = txUp.Sample(samPoint, UV-float2(GridSpacing.x,0) ).x;	
	
	float dT = txUp.Sample(samPoint, UV+float2(0,GridSpacing.y) ).y;
	float dD = txUp.Sample(samPoint, UV-float2(0,GridSpacing.y) ).y;
	
	float dx=GridSpacing.x;
	float div=0.0f;
	if (!IsBoundary(UV) && d>10.0f)
		div = (dR-dL + dT-dD)*0.5f/dx;
	
	return div;
}

float GetPressure(float2 _UV)
{
	float d = txUp.Sample(samPoint,_UV).w;
	return txP.Sample(samPoint,_UV).x;
}

///< Compute Jacobi Iterations
float4 PS_Jacobi(PS_INPUT _input) : SV_Target
{
	float dx=GridSpacing.x;
	const float c = -4.0f/(dx*dx);
	const float a = 1.0f/(dx*dx);
	
	const float2 UV = _input.UV;

	float PC=GetPressure(UV);
	
	float PR=GetPressure(UV+float2(GridSpacing.x,0));
	float PL=GetPressure(UV-float2(GridSpacing.x,0));	
	
	float PT=GetPressure(UV+float2(0,GridSpacing.y));
	float PD=GetPressure(UV-float2(0,GridSpacing.y));
	
	if (IsBoundary(UV+float2(GridSpacing.x,0)))
		PR=PC;
	if (IsBoundary(UV-float2(GridSpacing.x,0)))
		PL=PC;		
	if (IsBoundary(UV+float2(0,GridSpacing.y)))
		PT=PC;
	if (IsBoundary(UV-float2(0,GridSpacing.y)))
		PD=PC;

	float Div = txDiv.Sample(samPoint,UV).x;
	float Pressure = (Div-a*(PD+PT+PL+PR))/c;
	
	return Pressure;
}

///<
float4 PS_AddPressureGradient(PS_INPUT _input) : SV_Target
{
	const float2 UV 			= _input.UV;
	
	float4 samp = txUp.Sample(samPoint,UV).xyzw;
	float2 u = samp.xy;
	
	if (!IsBoundary(UV))
	{
		float PC		= GetPressure(UV);
	
		float PR 		= GetPressure(UV+float2(GridSpacing.x,0));
		float PL	 	= GetPressure(UV-float2(GridSpacing.x,0));
		
		float PT		= GetPressure(UV+float2(0,GridSpacing.y));
		float PD		= GetPressure(UV-float2(0,GridSpacing.y));
			
		if (IsBoundary(UV+float2(GridSpacing.x,0)))
			PR=PC;
		if (IsBoundary(UV-float2(GridSpacing.x,0)))
			PL=PC;	
		if (IsBoundary(UV+float2(0,GridSpacing.y)))
			PT=PC;
		if (IsBoundary(UV-float2(0,GridSpacing.y)))
			PD=PC;
			
		float2 GradP = float2(PR-PL,PT-PD);
		
		float dx=GridSpacing.x;
		u = u-0.5f*GradP/dx;
	}
	
	return float4(u,0,samp.w);
	
}

///< Render Particles

struct PS_RENDERPARTICLE_INPUT
{
    float4 Pos : SV_POSITION;
    float4 Data: TEXCOORD;
};

///< Vertex Shader
PARTICLE_DATA VS_RenderParticles(PARTICLE_DATA _input)
{
    return _input;    
}

///<
[maxvertexcount(4)]
void GS_RenderParticles(point PARTICLE_DATA input[1], inout TriangleStream<PS_RENDERPARTICLE_INPUT> _SpriteStream)
{
    PARTICLE_DATA Particle;
    
    const float2 Radius = (ScreenDimensions.yx/ScreenDimensions.x)*0.003f;

	const float3 Positions[4]
	=
	{
		float3( -1, 1, 0 ),
		float3( 1, 1, 0 ),
		float3( -1, -1, 0 ),
		float3( 1, -1, 0 )
	};
	    
    for(int i=0; i<4; i++)
    {
    	
        Particle.Pos = input[0].Pos + float4(Radius*Positions[i].xy,0,0);
		Particle.Pos.y = 1.0-Particle.Pos.y;
        Particle.Pos.xy = Particle.Pos.xy*2.0f -1.0f;
        

        Particle.Data = input[0].Data;
		
        _SpriteStream.Append(Particle);
    }
    _SpriteStream.RestartStrip();
}

///< Pixel Shader
float4 PS_RenderParticles( PS_RENDERPARTICLE_INPUT _input) : SV_Target
{	
	float d = saturate(((_input.Data.w-8.0f))/17.0f);
	
	return txMask.Sample(samLinear,float2(d,0) );
}

///<
float2 GetSplattedVelocity(float2 _UV, SamplerState _sampler)
{
	float4 samp=txUp.SampleLevel(_sampler,_UV,0);
	return samp.xy/samp.w;
}

