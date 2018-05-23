
Texture2D		txUp	: register( t0 );
Texture2D		txD_Up	: register( t1 );

Texture2D		txBoundaries	: register( t2 );

Texture3D		txDensity	: register( t3 );

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
	
	float fNumPixels=3;
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

float4 Splatting_PS(GS_SPLATTING_DATA _input) : SV_Target
{
	float r = length(_input.Radius);
	r=min(r,1);
	float d=(exp(1.0f-r)-1.0f)/(exp(1.0f)-1.0f);
	
	return float4(_input.Velocity*d, 0, d);    
}


///<
bool IsBoundary(float2 _UV)
{
	return  (_UV.x < GridSpacing.x || _UV.x>(1.0f-GridSpacing.x) || _UV.y<GridSpacing.y || _UV.y>(1.0f-GridSpacing.y) );
}

///<
float2 GetSplattedVelocity(float2 _UV, SamplerState _sampler)
{
	float4 samp=txUp.SampleLevel(_sampler,_UV,0);
	return samp.xy/samp.w;
}

///<
float GetDensity(float2 _UV)
{
	float h = txUp.Sample(samPoint,_UV).w;

	return h;
}

float2 GetDensityGradient(float2 _UV)
{
	float dR=GetDensity( _UV+float2(GridSpacing.x,0) );
	float dL=GetDensity( _UV-float2(GridSpacing.x,0) );	
	
	float dT=GetDensity( _UV+float2(0,GridSpacing.y) );
	float dD=GetDensity( _UV-float2(0,GridSpacing.y) );
	
	return float2(dR-dL,dT-dD);
}

///<
float2 GetHeightGradient(float2 _UV)
{
	float dR=txBoundaries.Sample(samLinear, _UV+float2(GridSpacing.x,0)).x;
	float dL=txBoundaries.Sample(samLinear, _UV-float2(GridSpacing.x,0)).x;
	
	float dT=txBoundaries.Sample(samLinear, _UV+float2(0,GridSpacing.y)).x;
	float dD=txBoundaries.Sample(samLinear, _UV-float2(0,GridSpacing.y)).x;
	
	return 50.0f*float2(dR-dL,dT-dD);	
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

	float2	dP		= GetDensityGradient(UV);
	float2	u		= GetSplattedVelocity(UV, samPoint);
	float	p		= GetDensity(UV);

	float2 deltaP=-Gravity.xy;
									
	float2 du = u + dt*(-PressureScale*0.02f*(dP + GetHeightGradient(UV).xy ) + 0.03f*deltaP -0.1f*u);
	
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
	
	particle.Data	= _Input.Data;
	
	float2 posX		= _Input.Pos.xy;	
	
	float2 u			= GetSplattedVelocity(posX, samLinear);		
	float2 uCorrected	= txD_Up.SampleLevel(samLinear, posX, 0).xy;
	
	particle.Data.xy	= PIC_FLIP*uCorrected + (1.0-PIC_FLIP)*(particle.Data.xy+(uCorrected-u)); 
	
	posX += particle.Data.xy*dt*0.5f;			
	posX = clamp(posX, 1.7f*GridSpacing.xy, 1.0f-1.7f*GridSpacing.xy);
	uCorrected	= txD_Up.SampleLevel(samLinear, posX, 0).xy;
	posX += uCorrected*dt*0.5f;			
	posX = clamp(posX, 1.7f*GridSpacing.xy, 1.0f-1.7f*GridSpacing.xy);

	particle.Pos	= float4(posX,0,1);	
	
	return particle;

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

[maxvertexcount(4)]
void GS_RenderParticles(point PARTICLE_DATA input[1], inout TriangleStream<PS_RENDERPARTICLE_INPUT> _SpriteStream)
{
    PARTICLE_DATA Particle;
    
    const float Radius = 0.15f;

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
    	
        Particle.Pos = input[0].Pos + Radius*float4(mul( Positions[i].xyz, (float3x3)ViewInv),0);
        
        Particle.Pos		= mul( Particle.Pos, View );
        Particle.Pos 		= mul( Proj, Particle.Pos );
        
        Particle.Data = input[0].Data;
		
        _SpriteStream.Append(Particle);
    }
    _SpriteStream.RestartStrip();
}

///< Pixel Shader
float4 PS_RenderParticles( PS_RENDERPARTICLE_INPUT _input) : SV_Target
{
	return _input.Data.w*float4(0.5f+_input.Data.xyz,1);    
}
