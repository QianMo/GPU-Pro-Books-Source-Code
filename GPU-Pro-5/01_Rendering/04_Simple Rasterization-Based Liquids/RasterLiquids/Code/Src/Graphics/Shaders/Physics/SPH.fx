
Texture3D		txUp	: register( t0 );
Texture3D		txD_Up	: register( t1 );

///< Scalars
Texture3D		txDiv	: register( t2 );
Texture3D		txP		: register( t3 );

Texture2D		txWaterHeight : register(t4);

#include "..\\CommonConstants.hlsl"
#include "SPHParams.hlsl"


struct FRAGMENT_DATA
{
    float4 Pos : POSITION;
    float3 UV : TEXCOORD0;
};

struct GS_OUTPUT
{
    float4 Pos               : SV_POSITION; 
    float3 UV             	 : TEXCOORD0;  
	
    uint SliceIndex          : SV_RenderTargetArrayIndex;  ///< used to choose the destination slice
};

///< Vertex Shader
FRAGMENT_DATA VS_VolumeSim(FRAGMENT_DATA _input)
{
	FRAGMENT_DATA output;
		
	output.Pos = float4(_input.Pos.xyz,1); 
    output.UV = _input.UV;  
	
    return output;    
}

///< Geometry Slices
[maxvertexcount (3)]
void GS_ARRAY(triangle FRAGMENT_DATA _input[3], inout TriangleStream<GS_OUTPUT> _triStream)
{
    GS_OUTPUT Out;
    
    Out.SliceIndex = _input[0].UV.z/GridSpacing.z;
    
    for(int v=0; v<3; v++)
    {
        Out.Pos     = _input[v].Pos;
		Out.UV      = _input[v].UV;

        _triStream.Append(Out);
    }
    _triStream.RestartStrip();
}

///<
bool IsBoundary(float3 _UVW)
{
	return  (_UVW.x < GridSpacing.x || _UVW.x>(1.0f-GridSpacing.x) || _UVW.y<GridSpacing.y || _UVW.y>(1.0f-GridSpacing.y) || _UVW.z<GridSpacing.z || _UVW.z>(1.0f-GridSpacing.z));
}

///<
float3 GetSplattedVelocity(float3 _UVW, SamplerState _sampler)
{
	float4 samp=txUp.SampleLevel(_sampler,_UVW,0);
	
	if (samp.w>0.01)
		return samp.xyz/samp.w;
	else
		return 0;
}

///<
float GetDensity(float3 _UVW)
{
	return txUp.Sample(samPoint,_UVW).w;
}

///<
float GetWeaklyPressure(float3 _UVW)
{
	float pr= (txUp.Sample(samPoint,_UVW).w)/InitialDensity.x; 
	
	return 0.02f*pow(pr,4); 
}

#include "SPHForces.hlsl"

///<
float3 GetDensityGradient(float3 _UVW)
{
	float dR=GetWeaklyPressure( _UVW+float3(GridSpacing.x,0,0) );
	float dL=GetWeaklyPressure( _UVW-float3(GridSpacing.x,0,0) );	
	
	float dT=GetWeaklyPressure( _UVW+float3(0,GridSpacing.y,0) );
	float dD=GetWeaklyPressure( _UVW-float3(0,GridSpacing.y,0) );
	
	float dF=GetWeaklyPressure( _UVW+float3(0,0,GridSpacing.z) );
	float dB=GetWeaklyPressure( _UVW-float3(0,0,GridSpacing.z) );
	
	return float3(dR-dL,dT-dD,dF-dB);
}

///<
float GetHeightDensity(float3 _UVW)
{

	float height = (txWaterHeight.Sample(samLinear,_UVW.xz).x);
	
	float dh = (max(height-(1.0-_UVW.y),0));
	float d=0;
	if (height>0.01f)
		d = (dh/height);
	
	return d;//(exp(d)-1.0f)/(exp(1.0f)-1.0f);
}

///<
float3 GetHeightGradient(float3 _UVW)
{
	float hR=GetHeightDensity( _UVW+float3(GridSpacing.x,0,0) );
	float hL=GetHeightDensity( _UVW-float3(GridSpacing.x,0,0) );	
	
	float hT=GetHeightDensity( _UVW+float3(0,GridSpacing.y,0) );
	float hD=GetHeightDensity( _UVW-float3(0,GridSpacing.y,0) );
	
	float hF=GetHeightDensity( _UVW+float3(0,0,GridSpacing.z) );
	float hB=GetHeightDensity( _UVW-float3(0,0,GridSpacing.z) );
	
	float3 grad=float3(hR-hL,hT-hD,hF-hB);
	return 1200.0f*grad;
}

///<
float4 PS_AddDensityGradient(GS_OUTPUT _input) : SV_Target
{	

const float3 Directions[5]=
	{
		float3(1,0,0),
		float3(-1,0,0),

		float3(0,1,0),
		
		float3(0,0,-1),
		float3(0,0,1)	
	};
	
	const float3 UVW = _input.UV;

	float3 dP		= GetDensityGradient(UVW);	
	float3 u		= GetSplattedVelocity(UVW, samPoint);
	float p			= GetDensity(UVW);
	
	float3 dE		= GetObstacleGradient(UVW);
	float3 dH		= GetHeightGradient(UVW);
		
	float3 grav=0;
	
	if(p>SurfaceDensity)
		grav=Gravity.xyz; 
		
	float3 du = u + dt*(-GridSpacing.x*(PressureScale*(dP+dE) + dH) + grav); 
	
	for (int i=0; i<5;++i)
	{
		if (IsBoundary(UVW+(GridSpacing*Directions[i]))) 
		{	
			float3 SetToZero=(1-abs(Directions[i]));
			du.xyz*=SetToZero;
		}
	}	

	return float4(du,p);
}

///< Transport
struct PARTICLE_DATA
{
    float4 Pos : POSITION;
    float4 Data: TEXCOORD0; 
};

///<
PARTICLE_DATA VS_AdvanceParticles(PARTICLE_DATA _Input)
{
	PARTICLE_DATA particle;	
		
	particle.Data	= _Input.Data;
	
	float3 posX = _Input.Pos.xyz;	
	
	float3 u			= GetSplattedVelocity(posX, samLinear);		
	float3 uCorrected	= txD_Up.SampleLevel(samLinear, posX, 0).xyz;
	
	float3 oldVel = particle.Data.xyz;
	particle.Data.xyz = PIC_FLIP*uCorrected + (1.0-PIC_FLIP)*(particle.Data.xyz+(uCorrected-u)); 
	
	float dv=(length(particle.Data.xyz)-length(oldVel));
	
	particle.Data.w += dv;
	
	posX += 0.5f*particle.Data.xyz*dt*GridSpacing.z/GridSpacing.xyz;	 		
	posX.xyz = clamp(posX.xyz, 1.7f*GridSpacing.xyz, 1.0f-1.7f*GridSpacing.xyz);
	
	
	uCorrected=txD_Up.SampleLevel(samLinear, posX, 0).xyz;
	posX += 0.5f*uCorrected*dt*GridSpacing.z/GridSpacing.xyz;	 		
	posX.xyz = clamp(posX.xyz, 1.7f*GridSpacing.xyz, 1.0f-1.7f*GridSpacing.xyz);
	
	particle.Pos	= float4(posX,1);	
	
	return particle;
}

///<
struct PS_RENDERPARTICLE_INPUT
{
    float4 Pos : SV_POSITION;
    float4 Data: TEXCOORD0;
    float2 UV : TEXCOORD1;
};

///< Vertex Shader
PARTICLE_DATA VS_RenderParticles(PARTICLE_DATA _input)
{
	PARTICLE_DATA output;
	
	float3 pos=_input.Pos.xyz;
	pos.y=1.0f-pos.y;
	
	float3 gridsRatio=GridSpacing.x/GridSpacing.xyz;
	pos = float3(pos*22.0f*gridsRatio ) + float3(-11,0,-11);
	
	output.Pos = float4(pos, _input.Pos.w);
	output.Data = _input.Data;

    return output;    
}

[maxvertexcount(4)]
void GS_RenderParticles(point PARTICLE_DATA input[1], inout TriangleStream<PS_RENDERPARTICLE_INPUT> _SpriteStream)
{
    PS_RENDERPARTICLE_INPUT Particle;
    
    const float Radius = 0.15f;

	const float3 Positions[4]
	=
	{
		float3( -1, 1, 0 ),
		float3( 1, 1, 0 ),
		float3( -1, -1, 0 ),
		float3( 1, -1, 0 )
	};
	
	const float2 UVs[4]
	=
	{
		float2( 0, 0 ),
		float2( 1, 0 ),
		float2( 0, 1 ),
		float2( 1, 1 )
	};
	    
	Particle.Data = input[0].Data;
    for (int i=0; i<4; i++)
    {
		float3 vert		=	mul( Positions[i].xyz, (float3x3)ViewInv);
        Particle.Pos	=	input[0].Pos + float4(Radius*vert,0);
        
        Particle.Pos	=	mul( Particle.Pos, View );        
        Particle.Pos 	=	mul( Proj, Particle.Pos );
        
        Particle.UV = UVs[i];
		
        _SpriteStream.Append(Particle);
    }
    _SpriteStream.RestartStrip();
}

///<
float GetHalow(float2 _UV)
{
	float r = length((_UV-0.5f)/0.5f);
	r=min(r,1);
	float d=(exp(1.0f-r)-1.0f)/(exp(1.0f)-1.0f);
	
	return d;
}

///< Pixel Shader
float4 PS_RenderParticles( PS_RENDERPARTICLE_INPUT _input) : SV_Target
{
	float3 blue=float3(0.2, 0.1, 1.2);
	float3 white=float3(1.1,1.1,1.1);
	
	float a=GetHalow(_input.UV);
	float fact = min(2.4f*length(_input.Data.xyz), 0.7f); //max(length(_input.Data.UVW)-1.0f,0); //);
	
	float3 c = (1.0-fact)*blue + fact*white;
	
	return float4(c, a);   
}

#include "SPHFLIP.hlsl"

