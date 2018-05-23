

#include "SPHParams.hlsl"


struct SPH_PARTICLE_DATA
{
    float4 Pos : POSITION;
    float4 Data: TEXCOORD0; ///< Vorticity
};

struct GS_SPLATTING_DATA
{
	float4 Pos : SV_POSITION;
    float3 Radius : TEXCOORD0;
	float3 Velocity : TEXCOORD1;
    
    uint SliceIndex          : SV_RenderTargetArrayIndex;  ///< used to choose the destination slice
};

///<
SPH_PARTICLE_DATA Splatting_VS(SPH_PARTICLE_DATA _Input)
{
	SPH_PARTICLE_DATA particle;	
	
	particle.Pos = _Input.Pos;
	particle.Data = _Input.Data;
	
	return particle;
}

float2 UVToWorld(float2 _UV)
{
	_UV.y=1.0f-_UV.y;
	return (_UV-0.5f)*2.0f;
}
//<
void CreateQuad(float4 _x, float3 _v, float _zSliceOffset, float _fNumSlices, inout TriangleStream<GS_SPLATTING_DATA> _SpriteStream)
{
	const float3 Positions[4]=
	{
		float3( -1, 1, 0 ),
		float3( 1, 1, 0 ),
		float3( -1, -1, 0 ),
		float3( 1, -1, 0 )
	};     
	
	GS_SPLATTING_DATA Vertex;
		
	Vertex.Velocity = _v;
	
	for(int i=0; i<4; ++i)
	{
		Vertex.Pos = float4(_x.xyz,1) + GridSpacing*float4(_fNumSlices*Positions[i].xy, _zSliceOffset, 0);
		
		Vertex.SliceIndex = Vertex.Pos.z/GridSpacing.z;

		float3 dx = (_x.xyz-Vertex.Pos.xyz);
		float3 dr = _fNumSlices*GridSpacing;
		float3 r = dx/dr;
		Vertex.Pos.xy=UVToWorld(Vertex.Pos.xy);

		Vertex.Radius = r;		
		
		_SpriteStream.Append(Vertex);
	}
	_SpriteStream.RestartStrip();
}

///< 3=24
///< 4=36
///< 6=52
[maxvertexcount(36)]
void Splatting_GS(point SPH_PARTICLE_DATA _input[1], inout TriangleStream<GS_SPLATTING_DATA> _SpriteStream)
{    
	const float fNumSlices	= 4.0f;    

	for (float sliceOffset=-fNumSlices; sliceOffset<fNumSlices; sliceOffset+=1.0)
	{		
		CreateQuad(_input[0].Pos, _input[0].Data.xyz, sliceOffset, fNumSlices, _SpriteStream);		 
	}
}

///< Pixel Shader
float4 Splatting_PS(GS_SPLATTING_DATA _input) : SV_Target
{
	float r = length(_input.Radius.xyz);
	r=min(r,1);

	float d=(exp(1.0f-r)-1.0f)/(exp(1.0f)-1.0f);
	
	return float4(_input.Velocity*d, d);    
}