#include "Math.fxh"

cbuffer def
{
	float4x4 matVP;
}

struct VSIn
{
	float4 rig 		: TEXCOORD0;

	float3 pos		: POSITION;
	float4 orient	: TEXCOORD1;
	float4 scale	: TEXCOORD2;
	float2 centre	: TEXCOORD3;
};

struct VSOut
{
	float4 pos 	: SV_Position;
};

VSOut main_vs( VSIn i )
{
	float4 pos 		= i.rig; 
	float3 scale 	= i.scale;
	float3 centre 	= float3( i.scale.w, i.centre );

	// R8G8B8A8_SNORM bugs on ati, have to use unorm
	pos *= 2;
	pos -= 1;

	pos.xyz *= scale;
	pos.xyz += centre;
	pos.xyz = rotByQuat( pos.xyz, i.orient );
	pos.xyz += i.pos;

	VSOut o;
	o.pos = mul( pos, matVP );

	return o;
}

float4 main_ps() : SV_Target
{
	return float4( 0, 1, 0, 0 );
}

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );
	}
}