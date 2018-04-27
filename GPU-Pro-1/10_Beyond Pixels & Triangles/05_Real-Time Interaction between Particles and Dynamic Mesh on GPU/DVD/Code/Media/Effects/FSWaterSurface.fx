//

cbuffer def
{
	float4x3	invV;
	float4x4 	invVP;
	float3 		wLDir0;
	float3 		wLDir1;
	float3		wCPos;
	float3		waterColr;
	float		time;
	float		refrK = 0.05;
};


struct VSIn
{
	float4 pos : POSITION;	
};

struct VSOut
{
	float4 pos 	: SV_Position;
	float2 spos : SPOS;
	float2 tex 	: TEXCOORD;
};

typedef VSOut PSIn;

VSOut main_vs( VSIn i )
{
	VSOut o;
	o.pos 		= i.pos;
	o.spos.xy	= i.pos.xy;
	o.tex		= float2( i.pos.x, -i.pos.y ) * 0.5 + 0.5;

	return o;
}

Texture2D tex;
TextureCube Scene;

SamplerState ss
{
	Filter = MIN_MAG_LINEAR_MIP_POINT;
	AddressU = CLAMP;
	AddressV = CLAMP;
	AddressW = CLAMP;
};

SamplerState ssw
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = WRAP;
	AddressV = WRAP;
};


// simplified refraction routing( returns unnormalized v )
float3 Refract( float3 v, float3 n, float k )
{
	float dpres = dot( v, n );
	float3 pv =  dpres * n;
	float3 d = pv - v;

	[flatten]
	if( dpres < -0.0001 )
		return v + k * d;
	else       
		return v;
}



float4 main_ps( PSIn i ) : SV_Target
{
	float4 norm_p = tex.SampleLevel( ss, i.tex, 0 );

	clip( 12.0 - norm_p.w );

	float3 norm = norm_p.xyz;

	{
		float l = length( norm );
		clip( l - 0.1 );
		norm /= l;
	}

	norm = mul( norm.xyz, (float3x3)invV );

	// return float4(norm.xyz,0);

	float3 wpos = mul( float4( i.spos.xy*norm_p.w, norm_p.w, 1 ), invVP );

	float3 wview = normalize( wpos - wCPos );

	float3 reflVec = reflect( wview, norm );

	float3 reflColr = Scene.Sample( ss, reflVec ) * waterColr * 1.33;

	float3 refrVec = Refract( wview, norm, refrK );

	float3 refrColr = Scene.Sample( ss, refrVec ) * waterColr;

	float fres = saturate( dot( -wview, norm ) );

	float3 colr = lerp( reflColr, refrColr, fres );

	float d0 = dot( norm, -wLDir0 );
	float d1 = dot( norm, -wLDir1 );

	float s = pow( saturate( dot( reflect( wLDir0, norm ), -wview ) ), 48 ) * (d0 > 0.0);

	s += pow( saturate( dot( reflect( wLDir1, norm ), -wview ) ), 192 ) * (d1 > 0.0) * 1.0;

	return float4( (colr + s*0.75), s );
}

DepthStencilState DSS_Disable
{
	DepthEnable = FALSE;
};

DepthStencilState DSS_Default
{
};

technique10 main
{
	pass
	{
		SetVertexShader( CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( NULL );
		SetPixelShader( CompileShader( ps_4_0, main_ps() ) );

		SetDepthStencilState( DSS_Disable, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}

