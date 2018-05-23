
float4x4 MVP;
float3 base_a;
float3 base_b;


texture tex;
sampler stex = sampler_state {
	Texture = <tex>;
	MinFilter = Linear;
	MagFilter = Linear;
	MipFilter = Point;
};

void VS_Render(	float4 pos			: POSITION,
				out float4 hpos		: POSITION,
				out float3 _p		: TEXCOORD0 )
{
	hpos = mul(pos,MVP);
	_p = pos.xyz;
}

float4 PS_Render( float3 p : TEXCOORD0 ) : COLOR
{
	// generate texcoords
	float2 uv;
	p = normalize(p);
	uv.x = atan2(p.x,p.y)*(1./2/3.141593*5);
	uv.y = -p.z;

	// sample texture
	float2 tx = tex2D(stex,uv).xy;
	float lum = tx.y*tx.y;

	// decode
    float3 txc = lerp(base_a,base_b,tx.x);
    float clum = dot( txc, float3(0.2126,0.7152,0.0722) );
	txc *= lum/(clum + 0.00000001);
	txc = sqrt(txc);
	
	return float4(txc,1);
}

technique tech {
	pass {
		VertexShader = compile vs_3_0 VS_Render();
		PixelShader = compile ps_3_0 PS_Render();
	}
}
