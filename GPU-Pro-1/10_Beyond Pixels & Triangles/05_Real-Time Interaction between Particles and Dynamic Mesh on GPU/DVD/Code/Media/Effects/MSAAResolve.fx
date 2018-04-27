//

struct VSOut
{
	float4 pos : SV_Position;
	float2 texc : TEXCOORD;
};

typedef VSOut PSIn;


VSOut main_vs( uint vid : SV_VertexID )
{

	float4 varray[4] = { float4(-1,-1,0,768),
						 float4(-1,+1,0,0),
						 float4(+1,-1,1024,768),
						 float4(+1,+1,1024,0) };

	VSOut o;
	o.pos 	= float4( varray[vid].xy, 0, 1 );
	o.texc  = varray[vid].zw;

	return o;
}


Texture2DMS<float4,8> SrcTex;

// the stupidest resolve ever ;]]
float4 main_ps( PSIn i ) : SV_Target0
{
	float4 s = 0;
	for( int ii = 0; ii < 8; ii ++ )
		s += SrcTex.Load( i.texc, ii );

	s /= 8;

	if( s.a == 0 )
		discard;

	return s;
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