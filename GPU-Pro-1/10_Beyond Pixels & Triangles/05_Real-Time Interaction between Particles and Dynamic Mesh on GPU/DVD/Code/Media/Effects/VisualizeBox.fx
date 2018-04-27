cbuffer def
{
	float4x4 	matWVP;
	float3 		pos;
	float3 		scale;
};

struct VSIn
{
	uint vidx : SV_VertexID;
};

struct VSOut
{
	uint dummy : DUMMY;
};

typedef VSOut GSIn;

struct GSOut
{
	float4 pos : SV_Position;
};

VSOut main_vs( VSIn i )
{
	VSOut o;
	o.dummy = i.vidx;
	
	return o;
}

// at least not 18 ;]
[maxvertexcount(17)]
void main_gs( in point GSIn ia[1], inout LineStream<GSOut> os )
{
	GSOut o;

	float sx = scale.x;
	float sy = scale.y;
	float sz = scale.z;

#define APPEND_PT(x,y,z) o.pos = mul( float4(pos + float3(x,y,z),1), matWVP ); os.Append( o );

	APPEND_PT(-sx,-sy,-sz);
	APPEND_PT(+sx,-sy,-sz);
	APPEND_PT(+sx,+sy,-sz);
	APPEND_PT(-sx,+sy,-sz);
	APPEND_PT(-sx,-sy,-sz);

	os.RestartStrip();

	APPEND_PT(-sx,-sy,-sz);
	APPEND_PT(-sx,-sy,+sz);
	APPEND_PT(+sx,-sy,+sz);
	APPEND_PT(+sx,-sy,-sz);

	os.RestartStrip();

	APPEND_PT(-sx,+sy,-sz);
	APPEND_PT(-sx,+sy,+sz);
	APPEND_PT(+sx,+sy,+sz);
	APPEND_PT(+sx,+sy,-sz);

	os.RestartStrip();

	APPEND_PT(-sx,-sy,+sz);
	APPEND_PT(-sx,+sy,+sz);

	os.RestartStrip();

	APPEND_PT(+sx,-sy,+sz);
	APPEND_PT(+sx,+sy,+sz);

}

float4 main_ps() : SV_Target0
{
	return float4( 1, 0, 0, 1 );
}

technique10 main
{
	pass
	{
		SetVertexShader( 	CompileShader( vs_4_0, main_vs() ) );
		SetGeometryShader( 	CompileShader( gs_4_0, main_gs() ) );
		SetPixelShader( 	CompileShader( ps_4_0, main_ps() ) );
	}
}