cbuffer def
{
	float	delta;
	float	gravity;

	float3	texVolumeTransform1;
	float3	texVolumeTransform2;

	float	texelStep;
	float	texelSize;

	float3	obstacleVolumeStart;
	float3	obstacleVolumeEnd;
	float3	obstacleVolumePos;
	float3	obstacleVolumeDims;

	uint4 cellBitsTransform;
	uint lastMapElement;
};

struct VSIn
{
	float4 pos_time	: POSITION;	// .w == time
	float3 speed	: SPEED;
};

typedef VSIn	VSOut;
typedef VSOut 	GSIn;
typedef GSIn	GSOut;

SamplerState ss
{
	Filter		= MIN_MAG_LINEAR_MIP_POINT;
	AddressU 	= CLAMP;
	AddressV 	= CLAMP;
};

Texture2D 		HeightmapTex;
Buffer<float3>	Vertices;

#ifdef MD_R16_UNORM_BLENDABLE
#define MD_BUFFEERMAP_FMT uint
#else
#define MD_BUFFEERMAP_FMT float
#endif

Buffer<MD_BUFFEERMAP_FMT>	BufferMap;
Buffer<uint4>				Indexes;

uint isNotInVolume( float3 pos, float3 start, float3 end )
{
	float3 inpos = min( end - pos, 0 );
	float3 inneg = min( pos - start, 0 );
	return asuint( dot( inpos, 1 ) + dot( inneg, 1 ) );
}

uint rotateBits( uint val )
{
	uint v1 = val & 0xff00;
	uint v2 = val << 16;
	                        // 01234567012345670123456701234567
	val = v1 | v2;			// 00000000111111111111111100000000

	v1 = val & 0x0f0f00;	// 00000000000011110000111100000000
	v2 = val >> 8 & 0xf0f0; // 00000000000000001111000011110000

	val = v1 | v2;			// 00000000000011111111111111110000

	v1 = val & 0x33330;     // 00000000000000110011001100110000
	v2 = val >> 4 & 0xcccc; // 00000000000000001100110011001100

	val = v1 | v2;          // 00000000000000111111111111111100

	v1 = val & 0x15554;     // 00000000000000010101010101010100
	v2 = val >> 2 & 0xaaaa; // 00000000000000001010101010101010

	val = (v1 | v2) >> cellBitsTransform.z & cellBitsTransform.w;

	return val;
}

VSOut main_vs( VSIn i )
{	
	float3 speed 	= i.speed;
	float3 pos 		= i.pos_time;
	float3 moveVec	= speed * delta;
	float3 nextPos	= pos + moveVec;

	float3 hgrav 	= float3(0, gravity, 0);

	float3 texPos = pos * texVolumeTransform1 + texVolumeTransform2;

	float h00 = HeightmapTex.SampleLevel( ss, texPos.xz, 0 );

	if( nextPos.y < h00 )
	{
		speed *= float3( 0.125, -0.125, 0.125 );
		speed += float3(1,0,0);

		i.pos_time.w *= 0.25;
		nextPos.y = h00;
	}
	else
	if( !isNotInVolume( pos, obstacleVolumeStart, obstacleVolumeEnd ) )
	{
		float3 diff = pos - obstacleVolumeStart;
		uint3 cellCoords = diff * obstacleVolumeDims;

		uint bufferCoord = cellCoords.z + 
							( cellCoords.y << cellBitsTransform.x ) + 
							( cellCoords.x << cellBitsTransform.y );

		bufferCoord = rotateBits( bufferCoord );

		uint indexesStart 	= BufferMap.Load( (int)bufferCoord - 1 );
		uint indexesEnd 	= BufferMap.Load( bufferCoord );

		// scan through triangles, see if we intersect one
		for( uint i = indexesStart; i < indexesEnd; i ++ )
		{
			uint3 idxes_x2 = Indexes.Load( i ) * 2;

			float3 verts[3];

			[unroll]
			for( uint ii = 0; ii < 3; ii++ )
			{
				verts[ii] = Vertices.Load( idxes_x2[ii] ) + obstacleVolumePos;
			}

			float3 AB = verts[1] - verts[0];
			float3 AC = verts[2] - verts[0];

			float3 triNorm = cross( AB, AC );

			float3 vec1 = pos 		- verts[0];
			float3 vec2 = nextPos 	- verts[0];

			float d1 = dot( vec1, triNorm );
			float d2 = dot( vec2, triNorm );

			// if d1 and d2 have different signs, then triangle plane was intersected
			if( (asuint(d1) ^ asuint(d2)) & 0x80000000 )
			{

				float3 OA = verts[0] - pos;
				float3 OB = verts[1] - pos;
				float3 OC = verts[2] - pos;

				float3 N1 = cross( OA, OB );
				float3 N2 = cross( OB, OC );
				float3 N3 = cross( OC, OA );

				uint3 d = asuint( float3( 	dot( N1, moveVec ), 
											dot( N2, moveVec ),
											dot( N3, moveVec ) ) );

				
				// if we're inside a pyramid (hence we intersect inside triangle) 
				// then all sign bits are 0
				if( d.x & d.y & d.z & 0x80000000 )
				{
					triNorm = normalize(triNorm);
					float3 normSpeed = -0.75 * dot( speed, triNorm ) * triNorm;
					speed += normSpeed;
					nextPos = pos + speed * delta;
					break;
				}
			}
		}
	}

	speed 				+= hgrav;

	VSOut o;

	o.pos_time.xyz		= nextPos;
	o.pos_time.w		= isNotInVolume( texPos, float3(0,h00,0), 1 ) ? 0 : max( i.pos_time.w - delta, 0 );
	o.speed 			= speed;

	return o;
}

[maxvertexcount(1)]
void main_gs( in point GSIn ia[1], inout PointStream<GSOut> os )
{
	if( asuint( ia[0].pos_time.w ) )
	{
		GSOut o;
		o.pos_time	= ia[0].pos_time;
		o.speed 	= ia[0].speed;

		os.Append( o );
	}
}

DepthStencilState DSS_Disabled
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
		SetGeometryShader( ConstructGSWithSO( CompileShader( gs_4_0, main_gs() ), "POSITION.xyzw;SPEED.xyz" ) );
		SetPixelShader( 0 );

		SetDepthStencilState( DSS_Disabled, 0 );
	}

	pass
	{
		SetDepthStencilState( DSS_Default, 0 );
	}
}