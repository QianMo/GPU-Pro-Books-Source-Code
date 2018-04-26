Texture2D importanceMap;
Texture2D gridDirectionMap;
#define gridDimension uint2(16, 64)

float3 getGridDirection(uint2 gridIndex)
{
	return gridDirectionMap.Load(uint3(gridIndex, 0));
}

float4 psEvaluateImportance(QuadOutput input) : SV_TARGET
{
	uint2 targetPos = (uint2)input.pos.xy / gridDimension;
	targetPos += uint2(200, 200);

	float3 normal = geometryViewportMap.Load(int3(targetPos, 0));
	
	float3 tangent = normalize(cross(normal, float3(0.57, 0.57, 0.57)));
	float3 binormal = cross(normal, tangent);

	// compute sample dir
	float3 sam = getGridDirection(input.pos.xy % gridDimension);

	// sample dir to world space
	float3 sampleDir = sam.x * tangent + sam.y * binormal + sam.z * normal;

	// return env(sample dir)
	return envMap.SampleLevel(linearSampler, sampleDir, 0).b;
}

technique10 evaluateImportance
{
	pass P0
	{
        SetVertexShader ( CompileShader( vs_4_0, vsQuad() ) );
        SetGeometryShader( NULL );
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psEvaluateImportance() ) );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
	}
}

void vsHalftoning()
{
}

[maxvertexcount(32)]
void gsInstantHalftoning( uint primitiveId : SV_primitiveID,
							   inout PointStream<EnvironmentSample> sampleStream )
{
	uint2 targetPixel = uint2(primitiveId % 128, primitiveId / 128);
	targetPixel += tileCorner;

	float4 screenPos = float4(targetPixel.xy, 0, 0) / float4(viewportSize.xy, 1, 1) * float4(2, -2, 0, 0) + float4(-1, 1, 0, 1);
	float4 geo = geometryViewportMap.Load(int3(targetPixel, 0));
	if(geo.w == 0)
		return;
	screenPos.z = geo.w;
	float3 normal = geo.xyz;
	float3 tangent = normalize(cross(normal, float3(0.57, 0.57, 0.57)));
	float3 binormal = cross(normal, tangent);
	float3x3 tangentFrame = float3x3(tangent, binormal, normal);

	EnvironmentSample sample;
	sample.screenPos = screenPos;

	float sumImpo = 0;
	[loop]for(uint u=0; u<gridDimension.y; u++)
	{
		[loop]for(uint v=0; v<gridDimension.x; v++)
		{
			float3 sam = getGridDirection(uint2(v, u));
			sample.dir.xyz = mul(sam, tangentFrame);
			sumImpo += envMap.SampleLevel(linearSampler, sample.dir.xyz, 0).r;
		}
	}
	float impoTreshold = sumImpo / 32.0;

	float iup = 0;
	float4 spreadRow[4] = {{iup, iup, iup, iup},{iup, iup, iup, iup},{iup, iup, iup, iup},{iup, iup, iup, iup}};
	float spreadPixel = iup;
	float spreadDiagonal = 0;
	[loop]for(uint j=0; j<gridDimension.y; j++)
	{
		float impo;
		uint kper4 =0;
		[loop]for(uint k=0; k<gridDimension.x; k+=4)
		{
			float rowSpreadAccumulator[4];
			for(uint xi=0; xi<4; xi++)
			{
				float3 sam = getGridDirection(uint2(k+xi, j));
				sample.dir.xyz = mul(sam, tangentFrame);//sam.x * tangent + sam.y * binormal + sam.z * normal;
				float impoc = envMap.SampleLevel(linearSampler, sample.dir.xyz, 0).r;
				impo = impoc + spreadRow[kper4][xi] + spreadPixel;
				if(impo > impoTreshold * 0.5)
				{
					sample.dir.w = impoc / impoTreshold * 52;
					sampleStream.Append( sample );
					impo -= impoTreshold;
				}
				rowSpreadAccumulator[xi] = impo * 0.375 + spreadDiagonal;
				spreadPixel = impo * 0.375;
				spreadDiagonal = impo * 0.25;
			}
			spreadRow[kper4] = float4(rowSpreadAccumulator[0], rowSpreadAccumulator[1], rowSpreadAccumulator[2], rowSpreadAccumulator[3]);
			kper4++;
		}
		j++;
		[loop]for(int k=gridDimension.x-5; k>=0; k-=4)
		{
			kper4--;
			float rowSpreadAccumulator[4];
			for(int xi=3; xi>=0; xi--)
			{
				float3 sam = getGridDirection(uint2(k+xi, j));
				sample.dir.xyz = mul(sam, tangentFrame);//sam.x * tangent + sam.y * binormal + sam.z * normal;
				float impoc = envMap.SampleLevel(linearSampler, sample.dir.xyz, 0).r;
				impo = impoc + spreadRow[kper4][xi] + spreadPixel;
				if(impo > impoTreshold * 0.5)
				{
					sample.dir.w = impoc / impoTreshold * 52;
					sampleStream.Append( sample );
					impo -= impoTreshold;
				}
				rowSpreadAccumulator[xi] = impo * 0.375 + spreadDiagonal;
				spreadPixel = impo * 0.375;
				spreadDiagonal = impo * 0.25;
			}
			spreadRow[kper4] = float4(rowSpreadAccumulator[0], rowSpreadAccumulator[1], rowSpreadAccumulator[2], rowSpreadAccumulator[3]);
		}
	}
}

technique10 instantHalftoning
{
    pass P0
	{
        SetVertexShader ( CompileShader( vs_4_0, vsHalftoning() ) );
        SetGeometryShader( 
	        ConstructGSWithSO( CompileShader( gs_4_0, gsInstantHalftoning()),
				"POSITION.xyzw; DIRECTION.xyzw" ) 
				);
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( NULL );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

[maxvertexcount(32)]
void gsRandomHalftoning( uint primitiveId : SV_primitiveID,
							   inout PointStream<EnvironmentSample> sampleStream )
{
	uint2 targetPixel = uint2(primitiveId % 128, primitiveId / 128);
	targetPixel += tileCorner;

	float4 screenPos = float4(targetPixel.xy, 0, 0) / float4(viewportSize.xy, 1, 1) * float4(2, -2, 0, 0) + float4(-1, 1, 0, 1);
	float4 geo = geometryViewportMap.Load(int3(targetPixel, 0));
	if(geo.w == 0)
		return;
	screenPos.z = geo.w;
	float3 normal = geo.xyz;
	float3 tangent = normalize(cross(normal, float3(0.57, 0.57, 0.57)));
	float3 binormal = cross(normal, tangent);
	float3x3 tangentFrame = float3x3(tangent, binormal, normal);

	EnvironmentSample sample;
	sample.screenPos = screenPos;

	float sumImpo = 0;
	[loop]for(uint u=0; u<gridDimension.y; u++)
	{
		[loop]for(uint v=0; v<gridDimension.x; v++)
		{
			float3 sam = getGridDirection(uint2(v, u));
			sample.dir.xyz = mul(sam, tangentFrame);
			sumImpo += envMap.SampleLevel(linearSampler, sample.dir.xyz, 0).r;
		}
	}
	float impoTreshold = sumImpo / 32.0;

	[loop]for(uint j=0; j<gridDimension.y; j++)
	{
		[loop]for(uint k=0; k<gridDimension.x; k++)
		{
			float3 sam = getGridDirection(uint2(k, j));
			sample.dir.xyz = mul(sam, tangentFrame);//sam.x * tangent + sam.y * binormal + sam.z * normal;
			float impoc = envMap.SampleLevel(linearSampler, sample.dir.xyz, 0).r;
			impoc /= impoTreshold;
			float rnd = randomMap.Load(uint3((targetPixel.x + k * 61)%512, (targetPixel.y + j*53)%512, 0)).w;
			if(impoc > rnd)
			{
				sample.dir.w = impoc * 64;
				sampleStream.Append( sample );
			}
		}
	}
}

technique10 randomHalftoning
{
    pass P0
	{
        SetVertexShader ( CompileShader( vs_4_0, vsHalftoning() ) );
        SetGeometryShader( 
	        ConstructGSWithSO( CompileShader( gs_4_0, gsRandomHalftoning()),
				"POSITION.xyzw; DIRECTION.xyzw" ) 
				);
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( NULL );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

[maxvertexcount(32)]
void gsInstantHalftoningForSplatting( uint primitiveId : SV_primitiveID,
							   inout PointStream<EnvironmentSample> sampleStream )
{
	uint2 targetPixel = uint2(primitiveId % 128, primitiveId / 128);
	uint2 cica = targetPixel;
	targetPixel += uint2(200,200);

	float4 screenPos = float4(targetPixel.xy, 0, 0) / float4(viewportSize.xy, 1, 1) * float4(2, -2, 0, 0) + float4(-1, 1, 0, 1);
	float4 geo = geometryViewportMap.Load(int3(targetPixel, 0));
	if(geo.w == 0)
		return;
	screenPos.z = geo.w;
	float3 normal = geo.xyz;
	float3 tangent = normalize(cross(normal, float3(0.57, 0.57, 0.57)));
	float3 binormal = cross(normal, tangent);
	float3x3 tangentFrame = float3x3(tangent, binormal, normal);

	EnvironmentSample sample;
	sample.screenPos = screenPos;

	float sumImpo = 0;
	[loop]for(uint u=0; u<gridDimension.y; u++)
	{
		[loop]for(uint v=0; v<gridDimension.x; v++)
		{
			float3 sam = getGridDirection(uint2(v, u));
			sample.dir.xyz = mul(sam, tangentFrame);
			sumImpo += envMap.SampleLevel(linearSampler, sample.dir.xyz, 0).r;
		}
	}
	float iup = sumImpo / 32.0 * 0.4;
	sumImpo += iup * 17;
	float impoTreshold = sumImpo / 49.0;

	float4 spreadRow[4] = {{iup, iup, iup, iup},{iup, iup, iup, iup},{iup, iup, iup, iup},{iup, iup, iup, iup}};
	float spreadPixel = iup;
	float spreadDiagonal = 0;
	[loop]for(uint j=0; j<gridDimension.y; j++)
	{
		float impo;
		uint kper4 =0;
		[loop]for(uint k=0; k<gridDimension.x; k+=4)
		{
			float rowSpreadAccumulator[4];
			for(uint xi=0; xi<4; xi++)
			{
				float3 sam = getGridDirection(uint2(k+xi, j));
				sample.dir.xyz = mul(sam, tangentFrame);//sam.x * tangent + sam.y * binormal + sam.z * normal;
				float impoc = envMap.SampleLevel(linearSampler, sample.dir.xyz, 0).r;
				impo = impoc + spreadRow[kper4][xi] + spreadPixel;
				if(impo > impoTreshold)
				{
					sample.dir.w = impoc * 2;
					sample.dir = float4(cica, k, j);
					sampleStream.Append( sample );
					impo -= impoTreshold;
				}
				rowSpreadAccumulator[xi] = impo * 0.375 + spreadDiagonal;
				spreadPixel = impo * 0.375;
				spreadDiagonal = impo * 0.25;
			}
			spreadRow[kper4] = float4(rowSpreadAccumulator[0], rowSpreadAccumulator[1], rowSpreadAccumulator[2], rowSpreadAccumulator[3]);
			kper4++;
		}
		j++;
		[loop]for(int k=gridDimension.x-5; k>=0; k-=4)
		{
			kper4--;
			float rowSpreadAccumulator[4];
			for(int xi=3; xi>=0; xi--)
			{
				float3 sam = getGridDirection(uint2(k+xi, j));
				sample.dir.xyz = mul(sam, tangentFrame);//sam.x * tangent + sam.y * binormal + sam.z * normal;
				float impoc = envMap.SampleLevel(linearSampler, sample.dir.xyz, 0).r;
				impo = impoc + spreadRow[kper4][xi] + spreadPixel;
				if(impo > impoTreshold)
				{
					sample.dir.w = impoc * 2;
					sample.dir = float4(cica, k, j);
					sampleStream.Append( sample );
					impo -= impoTreshold;
				}
				rowSpreadAccumulator[xi] = impo * 0.375 + spreadDiagonal;
				spreadPixel = impo * 0.375;
				spreadDiagonal = impo * 0.25;
			}
			spreadRow[kper4] = float4(rowSpreadAccumulator[0], rowSpreadAccumulator[1], rowSpreadAccumulator[2], rowSpreadAccumulator[3]);
		}
	}
}

technique10 instantHalftoningForSplatting
{
    pass P0
	{
        SetVertexShader ( CompileShader( vs_4_0, vsHalftoning() ) );
        SetGeometryShader( 
	        ConstructGSWithSO( CompileShader( gs_4_0, gsInstantHalftoningForSplatting()),
				"POSITION.xyzw; DIRECTION.xyzw" ) 
				);
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( NULL );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}

float4 vsSplatting(EnvironmentSample input) : SV_POSITION
{

	uint2 tgt = input.dir.xy * gridDimension + input.dir.zw;

	float4 screenPos = float4(tgt.xy, 0, 0) / float4(viewportSize.xy, 1, 1) * float4(2, -2, 0, 0) + float4(-1, 1, 0, 1);
	return screenPos;
}

float psSplatting() : SV_TARGET
{
	return float4(1, 0.3, 0.3, 1);
}

technique10 splatting
{
    pass P0
	{
        SetVertexShader ( CompileShader( vs_4_0, vsSplatting() ) );
        SetGeometryShader( NULL	);
		SetRasterizerState( defaultRasterizer );
        SetPixelShader( CompileShader( ps_4_0, psSplatting() ) );
		SetDepthStencilState( noDepthTestCompositor, 0);
		SetBlendState(defaultBlender, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff);
    }
}