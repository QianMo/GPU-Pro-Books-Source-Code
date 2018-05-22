float extrudeBias = 0.1;
float extrudeAmt = 500;


struct IaosShadowVolume
{
    float4 pos			: POSITION;
    float3 normal		: NORMAL;
};

struct VsosShadowVolume
{
    float4 pos			: POSITION;
    float3 normal		: NORMAL;
};

typedef VsosTrafo GsosShadowVolume;

VsosShadowVolume vsShadowVolume(IaosShadowVolume input)
{
	VsosShadowVolume output = (VsosShadowVolume)0;
	output.pos = mul(input.pos, modelMatrix);
	output.normal = mul(modelMatrixInverse, float4(input.normal.xyz, 0.0));
	return output;
}

//
// Helper to detect a silhouette edge and extrude a volume from it
//
void detectAndProcessSilhouette( float3 flatNormal,
								 VsosShadowVolume v1,    // Shared vertex
                                 VsosShadowVolume v2,    // Shared vertex
                                 VsosShadowVolume vAdj,  // Adjacent triangle vertex
                                 inout TriangleStream<GsosShadowVolume> shadowTriangleStream // triangle stream
                                 )
{    
    float3 nAdj = cross( v2.pos - vAdj.pos, v1.pos - vAdj.pos );
    
	if( dot(flatNormal, nAdj) > 0)
		return;
    
	float3 lightPos = float3(lightx, lighty, lightz);

    float3 outpos[4];
    float3 extrude1 = normalize(v1.pos - lightPos);
    float3 extrude2 = normalize(v2.pos - lightPos);
        
    outpos[0] = v1.pos + extrudeBias*extrude1;
    outpos[1] = v1.pos + extrudeAmt*extrude1;
    outpos[2] = v2.pos + extrudeBias*extrude2;
    outpos[3] = v2.pos + extrudeAmt*extrude2;
        
    // Extrude silhouette to create two new triangles
    GsosShadowVolume output = (GsosShadowVolume)0;
    for(int v=0; v<4; v++)
    {
		output.worldPos = float4(outpos[v],1);
        output.pos = mul( float4(outpos[v],1), viewProjMatrix );
        shadowTriangleStream.Append( output );
    }
    shadowTriangleStream.RestartStrip();
}

//
// GS for generating shadow volumes
//
[maxvertexcount(18)]
void gsShadowVolume( triangleadj VsosShadowVolume input[6], inout TriangleStream<GsosShadowVolume> shadowTriangleStream )
{
    // Compute un-normalized triangle normal
    float3 flatNormal = cross( input[2].pos - input[0].pos, input[4].pos - input[0].pos );

	float3 lightPos = float3(lightx, lighty, lightz);
    

    // Compute direction from this triangle to the light
    float3 lightDir = lightPos - input[0].pos;
    
    //if we're facing the light
    if( dot(flatNormal, lightDir) > 0.0f )
    {
        // For each edge of the triangle, determine if it is a silhouette edge
        detectAndProcessSilhouette(lightDir, input[0], input[2], input[1], shadowTriangleStream );
        detectAndProcessSilhouette(lightPos - input[2].pos, input[2], input[4], input[3], shadowTriangleStream );
        detectAndProcessSilhouette(lightDir, input[4], input[0], input[5], shadowTriangleStream );
        
        //near cap
        GsosShadowVolume output = (GsosShadowVolume)0;
        for(int v=0; v<6; v+=2)
        {
            float3 extrude = normalize(input[v].pos - lightPos);

            float3 pos = input[v].pos + extrudeBias * extrude;
			output.worldPos = float4(pos, 1);
            output.pos = mul( float4(pos,1), viewProjMatrix );
            shadowTriangleStream.Append( output );
        }
        shadowTriangleStream.RestartStrip();
        
        //far cap (reverse the order)
        for(int v=4; v>=0; v-=2)
        {
            float3 extrude = normalize(input[v].pos - lightPos);
        
            float3 pos = input[v].pos + extrudeAmt*extrude;
			output.worldPos = float4(pos, 1);
            output.pos = mul( float4(pos,1), viewProjMatrix );
            shadowTriangleStream.Append( output );
        }
        shadowTriangleStream.RestartStrip();
    }
}

float4 psShadowVolume() : SV_TARGET
{
	return 1;
}

technique11 shadowvolume
{
	pass shadowvolume
	{
		SetVertexShader ( CompileShader( vs_5_0, vsShadowVolume() ) );
		SetGeometryShader ( CompileShader( gs_5_0, gsShadowVolume() ) );
		SetPixelShader( CompileShader( ps_5_0, psShadowVolume() ) );
	}
}