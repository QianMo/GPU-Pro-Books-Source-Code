
cbuffer FrameParams
{
    matrix g_mWorldViewProj;
}

struct VS_OUTPUT
{
    float4 Pos : SV_POSITION;     // Projection coord
	float4 Color : COLOR;
};

#pragma warning (disable: 3571) // warning X3571: pow(f, e) will not work for negative f


VS_OUTPUT RenderBoundBoxVS( uint id : SV_VertexID,
                            float3 BoundBoxMinXYZ : BOUND_BOX_MIN_XYZ,
                            float3 BoundBoxMaxXYZ : BOUND_BOX_MAX_XYZ,
                            float4 BoundBoxColor  : BOUND_BOX_COLOR )
{
    float4 BoxCorners[8]=
    {
        float4(BoundBoxMinXYZ.x, BoundBoxMinXYZ.y, BoundBoxMinXYZ.z, 1.f),
        float4(BoundBoxMinXYZ.x, BoundBoxMaxXYZ.y, BoundBoxMinXYZ.z, 1.f),
        float4(BoundBoxMaxXYZ.x, BoundBoxMaxXYZ.y, BoundBoxMinXYZ.z, 1.f),
        float4(BoundBoxMaxXYZ.x, BoundBoxMinXYZ.y, BoundBoxMinXYZ.z, 1.f),

        float4(BoundBoxMinXYZ.x, BoundBoxMinXYZ.y, BoundBoxMaxXYZ.z, 1.f),
        float4(BoundBoxMinXYZ.x, BoundBoxMaxXYZ.y, BoundBoxMaxXYZ.z, 1.f),
        float4(BoundBoxMaxXYZ.x, BoundBoxMaxXYZ.y, BoundBoxMaxXYZ.z, 1.f),
        float4(BoundBoxMaxXYZ.x, BoundBoxMinXYZ.y, BoundBoxMaxXYZ.z, 1.f),
    };

    const int RibIndices[12*2] = {0,1, 1,2, 2,3, 3,0,
                                  4,5, 5,6, 6,7, 7,4,
                                  0,4, 1,5, 2,6, 3,7};
    VS_OUTPUT Out;
    Out.Pos = mul( BoxCorners[RibIndices[id]], g_mWorldViewProj );
    Out.Color = pow(BoundBoxColor, 2.2); // gamma correction
    return Out;
}

float4 g_vQuadTreePreviewPos_PS;
float4 g_vScreenPixelSize;
VS_OUTPUT RenderQuadTreeVS( uint id : SV_VertexID,
                            float3 BoundBoxMinXYZ : BOUND_BOX_MIN_XYZ,
                            float3 BoundBoxMaxXYZ : BOUND_BOX_MAX_XYZ,
                            float4 BoundBoxColor  : BOUND_BOX_COLOR )
{
	BoundBoxMinXYZ.z = 1 - BoundBoxMinXYZ.z;
	BoundBoxMaxXYZ.z = 1 - BoundBoxMaxXYZ.z;
	BoundBoxMinXYZ.xz *= g_vQuadTreePreviewPos_PS.zw * float2(1,-1);
	BoundBoxMaxXYZ.xz *= g_vQuadTreePreviewPos_PS.zw * float2(1,-1);
	BoundBoxMinXYZ.xz += g_vQuadTreePreviewPos_PS.xy;
	BoundBoxMaxXYZ.xz += g_vQuadTreePreviewPos_PS.xy;
	
	BoundBoxMinXYZ.xz += g_vScreenPixelSize.xy/2.f * float2(1,-1);
	BoundBoxMaxXYZ.xz -= g_vScreenPixelSize.xy/2.f * float2(1,-1);
    float4 QuadCorners[4]=
    {
        float4(BoundBoxMinXYZ.x, BoundBoxMinXYZ.z, 0.5, 1.f),
        float4(BoundBoxMinXYZ.x, BoundBoxMaxXYZ.z, 0.5, 1.f),
        float4(BoundBoxMaxXYZ.x, BoundBoxMaxXYZ.z, 0.5, 1.f),
        float4(BoundBoxMaxXYZ.x, BoundBoxMinXYZ.z, 0.5, 1.f),
    };

    const int RibIndices[4*2] = {0,1, 1,2, 2,3, 3,0};
    VS_OUTPUT Out;
    Out.Pos = QuadCorners[RibIndices[id]];
    Out.Color = pow(BoundBoxColor, 2.2); // gamma correction
    return Out;
}


float4 RenderBoundBoxPS(VS_OUTPUT In) : SV_TARGET
{
    return In.Color;
}

DepthStencilState DSS_EnableDepthTest
{
    DepthEnable = TRUE;
    DepthWriteMask = ALL;
    StencilEnable = FALSE;
};

DepthStencilState DSS_DisableDepthTest
{
    DepthEnable = FALSE;
    DepthWriteMask = ZERO;
};

RasterizerState RS_SolidFill_NoCull
{
    FILLMODE = Solid;
    CullMode = NONE;
};

// Blend state disabling blending
BlendState BS_DisableBlending
{
    BlendEnable[0] = FALSE;
    BlendEnable[1] = FALSE;
    BlendEnable[2] = FALSE;
};

technique11 RenderBoundBox_FeatureLevel10
{
    pass P0
    {
        SetBlendState( BS_DisableBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_EnableDepthTest, 0 );

        SetVertexShader( CompileShader( vs_4_0, RenderBoundBoxVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, RenderBoundBoxPS() ) );
    }
}

technique11 RenderQuadTree_FeatureLevel10
{
    pass P0
    {
        SetBlendState( BS_DisableBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_DisableDepthTest, 0 );

        SetVertexShader( CompileShader( vs_4_0, RenderQuadTreeVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, RenderBoundBoxPS() ) );
    }
}
