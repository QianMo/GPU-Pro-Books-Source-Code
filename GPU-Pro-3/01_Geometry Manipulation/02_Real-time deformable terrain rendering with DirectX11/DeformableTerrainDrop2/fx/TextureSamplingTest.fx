#include <common.fxh>

struct GenerateQuadVS_OUTPUT
{
    float4 m_ScreenPos_PS : SV_POSITION;
    float2 m_UV : TEXCOORD0;
};

Texture2D<float> g_SrcTex; // Source UNORM texture

GenerateQuadVS_OUTPUT GenerateQuadVS( in uint VertexId : SV_VertexID )
{
    float4 MinMaxXY = float4(-1,1,1,-1);
    float4 MinMaxUV = float4(0,0,1,1);
    
    GenerateQuadVS_OUTPUT Verts[4] = 
    {
        {float4(MinMaxXY.xy, 0.5, 1.0), MinMaxUV.xy}, 
        {float4(MinMaxXY.xw, 0.5, 1.0), MinMaxUV.xw},
        {float4(MinMaxXY.zy, 0.5, 1.0), MinMaxUV.zy},
        {float4(MinMaxXY.zw, 0.5, 1.0), MinMaxUV.zw}
    };

    return Verts[VertexId];
}



float TestPS(GenerateQuadVS_OUTPUT In) : SV_TARGET
{
    float2 TexSize;
    g_SrcTex.GetDimensions(TexSize.x, TexSize.y);
    // Sample the texture. The result should be average of two neighboring texels:
    //     _____ _____
    //    |     |     |
    //    |     X     |    X - sampling point
    //    |_____|_____|
    //
    return g_SrcTex.Sample(samLinearClamp, In.m_UV + float2(0.5/TexSize.x, 0) ) * HEIGHT_MAP_SAMPLING_SCALE;
}

technique11 TestTextureSamplingTech
{
    pass 
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, GenerateQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, TestPS() ) );
    }
}
