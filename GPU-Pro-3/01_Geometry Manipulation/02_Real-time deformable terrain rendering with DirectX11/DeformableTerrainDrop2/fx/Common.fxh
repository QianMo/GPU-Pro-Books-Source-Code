SamplerState samLinearClamp
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

SamplerState samPointClamp
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
    AddressW = Clamp;
};

RasterizerState RS_SolidFill_NoCull
{
    FILLMODE = Solid;
    CullMode = NONE;
};

DepthStencilState DSS_DisableDepthTest
{
    DepthEnable = FALSE;
    DepthWriteMask = ZERO;
};

DepthStencilState DSS_EnableDepthTest
{
    DepthEnable = TRUE;
    DepthWriteMask = ALL;
};

// Blend state disabling blending
BlendState NoBlending
{
    BlendEnable[0] = FALSE;
    BlendEnable[1] = FALSE;
    BlendEnable[2] = FALSE;
};

#define HEIGHT_MAP_SAMPLING_SCALE 65535.f

uint QuantizeValue(uint Val, uint Quantizer)
{
    return (Val + Quantizer) / (2 * Quantizer + 1);
}

uint DequantizeValue(uint QuantizedVal, uint Quantizer)
{
    return QuantizedVal * (2 * Quantizer + 1); 
}

cbuffer cbFogParams
{
    float g_FogScale = 1.f / 50000.f;
    float4 g_FogColor = float4(40.f, 90.f, 150.f, 0) / 255.f;
}

float3 ApplyFog(in float3 Color, in float fDistToCamera)
{
    float FogFactor = exp( -fDistToCamera * g_FogScale );
    return Color.rgb * FogFactor + (1-FogFactor)*g_FogColor.rgb;
}
