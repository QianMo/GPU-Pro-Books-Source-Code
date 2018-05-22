

cbuffer cbScatteringParams
{
    // Atmospheric light scattering constants
    float4 g_vTotalRayleighBeta = float4(4.1729738e-005f, 7.0566675e-005f, 0.00014632706f, 6.0000000f);
    float4 g_vAngularRayleighBeta = float4(2.4905603e-006f, 4.2116380e-006f, 8.7332528e-006f, 6.0000000f);
    float4 g_vTotalMieBeta = float4(1.8011092e-007f, 2.3224241e-007f, 3.2968325e-007f, 0.00030000001f);
    float4 g_vAngularMieBeta = float4(4.1875015e-008f, 5.4454276e-008f, 7.8414160e-008f, 0.00030000001f);
    float4 g_vHG_g = float4(0.0099750161, 1.9900250f, -1.9900000f, 1.0000000f); // = float4(1 - HG_g*HG_g, 1 + HG_g*HG_g, -2*HG_g, 1.0);

#define INSCATTERING_MULTIPLIER 27.f/3.f   ///< Light scattering constant - Inscattering multiplier    
//#define g_Reflectance 0.1f
//const float4 g_vSoilReflectivity = float4(0.138f, 0.113f, 0.08f, 1.0f) * g_Reflectance;
    float4 g_vSoilReflectivity = float4(1.f, 1.f, 1.f, 1.f);
    
    float g_DistanceScaler = 0.05;
}

cbuffer cbFrameParams 
{
    matrix g_mWorldViewProjInvMatr;
    float3 g_vCameraPos;
}

cbuffer cbLighrParams
{
    // WARNING: these parameters are duplicated in Terrain.fx
    // These parameters are set from host side
    float4 g_vDirectionOnSun = {0.f, 0.769666f, 0.638446f, 1.f}; ///< Direction on sun
    float4 g_vSunColorAndIntensityAtGround = {0.640682f, 0.591593f, 0.489432f, 3.f}; ///< Sun color
    float4 g_vAmbientLight = {0.191534f, 0.127689f, 0.25f, 0.f}; ///< Ambient light
}

SamplerState samPointClamp
{
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};

Texture2D<float3> g_tex2DColorBuffer;
Texture2D<float> g_tex2DDepthBuffer;

struct GenerateQuadVS_OUTPUT
{
    float4 m_ScreenPos_PS : SV_POSITION;
    float2 m_ScrUV : TEXCOORD0; 
};

GenerateQuadVS_OUTPUT GenerateQuadVS( in uint VertexId : SV_VertexID )
{
    float4 DstTextureMinMaxUV = float4(-1,1,1,-1);
    float4 SrcElevAreaMinMaxUV = float4(0,0,1,1);
    
    GenerateQuadVS_OUTPUT Verts[4] = 
    {
        {float4(DstTextureMinMaxUV.xy, 0.5, 1.0), SrcElevAreaMinMaxUV.xy}, 
        {float4(DstTextureMinMaxUV.xw, 0.5, 1.0), SrcElevAreaMinMaxUV.xw},
        {float4(DstTextureMinMaxUV.zy, 0.5, 1.0), SrcElevAreaMinMaxUV.zy},
        {float4(DstTextureMinMaxUV.zw, 0.5, 1.0), SrcElevAreaMinMaxUV.zw}
    };

    return Verts[VertexId];
}

float3 ReconstructWorldSpacePos(in float2 DepthBufferTextureUV)
{
    float4 PositionPS;
    PositionPS.x =  2.0 * DepthBufferTextureUV.x - 1.0;
    PositionPS.y = -2.0 * DepthBufferTextureUV.y + 1.0;
    float Depth = g_tex2DDepthBuffer.SampleLevel(samPointClamp, DepthBufferTextureUV, 0);
    PositionPS.z = Depth;
    PositionPS.w = 1.0;

    float4 ReconstructedPosWS;
    ReconstructedPosWS = mul( PositionPS, g_mWorldViewProjInvMatr );
    ReconstructedPosWS /= ReconstructedPosWS.w;
    return ReconstructedPosWS.xyz;
}




void ComputeAtmosphericLighting(
    in float3 in_eyeVector,
    in float in_Dist,
    out float3 out_vExtinction,
    out float3 out_vInScatteringColor)
{
    out_vExtinction = exp( -(g_vTotalRayleighBeta.rgb +  g_vTotalMieBeta.rgb) * in_Dist * g_DistanceScaler );   
    

    //    sun
    //      \
    //       \
    //    ----\------eye
    //         \theta 
    //          \
    //    
    // compute cosine of theta angle
    float cosTheta = dot(in_eyeVector, g_vDirectionOnSun.xyz);

    // Compute Rayleigh scattering Phase Function
    // According to formula for the Rayleigh Scattering phase function presented in the 
    // "Rendering Outdoor Light Scattering in Real Time" by Hoffman and Preetham (see p.36 and p.51), 
    // BethaR(Theta) is calculated as follows:
    // 3/(16PI) * BethaR * (1+cos^2(theta))
    // g_vAngularRayleighBeta == (3*PI/16) * g_vTotalRayleighBeta, hence:
    float3 RayleighScatteringPhaseFunc = g_vAngularRayleighBeta.rgb * (1.0 + cosTheta*cosTheta);

    // Compute Henyey-Greenstein approximation of the Mie scattering Phase Function
    // According to formula for the Mie Scattering phase function presented in the 
    // "Rendering Outdoor Light Scattering in Real Time" by Hoffman and Preetham 
    // (see p.38 and p.51),  BethaR(Theta) is calculated as follows:
    // 1/(4PI) * BethaM * (1-g^2)/(1+g^2-2g*cos(theta))^(3/2)
    // const float4 g_vHG_g = float4(1 - g*g, 1 + g*g, -2*g, 1);
    float HGTemp = rsqrt( g_vHG_g.y + g_vHG_g.z*cosTheta);
    // g_vAngularMieBeta is calculated according to formula presented in "A practical Analytic 
    // Model for Daylight" by Preetham & Hoffman (see p.23)
    float3 fMieScatteringPhaseFunc_HGApprox = g_vAngularMieBeta.rgb * g_vHG_g.x * (HGTemp*HGTemp*HGTemp);
    
 
    float3 vInScattering;

    vInScattering = (1 - out_vExtinction);

    vInScattering.rgb *= (RayleighScatteringPhaseFunc.rgb + fMieScatteringPhaseFunc_HGApprox.rgb) / (g_vTotalRayleighBeta.rgb +  g_vTotalMieBeta.rgb);
    vInScattering.rgb *= INSCATTERING_MULTIPLIER;

    out_vInScatteringColor.rgb = vInScattering.rgb * g_vSunColorAndIntensityAtGround.rgb * g_vSunColorAndIntensityAtGround.w;
}

float4 PerformPostProcessPS(GenerateQuadVS_OUTPUT In) : SV_TARGET
{
    float3 ReconstructedPosWS = ReconstructWorldSpacePos(In.m_ScrUV);
    
    float3 EyeVector = ReconstructedPosWS - g_vCameraPos;
    float DistanceToCamera = length(EyeVector);
    EyeVector = EyeVector / DistanceToCamera;

    float TraceLength = DistanceToCamera;
    //float g_MaxTracingDistance = 0000 / g_DistanceScaler;
    //TraceLength = min(TraceLength, g_MaxTracingDistance);

    float3 SceneColor = g_tex2DColorBuffer.Sample(samPointClamp, In.m_ScrUV);
    SceneColor = pow( max(SceneColor,0), 1/2.2);

    float3 vExtinction, vInScatteringColor;
    
    ComputeAtmosphericLighting(EyeVector, TraceLength, vExtinction, vInScatteringColor);
    float3 vAttenuatedColor = SceneColor.rgb * /* g_vSunColorAndIntensityAtGround.w * */ vExtinction.rgb * g_vSoilReflectivity.rgb;
	
    return float4(vAttenuatedColor + vInScatteringColor,1);
}



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

// Blend state disabling blending
BlendState NoBlending
{
    BlendEnable[0] = FALSE;
    BlendEnable[1] = FALSE;
    BlendEnable[2] = FALSE;
};

technique11 PerformPostProcessing_FeatureLevel10
{
    pass
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_4_0, GenerateQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PerformPostProcessPS() ) );
    }
}
