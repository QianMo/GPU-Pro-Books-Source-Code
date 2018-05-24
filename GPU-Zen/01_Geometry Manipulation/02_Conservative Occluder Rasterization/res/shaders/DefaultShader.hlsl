struct VSInput
{
    float3 position : POSITION0;
    float3 normal   : NORMAL0;
    float2 uv       : UV0;
    float3 tangent  : TANGENT0;
    float3 binormal : BINORMAL0;
};

struct PSInput
{
    float4 position       : SV_POSITION;
    float3 world_position : POSITION0;
    float3 normal         : NORMAL0;
    float2 uv             : UV0;
    float3 tangent        : TANGENT0;
    float3 binormal       : BINORMAL0;
};

struct Material
{
    uint diffuse;
    uint normal;
};

cbuffer ObjectConstants : register( b0 )
{
    uint g_InstanceOffset;
    uint g_MaterialIndex;
};

cbuffer FrameConstants : register( b1 )
{
    float4x4 g_ViewProjection;
    float4   g_CameraPosition;
    float4   g_LightDirection;
};

StructuredBuffer< uint > g_InstanceIndexMappingsBuffer : register( t0 );
StructuredBuffer< float4x4 > g_WorldMatrixBuffer : register( t1 );
StructuredBuffer< Material > g_MaterialBuffer : register( t2 );
Texture2D g_Textures[ TEXTURE_COUNT ] : register( t3 );
SamplerState g_Sampler : register( s0 );

PSInput VSMain( VSInput input, uint instance_id : SV_InstanceID )
{
    uint instance_index = g_InstanceIndexMappingsBuffer[ g_InstanceOffset + instance_id ];
    float4x4 world = g_WorldMatrixBuffer[ instance_index ];

    PSInput output = ( PSInput )0;
    output.position = mul( float4( input.position, 1.0 ), world );
    output.world_position = output.position.xyz;
    output.position = mul( output.position, g_ViewProjection );
    output.normal = mul( input.normal, ( float3x3 )world );
    output.tangent = mul( input.tangent, ( float3x3 )world );
    output.binormal = mul( input.binormal, ( float3x3 )world );
    output.uv = input.uv;
    return output;
}

float4 PSMain( PSInput input ) : SV_Target
{
    Material material = g_MaterialBuffer[ g_MaterialIndex ];

    float4 diffuse_sample = g_Textures[ NonUniformResourceIndex( material.diffuse ) ].Sample( g_Sampler, input.uv );
    float4 normal_sample = g_Textures[ NonUniformResourceIndex( material.normal ) ].Sample( g_Sampler, input.uv );

    clip( diffuse_sample.a < 0.1 ? -1 : 1 );

    float3x3 world_to_tangent = float3x3( input.tangent, input.binormal, input.normal );
    float3 normal = normalize( mul( normal_sample.xyz, world_to_tangent ) );
    
    float  n_dot_l = saturate( dot( normal, -g_LightDirection.xyz ) );
    return float4( ( n_dot_l + 0.4 ) * diffuse_sample.xyz, 1.0 );
}