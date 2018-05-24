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
    uint ambient;
    uint diffuse_grass;
    uint diffuse_rock;
    uint diffuse_wall;
    uint mask_rock;
    uint mask_wall;
    uint normal_grass;
    uint normal_rock;
    uint normal_wall;
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

    float4 ambient_sample = g_Textures[ NonUniformResourceIndex( material.ambient ) ].Sample( g_Sampler, input.uv );
    float4 diffuse_grass_sample = g_Textures[ NonUniformResourceIndex( material.diffuse_grass ) ].Sample( g_Sampler, input.uv * float2( 30.0, 30.0 ) );
    float4 diffuse_rock_sample = g_Textures[ NonUniformResourceIndex( material.diffuse_rock ) ].Sample( g_Sampler, input.uv * float2( 10.0, 10.0 ) );
    float4 diffuse_wall_sample = g_Textures[ NonUniformResourceIndex( material.diffuse_wall ) ].Sample( g_Sampler, input.uv * float2( 60.0, 60.0 ) );
    float4 mask_rock_sample = g_Textures[ NonUniformResourceIndex( material.mask_rock ) ].Sample( g_Sampler, input.uv );
    float4 mask_wall_sample = g_Textures[ NonUniformResourceIndex( material.mask_wall ) ].Sample( g_Sampler, input.uv );
    float4 normal_grass_sample = g_Textures[ NonUniformResourceIndex( material.normal_grass ) ].Sample( g_Sampler, input.uv * float2( 30.0, 30.0 ) );
    float4 normal_rock_sample = g_Textures[ NonUniformResourceIndex( material.normal_rock ) ].Sample( g_Sampler, input.uv * float2( 10.0, 10.0 ) );
    float4 normal_wall_sample = g_Textures[ NonUniformResourceIndex( material.normal_wall ) ].Sample( g_Sampler, input.uv * float2( 60.0, 60.0 ) );

    float4 diffuse_lerp_0 = lerp( diffuse_grass_sample, diffuse_rock_sample, mask_rock_sample );
    float4 diffuse_lerp_1 = lerp( diffuse_lerp_0, diffuse_wall_sample, mask_wall_sample );
    float4 diffuse_lerp_2 = lerp( diffuse_lerp_1, ambient_sample, ambient_sample.a * 1.5 );

    float4 normal_lerp_0 = lerp( normal_grass_sample, normal_rock_sample, mask_rock_sample );
    float4 normal_lerp_1 = lerp( normal_lerp_0, normal_wall_sample, mask_wall_sample );

    float3x3 world_to_tangent = float3x3( input.tangent, input.binormal, input.normal );
    float3 normal = normalize( mul( normal_lerp_1.xyz, world_to_tangent ) );

    float n_dot_l = saturate( dot( normal, -g_LightDirection.xyz ) );
    return float4( ( n_dot_l + 0.6 ) * diffuse_lerp_2.xyz + ambient_sample.xyz, 1.0 );
}