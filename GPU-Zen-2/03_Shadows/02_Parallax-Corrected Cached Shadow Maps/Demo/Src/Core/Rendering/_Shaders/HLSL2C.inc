#if !defined(__HLSL2C_H) && defined(__cplusplus)
#define __HLSL2C_H

struct ConstantBuffer_b0 { static const unsigned BANK = 0; };
struct ConstantBuffer_b1 { static const unsigned BANK = 1; };
struct ConstantBuffer_b2 { static const unsigned BANK = 2; };
struct ConstantBuffer_b3 { static const unsigned BANK = 3; };

#define uint        UINT
#define float4x4    Mat4x4
#define float4      Vec4
#define int4        Vec4i
#define float3      UVec3
#define float2      UVec2
#define cbuffer     struct
#define register(a) ConstantBuffer_##a

#endif

