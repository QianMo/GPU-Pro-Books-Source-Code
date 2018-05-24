//-----------------------------------------------------------------------------
// File: IntelExtensions.hlsl
//
// Desc: HLSL extensions available on Intel processor graphic platforms
// 
// Copyright (c) Intel Corporation 2012-2013. All rights reserved.
//-----------------------------------------------------------------------------

#ifndef PS_ORDERING_AVAILABLE
#   define PS_ORDERING_AVAILABLE 0
#endif

//
// Intel extension buffer structure, generic interface for 
//   0-operand extensions (e.g. sync opcodes)  
//   1-operand unary operations (e.g. render target writes) 
//   2-operand binary operations (future extensions) 
//   3-operand ternary operations (future extensions)  
//
struct IntelExtensionStruct
{
    uint   opcode; 	// opcode to execute
    uint   rid;		// resource ID
    uint   sid;		// sampler ID

    float4 src0f;	// float source operand  0
    float4 src1f;	// float source operand  0
    float4 src2f;	// float source operand  0
    float4 dst0f;	// float destination operand 

    uint4  src0u;
    uint4  src1u;
    uint4  src2u;
    uint4  dst0u;

    float  pad[180]; // total lenght 860
};

//
// Define extension opcodes (no enums in HLSL)
//
#define INTEL_EXT_BEGIN_PIXEL_ORDERING			1



// Define RW buffer for Intel extensions.
// Application should bind null resource, operations will be ignored.
RWStructuredBuffer<IntelExtensionStruct> g_IntelExt : register( u7 );


//
// Initialize Intel HSLS Extensions
// This method should be called before any other extension function 
// 
void IntelExt_Init()
{
#if PS_ORDERING_AVAILABLE
    uint4 init = { 0x63746e69, 0x6c736c68, 0x6e747865, 0x0 }; // intc hlsl extn 
    g_IntelExt[0].src0u = init; 
#endif
}


//
// Start pixel ordering on specific read-write resource
// 
void IntelExt_BeginPixelShaderOrderingOnUAV( uint resourceId )
{
#if PS_ORDERING_AVAILABLE
    DeviceMemoryBarrier();	
    uint opcode = g_IntelExt.IncrementCounter();
    g_IntelExt[opcode].opcode = INTEL_EXT_BEGIN_PIXEL_ORDERING;
    g_IntelExt[opcode].rid = resourceId;
#endif
}

//
// Start pixel ordering on all read-write resources
// 
void IntelExt_BeginPixelShaderOrdering( )
{
#if PS_ORDERING_AVAILABLE
    DeviceMemoryBarrier();	
    uint opcode = g_IntelExt.IncrementCounter();
    g_IntelExt[opcode].opcode = INTEL_EXT_BEGIN_PIXEL_ORDERING;
    g_IntelExt[opcode].rid = 0xFFFF;
#endif
}
