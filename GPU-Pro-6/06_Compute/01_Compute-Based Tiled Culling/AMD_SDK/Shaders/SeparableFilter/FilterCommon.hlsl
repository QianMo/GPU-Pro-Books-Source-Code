//
// Copyright 2014 ADVANCED MICRO DEVICES, INC.  All Rights Reserved.
//
// AMD is granting you permission to use this software and documentation (if
// any) (collectively, the “Materials”) pursuant to the terms and conditions
// of the Software License Agreement included with the Materials.  If you do
// not have a copy of the Software License Agreement, contact your AMD
// representative for a copy.
// You agree that you will not reverse engineer or decompile the Materials,
// in whole or in part, except as allowed by applicable law.
//
// WARRANTY DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND.  AMD DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE, NON-INFRINGEMENT, THAT THE SOFTWARE
// WILL RUN UNINTERRUPTED OR ERROR-FREE OR WARRANTIES ARISING FROM CUSTOM OF
// TRADE OR COURSE OF USAGE.  THE ENTIRE RISK ASSOCIATED WITH THE USE OF THE
// SOFTWARE IS ASSUMED BY YOU.
// Some jurisdictions do not allow the exclusion of implied warranties, so
// the above exclusion may not apply to You. 
// 
// LIMITATION OF LIABILITY AND INDEMNIFICATION:  AMD AND ITS LICENSORS WILL
// NOT, UNDER ANY CIRCUMSTANCES BE LIABLE TO YOU FOR ANY PUNITIVE, DIRECT,
// INCIDENTAL, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM USE OF
// THE SOFTWARE OR THIS AGREEMENT EVEN IF AMD AND ITS LICENSORS HAVE BEEN
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.  
// In no event shall AMD's total liability to You for all damages, losses,
// and causes of action (whether in contract, tort (including negligence) or
// otherwise) exceed the amount of $100 USD.  You agree to defend, indemnify
// and hold harmless AMD and its licensors, and any of their directors,
// officers, employees, affiliates or agents from and against any and all
// loss, damage, liability and other expenses (including reasonable attorneys'
// fees), resulting from Your use of the Software or violation of the terms and
// conditions of this Agreement.  
//
// U.S. GOVERNMENT RESTRICTED RIGHTS: The Materials are provided with "RESTRICTED
// RIGHTS." Use, duplication, or disclosure by the Government is subject to the
// restrictions as set forth in FAR 52.227-14 and DFAR252.227-7013, et seq., or
// its successor.  Use of the Materials by the Government constitutes
// acknowledgement of AMD's proprietary rights in them.
// 
// EXPORT RESTRICTIONS: The Materials may be subject to export restrictions as
// stated in the Software License Agreement.
//

//--------------------------------------------------------------------------------------
// File: FilterCommon.hlsl
//
// Common defines for separable filtering kernels
//--------------------------------------------------------------------------------------


// Defines passed in at compile time
//#define LDS_PRECISION               ( 8 , 16, or 32 )
//#define USE_APPROXIMATE_FILTER      ( 0, 1 )      
//#define USE_COMPUTE_SHADER           
//#define KERNEL_RADIUS               ( 16 )   // Must be an even number


// Defines that control the CS logic of the kernel 
#define KERNEL_DIAMETER             ( KERNEL_RADIUS * 2 + 1 )  
#define RUN_LINES                   ( 2 )   // Needs to match g_uRunLines in SeparableFilter11.cpp  
#define RUN_SIZE                    ( 128 ) // Needs to match g_uRunSize in SeparableFilter11.cpp  
#define KERNEL_DIAMETER_MINUS_ONE	( KERNEL_DIAMETER - 1 )
#define RUN_SIZE_PLUS_KERNEL	    ( RUN_SIZE + KERNEL_DIAMETER_MINUS_ONE )
#define PIXELS_PER_THREAD           ( 4 )  
#define NUM_THREADS                 ( RUN_SIZE / PIXELS_PER_THREAD )
#define SAMPLES_PER_THREAD          ( RUN_SIZE_PLUS_KERNEL / NUM_THREADS )
#define EXTRA_SAMPLES               ( RUN_SIZE_PLUS_KERNEL - ( NUM_THREADS * SAMPLES_PER_THREAD ) )


// The samplers
SamplerState g_PointSampler : register (s0);
SamplerState g_LinearClampSampler : register (s1);


// Adjusts the sampling step size if using approximate filtering
#if ( USE_APPROXIMATE_FILTER == 1 )

    #define STEP_SIZE ( 2 )

#else

    #define STEP_SIZE ( 1 )

#endif


// Constant buffer used by the CS & PS
cbuffer cbSF : register( b0 )
{
    float4    g_f4OutputSize;   // x = Width, y = Height, z = Inv Width, w = Inv Height
}


// Input structure used by the PS
struct PS_RenderQuadInput
{
    float4 f4Position : SV_POSITION;              
    float2 f2TexCoord : TEXCOORD0;
};

#if ( REQUIRE_HDR == 1 )

    //--------------------------------------------------------------------------------------
    // Packs a float2 to a unit
    //--------------------------------------------------------------------------------------
    uint Float2ToUint( float2 f2Value )
    {
        return ( f32tof16( f2Value.x ) ) + ( f32tof16( f2Value.y ) << 16 );
    }


    //--------------------------------------------------------------------------------------
    // Unpacks a uint to a float2
    //--------------------------------------------------------------------------------------
    float2 UintToFloat2( uint uValue )
    {
        return float2( f16tof32( uValue ), f16tof32( uValue >> 16 ) );
    }

#else

    //--------------------------------------------------------------------------------------
    // Packs a float2 to a unit
    //--------------------------------------------------------------------------------------
    uint Float2ToUint( float2 f2Value )
    {
        return ( ( ( (uint)( f2Value.y * 65535.0f ) ) << 16 ) | 
                 ( (uint)( f2Value.x * 65535.0f ) ) );
    }


    //--------------------------------------------------------------------------------------
    // Unpacks a uint to a float2
    //--------------------------------------------------------------------------------------
    float2 UintToFloat2( uint uValue )
    {
        return float2(  ( uValue & 0x0000FFFF ) / 65535.0f, 
                      ( ( uValue & 0xFFFF0000 ) >> 16 ) / 65535.0f );
    }

#endif

//--------------------------------------------------------------------------------------
// Packs a float4 to a unit
//--------------------------------------------------------------------------------------
uint Float4ToUint( float4 f4Value )
{
    return (    ( ( (uint)( f4Value.w * 255.0f ) ) << 24 ) | 
                ( ( (uint)( f4Value.z * 255.0f ) ) << 16 ) | 
                ( ( (uint)( f4Value.y * 255.0f ) ) << 8 ) | 
                ( (uint)( f4Value.x * 255.0f ) ) );
}


//--------------------------------------------------------------------------------------
// Unpacks a uint to a float4
//--------------------------------------------------------------------------------------
float4 UintToFloat4( uint uValue )
{
    return float4(  ( uValue & 0x000000FF ) / 255.0f, 
                    ( ( uValue & 0x0000FF00 ) >> 8 ) / 255.0f,
                    ( ( uValue & 0x00FF0000 ) >> 16 ) / 255.0f,
                    ( ( uValue & 0xFF000000 ) >> 24 ) / 255.0f );
}


//--------------------------------------------------------------------------------------
// Packs a float3 to a unit
//--------------------------------------------------------------------------------------
uint Float3ToUint( float3 f3Value )
{
    return (    ( ( (uint)( f3Value.z * 255.0f ) ) << 16 ) | 
                ( ( (uint)( f3Value.y * 255.0f ) ) << 8 ) | 
                ( (uint)( f3Value.x * 255.0f ) ) );
}


//--------------------------------------------------------------------------------------
// Unpacks a uint to a float3
//--------------------------------------------------------------------------------------
float3 UintToFloat3( uint uValue )
{
    return float3(  ( uValue & 0x000000FF ) / 255.0f, 
                    ( ( uValue & 0x0000FF00 ) >> 8 ) / 255.0f,
                    ( ( uValue & 0x00FF0000 ) >> 16 ) / 255.0f );
}


//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
