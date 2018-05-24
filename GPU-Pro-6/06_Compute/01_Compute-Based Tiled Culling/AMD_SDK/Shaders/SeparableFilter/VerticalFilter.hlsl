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
// File: VerticalFilter.hlsl
//
// Implements the vertical pass of the kernel.
//--------------------------------------------------------------------------------------


#if ( USE_COMPUTE_SHADER == 1 )

    //--------------------------------------------------------------------------------------
    // Compute shader implementing the vertical pass of a separable filter
    //--------------------------------------------------------------------------------------
    [numthreads( NUM_THREADS, RUN_LINES, 1 )]
    void CSFilterY( uint3 Gid : SV_GroupID, uint3 GTid : SV_GroupThreadID )
    {
        // Sampling and line offsets from group thread IDs
        int iSampleOffset = GTid.x * SAMPLES_PER_THREAD;
        int iLineOffset = GTid.y;
        
        // Group and pixel coords from group IDs
        int2 i2GroupCoord = int2( ( Gid.x * RUN_LINES ), ( Gid.y * RUN_SIZE ) - KERNEL_RADIUS );
        int2 i2Coord = int2( i2GroupCoord.x, i2GroupCoord.y + iSampleOffset );
        
        // Sample and store to LDS
        [unroll]
        for( int i = 0; i < SAMPLES_PER_THREAD; ++i )
        {
            WRITE_TO_LDS( Sample( i2Coord + int2( GTid.y, i ), float2( 0.0f, 0.5f ) ), iLineOffset, iSampleOffset + i )
        }
                       
        // Optionally load some extra texels as required by the exact kernel size 
        if( GTid.x < EXTRA_SAMPLES )
        {
            WRITE_TO_LDS( Sample( i2GroupCoord + int2( GTid.y, RUN_SIZE_PLUS_KERNEL - 1 - GTid.x ), float2( 0.0f, 0.5f ) ), iLineOffset, RUN_SIZE_PLUS_KERNEL - 1 - GTid.x )
        }
        
        // Sync threads
        GroupMemoryBarrierWithGroupSync();

        // Adjust pixel offset for computing at PIXELS_PER_THREAD
        int iPixelOffset = GTid.x * PIXELS_PER_THREAD;
        i2Coord = int2( i2GroupCoord.x, i2GroupCoord.y + iPixelOffset );

        // Since we start with the first thread position, we need to increment the coord by KERNEL_RADIUS 
        i2Coord.y += KERNEL_RADIUS;

        // Ensure we don't compute pixels off screen
        if( i2Coord.y < g_f4OutputSize.y  )
        {
            int2 i2Center = i2Coord + int2( GTid.y, 0 );
            int2 i2Inc = int2( 0, 1 );
            
            // Compute the filter kernel using LDS values
            ComputeFilterKernel( iPixelOffset, iLineOffset, i2Center, i2Inc );
        }
    }

#else // USE_COMPUTE_SHADER

    //--------------------------------------------------------------------------------------
    // Pixel shader implementing the vertical pass of a separable filter
    //--------------------------------------------------------------------------------------
    PS_Output PSFilterY( PS_RenderQuadInput I ) : SV_TARGET
    {   
        PS_Output O = (PS_Output)0;
        RAWDataItem RDI[1];
        int iPixel, iIteration;
        KernelData KD[1];
        
        // Load the center sample(s)
        int2 i2KernelCenter = int2( g_f4OutputSize.xy * I.f2TexCoord );
        RDI[0] = Sample( int2( i2KernelCenter.x, i2KernelCenter.y ), float2( 0.0f, 0.0f ) );
        
        // Macro defines what happens at the kernel center
        KERNEL_CENTER( KD, iPixel, 1, O, RDI )

        i2KernelCenter.y -= KERNEL_RADIUS;

        // First half of the kernel
        [unroll]
        for( iIteration = 0; iIteration < KERNEL_RADIUS; iIteration += STEP_SIZE )
        {
            // Load the sample(s) for this iteration
            RDI[0] = Sample( int2( i2KernelCenter.x, i2KernelCenter.y + iIteration ), float2( 0.0f, 0.5f ) );

            // Macro defines what happens for each kernel iteration
            KERNEL_ITERATION( iIteration, KD, iPixel, 1, O, RDI )
        }

        // Second half of the kernel
        [unroll]
        for( iIteration = KERNEL_RADIUS + 1; iIteration < KERNEL_DIAMETER; iIteration += STEP_SIZE )
        {
            // Load the sample(s) for this iteration
            RDI[0] = Sample( int2( i2KernelCenter.x, i2KernelCenter.y + iIteration ), float2( 0.0f, 0.5f ) );

            // Macro defines what happens for each kernel iteration
            KERNEL_ITERATION( iIteration, KD, iPixel, 1, O, RDI )
        }

        // Macros define final weighting
        KERNEL_FINAL_WEIGHT( KD, iPixel, 1, O )

        return O;
    }

#endif // USE_COMPUTE_SHADER


//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
