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
// File: FilterKernel.hlsl
//
// Defines the compute kernel for a separable filter. It calls the macros defined by the 
// user supplied HLSL such as GaussianFilter.hlsl and BilateralFilter.hlsl
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
// Samples from inputs defined by the SampleFromInput macro
//--------------------------------------------------------------------------------------
RAWDataItem Sample( int2 i2Position, float2 f2Offset )
{
    RAWDataItem RDI;
    float2 f2SamplePosition = float2( i2Position ) + float2( 0.5f, 0.5f );

    #if ( USE_APPROXIMATE_FILTER == 1 )

        f2SamplePosition += f2Offset;
            
    #endif

    f2SamplePosition *= g_f4OutputSize.zw;
    SAMPLE_FROM_INPUT( g_LinearClampSampler, f2SamplePosition, RDI )
          
    return RDI;
}

#if ( USE_COMPUTE_SHADER == 1 )

    //--------------------------------------------------------------------------------------
    // Macro for caching LDS reads, this has the effect of drastically reducing reads from the 
    // LDS by up to 4x
    //--------------------------------------------------------------------------------------
    #define CACHE_LDS_READS( _iIteration, _iLineOffset, _iPixelOffset, _RDI ) \
        /* Trickle LDS values down within the GPRs*/ \
        [unroll] for( iPixel = 0; iPixel < PIXELS_PER_THREAD - STEP_SIZE; ++iPixel ) { \
            _RDI[iPixel] = _RDI[iPixel + STEP_SIZE]; } \
        /* Load new LDS value(s) */ \
        [unroll] for( iPixel = 0; iPixel < STEP_SIZE; ++iPixel ) { \
            READ_FROM_LDS( _iLineOffset, ( _iPixelOffset + _iIteration + iPixel ), _RDI[(PIXELS_PER_THREAD - STEP_SIZE + iPixel)] ) }


    //--------------------------------------------------------------------------------------
    // Defines the filter kernel logic. User supplies macro's for custom filter
    //--------------------------------------------------------------------------------------
    void ComputeFilterKernel( int iPixelOffset, int iLineOffset, int2 i2Center, int2 i2Inc )
    {
        CS_Output O = (CS_Output)0;
        KernelData KD[PIXELS_PER_THREAD];
        int iPixel, iIteration;
        RAWDataItem RDI[PIXELS_PER_THREAD];
       
        #if ( USE_APPROXIMATE_FILTER == 1 )

            // Read the kernel center values in directly from the input surface(s), as the LDS
            // values are pre-filtered, and therefore do not represent the kernel center
            [unroll] 
            for( iPixel = 0; iPixel < PIXELS_PER_THREAD; ++iPixel )  
            {
                float2 f2SamplePosition = ( float2( i2Center + ( iPixel * i2Inc ) ) + float2( 0.5f, 0.5f ) ) * g_f4OutputSize.zw;
                SAMPLE_FROM_INPUT( g_PointSampler, f2SamplePosition, RDI[iPixel] )
            }

        #else

            // Read the kernel center values in from the LDS
            [unroll] 
            for( iPixel = 0; iPixel < PIXELS_PER_THREAD; ++iPixel ) 
            {
                READ_FROM_LDS( iLineOffset, ( iPixelOffset + KERNEL_RADIUS + iPixel ), RDI[iPixel] )
            }

        #endif
        

        // Macro defines what happens at the kernel center
        KERNEL_CENTER( KD, iPixel, PIXELS_PER_THREAD, O, RDI )
            
        // Prime the GPRs for the first half of the kernel
        [unroll]
        for( iPixel = 0; iPixel < PIXELS_PER_THREAD; ++iPixel )
        {
            READ_FROM_LDS( iLineOffset, ( iPixelOffset + iPixel ), RDI[iPixel] )
        }

        // Increment the LDS offset by PIXELS_PER_THREAD
        iPixelOffset += PIXELS_PER_THREAD;

        // First half of the kernel
        [unroll]
        for( iIteration = 0; iIteration < KERNEL_RADIUS; iIteration += STEP_SIZE )
        {
            // Macro defines what happens for each kernel iteration  
            KERNEL_ITERATION( iIteration, KD, iPixel, PIXELS_PER_THREAD, O, RDI )

            // Macro to cache LDS reads in GPRs
            CACHE_LDS_READS( iIteration, iLineOffset, iPixelOffset, RDI ) 
        }

        // Prime the GPRs for the second half of the kernel
        [unroll]
        for( iPixel = 0; iPixel < PIXELS_PER_THREAD; ++iPixel )
        {
            READ_FROM_LDS( iLineOffset, ( iPixelOffset - PIXELS_PER_THREAD + iIteration + 1 + iPixel ), RDI[iPixel] )
        }
        
        // Second half of the kernel
        [unroll]
        for( iIteration = KERNEL_RADIUS + 1; iIteration < KERNEL_DIAMETER; iIteration += STEP_SIZE )
        {
            // Macro defines what happens for each kernel iteration  
            KERNEL_ITERATION( iIteration, KD, iPixel, PIXELS_PER_THREAD, O, RDI )

            // Macro to cache LDS reads in GPRs
            CACHE_LDS_READS( iIteration, iLineOffset, iPixelOffset, RDI )
        }
        
        // Macros define final weighting and output 
        KERNEL_FINAL_WEIGHT( KD, iPixel, PIXELS_PER_THREAD, O )
        KERNEL_OUTPUT( i2Center, i2Inc, iPixel, PIXELS_PER_THREAD, O, KD )
    }

#endif


//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
