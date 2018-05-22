// -*- c++ -*-
/**
 \file VVDoF_blur.glsl

 Computes the near field blur.  This is included by both the horizontal
 (first) and vertical (second) passes.

 The shader produces two outputs:

 * blurResult

   A buffer that is the scene blurred with a spatially varying Gaussian
   kernel that attempts to make the point-spread function at each pixel
   equal to its circle of confusion radius.

   blurResult.rgb = color
   blurResult.a   = normalized coc radius

 * nearResult

   A buffer that contains only near field, blurred with premultiplied
   alpha.  The point spread function is a fixed RADIUS in this region.

   nearResult.rgb = coverage * color
   nearResult.a   = coverage

 */
#include "compatability.glsl"
#line 30

/** Maximum blur radius for any point in the scene, in pixels.  Used to
    reconstruct the CoC radius from the normalized CoC radius. */
uniform int         maxCoCRadiusPixels;

/** Source image in RGB, normalized CoC in A. */
uniform sampler2D	blurSourceBuffer;

uniform int         nearBlurRadiusPixels;
uniform float       invNearBlurRadiusPixels;


#if HORIZONTAL
	const int2 direction = int2(1, 0);
#else
	const int2 direction = int2(0, 1);

	/** For the second pass, the output of the previous near-field blur pass. */
	uniform sampler2D  nearSourceBuffer;
#endif

#if __VERSION__ < 130
#	define nearResult gl_FragData[0]
#	define blurResult gl_FragData[1]
#else
	out layout(location = 0) float4 nearResult;
	out layout(location = 1) float4 blurResult;
#endif

    
bool inNearField(float radiusPixels) {
    return radiusPixels > 0.25;
}

void main() {

    const int GAUSSIAN_TAPS = 6;
    float gaussian[GAUSSIAN_TAPS + 1];
    
    // 11 x 11 separated Gaussian weights.  This does not dictate the 
    // blur kernel for depth of field; it is scaled to the actual
    // kernel at each pixel.
    gaussian[6] = 0.00000000000000;  // Weight applied to outside-radius values
    gaussian[5] = 0.04153263993208;
    gaussian[4] = 0.06352050813141;
    gaussian[3] = 0.08822292796029;
    gaussian[2] = 0.11143948794984;
    gaussian[1] = 0.12815541114232;
    gaussian[0] = 0.13425804976814;
    
    // Accumulate the blurry image color
    blurResult.rgb  = float3(0.0f);
    float blurWeightSum = 0.0f;
    
    // Accumulate the near-field color and coverage
    nearResult = float4(0.0f);
    float nearWeightSum = 0.000f;
    
    // Location of the central filter tap (i.e., "this" pixel's location)
    // Account for the scaling down by 50% during blur
    int2 A = int2(gl_FragCoord.xy) * (direction + ivec2(1));
    
    float packedA = texelFetch(blurSourceBuffer, A, 0).a;
    float r_A = (packedA * 2.0 - 1.0) * maxCoCRadiusPixels;
    
    // Map r_A << 0 to 0, r_A >> 0 to 1
    float nearFieldness_A = saturate(r_A * 4.0);

    
    for (int delta = -maxCoCRadiusPixels; delta <= maxCoCRadiusPixels; ++delta)	{
        // Tap location near A
        int2   B = A + (direction * delta);

        // Packed values
        float4 blurInput = texelFetch(blurSourceBuffer, clamp(B, int2(0), textureSize(blurSourceBuffer, 0) - int2(1)), 0);

        // Signed kernel radius at this tap, in pixels
        float r_B = (blurInput.a * 2.0 - 1.0) * float(maxCoCRadiusPixels);
        
        /////////////////////////////////////////////////////////////////////////////////////////////
        // Compute blurry buffer

        float weight = 0.0;
        
        float wNormal  = 
            // Only consider mid- or background pixels (allows inpainting of the near-field)
            float(! inNearField(r_B)) * 
            
            // Only blur B over A if B is closer to the viewer (allow 0.5 pixels of slop, and smooth the transition)
            saturate(abs(r_A) - abs(r_B) + 1.5) *
            
            // Stretch the Gaussian extent to the radius at pixel B.
            gaussian[clamp(int(float(abs(delta) * (GAUSSIAN_TAPS - 1)) / (0.001 + abs(r_B * 0.5))), 0, GAUSSIAN_TAPS)];

        weight = lerp(wNormal, 1.0, nearFieldness_A);

        // far + mid-field output 
        blurWeightSum  += weight;
        blurResult.rgb += blurInput.rgb * weight;
        
        ///////////////////////////////////////////////////////////////////////////////////////////////
        // Compute near-field super blurry buffer
        
        float4 nearInput;
#   	if HORIZONTAL
            // On the first pass, extract coverage from the near field radius
            // Note that the near field gets a box-blur instead of a Gaussian 
            // blur; we found that the quality improvement was not worth the 
            // performance impact of performing the Gaussian computation here as well.

            // Curve the contribution based on the radius.  We tuned this particular
            // curve to blow out the near field while still providing a reasonable
            // transition into the focus field.
            nearInput.a = float(abs(delta) <= r_B) * saturate(r_B * invNearBlurRadiusPixels * 4.0);
            nearInput.a *= nearInput.a;
            nearInput.a *= nearInput.a;

            // Compute premultiplied-alpha color
            nearInput.rgb = blurInput.rgb * nearInput.a;
            
#       else
            // On the second pass, use the already-available alpha values
            nearInput = texelFetch(nearSourceBuffer, clamp(B, int2(0), textureSize(nearSourceBuffer, 0) - int2(1)), 0);
#       endif
                    
        // We subsitute the following efficient expression for the more complex: weight = gaussian[clamp(int(float(abs(delta) * (GAUSSIAN_TAPS - 1)) * invNearBlurRadiusPixels), 0, GAUSSIAN_TAPS)];
        weight =  float(abs(delta) < nearBlurRadiusPixels);
        nearResult += nearInput * weight;
        nearWeightSum += weight;
    }
    
#   if HORIZONTAL
        // Retain the packed radius on the first pass.  On the second pass it is not needed.
        blurResult.a = packedA;
#   else
        blurResult.a = 1.0;
#   endif

    // Normalize the blur
    blurResult.rgb /= blurWeightSum;

    // The taps are already normalized, but our Gaussian filter doesn't line up 
    // with them perfectly so there is a slight error.
    nearResult /= nearWeightSum;  
}
