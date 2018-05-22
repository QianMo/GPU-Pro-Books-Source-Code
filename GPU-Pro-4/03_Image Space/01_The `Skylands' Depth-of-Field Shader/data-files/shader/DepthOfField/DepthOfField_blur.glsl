// -*- c++ -*-
/**
 \file DepthOfField_blur.glsl

 This is included by both the horizontal
 (first) and vertical (second) passes.

 The output blurResult is a buffer that is the scene blurred with a spatially varying kernel
 kernel that attempts to make the point-spread function at each pixel
 equal to its circle of confusion radius.

   blurResult.rgb = color
   blurResult.a   = normalized coc radius

 */
#include "../compatibility.glsl"
#line 18

/** Maximum blur radius for any point in the scene, in pixels.  Used to
    reconstruct the CoC radius from the normalized CoC radius. */
uniform int         maxCoCRadiusPixels;

/** Source image in RGB, normalized CoC in A. */
uniform sampler2D	blurSourceBuffer;

#if HORIZONTAL
	const int2 direction = int2(1, 0);
#else
	const int2 direction = int2(0, 1);

	/** For the second pass, the output of the previous near-field blur pass. */
	uniform sampler2D  nearSourceBuffer;
#endif

#if __VERSION__ < 130
#	define blurResult gl_FragColor
#else
	out layout(location = 0) float4 blurResult;
#endif
    
void main() {

    const int KERNEL_TAPS = 6;
	float kernel[KERNEL_TAPS + 1];
	// 11 x 11 separated kernel weights.  This does not dictate the 
    // blur kernel for depth of field; it is scaled to the actual
    // kernel at each pixel.
	kernel[6] = 0.00000000000000;  // Weight applied to outside-radius values

    // Custom // Gaussian
	kernel[5] = 0.50;//0.04153263993208;
	kernel[4] = 0.60;//0.06352050813141;
	kernel[3] = 0.75;//0.08822292796029;
	kernel[2] = 0.90;//0.11143948794984;
	kernel[1] = 1.00;//0.12815541114232;
	kernel[0] = 1.00;//0.13425804976814;

	// Accumulate the blurry image color
	blurResult.rgb  = float3(0.0f);
	float blurWeightSum = 0.0f;
    
	// Location of the central filter tap (i.e., "this" pixel's location).
    // Since the blur filter reduces the size by 75% in the dimension along which it is filtering,
    // we need to alter the coordinate scale
	int2 A = int2(gl_FragCoord.xy) * (direction * 3 + ivec2(1));

    float packedA = texelFetch(blurSourceBuffer, A, 0).a;
    float r_A = (packedA * 2.0 - 1.0) * maxCoCRadiusPixels;

    int2 maxCoord = textureSize(blurSourceBuffer, 0) - int2(1);
    
    for (int delta = -maxCoCRadiusPixels; delta <= maxCoCRadiusPixels; ++delta)	{
		// Tap location near A
		int2   B = A + (direction * delta);

		// Packed values
		float4 blurInput = texelFetch(blurSourceBuffer, clamp(B, int2(0), maxCoord), 0);

		// Signed kernel radius at this tap, in pixels
		float r_B = (blurInput.a * 2.0 - 1.0) * float(maxCoCRadiusPixels);
        
        // Stretch the kernel extent to the radius at pixel B.
        float weight = kernel[clamp(int(float(abs(delta) * (KERNEL_TAPS - 1)) / (0.001 + abs(r_B * 0.8))), 0, KERNEL_TAPS)];

		blurWeightSum  += weight;
		blurResult.rgb += blurInput.rgb * weight;
	}

#   if HORIZONTAL
	    // Retain the packed radius on the first pass.  On the second pass it is not needed.
	    blurResult.a = packedA;
#   else
        blurResult.a = 1.0;
#   endif

	// Normalize the blur
	blurResult.rgb /= blurWeightSum;
}
