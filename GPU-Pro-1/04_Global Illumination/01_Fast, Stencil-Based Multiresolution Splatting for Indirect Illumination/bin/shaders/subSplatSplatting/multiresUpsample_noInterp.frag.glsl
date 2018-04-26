// This shader upsamples the multresolution buffer without
//   performing any fancy interpolation on it (i.e., nearest neighbor).
//
// The interpolation process is iterative, so at any one step, only
//   the illumination at one level and the next-coarsest resolution 
//   are summed together.  This occurs in a coarse-to-fine manner
//   so after the first pass LV_max and LV_max-1 are summed together,
//   the next pass combines those with LV_(max-2), and after all the
//   passes, we have LV_0 + LV_1 + LV_2 + ... + LV_max.

// The current, partially-upsampled multiresolution illumination buffer 
uniform sampler2D multiResBuf;

void main( void )
{
	// Add the color at the current resolution with the next coarser level
	gl_FragColor = texture2D( multiResBuf, gl_TexCoord[0].xy ) + 
	               texture2D( multiResBuf, gl_TexCoord[1].xy );
}



