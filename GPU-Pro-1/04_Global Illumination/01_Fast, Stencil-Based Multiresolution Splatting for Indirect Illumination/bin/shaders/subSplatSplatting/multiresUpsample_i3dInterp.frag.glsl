// This shader upsamples the multresolution buffer with the interpolation
//   scene proposed by our I3D 2009 paper.
//
// The interpolation process is iterative, so at any one step, only
//   the illumination at one level and the next-coarsest resolution 
//   are summed together.  This occurs in a coarse-to-fine manner
//   so after the first pass LV_max and LV_max-1 are summed together,
//   the next pass combines those with LV_(max-2), and after all the
//   passes, we have LV_0 + LV_1 + LV_2 + ... + LV_max.

// The current, partially-upsampled multiresolution illumination buffer 
uniform sampler2D multiResBuf;

// The resolution of the coarse buffer we're currently upsampling
uniform float coarseRes;

// The boundaries of the two multresolution images (.x = fine, .y = coarse)
uniform vec2 leftEdge, rightEdge, topEdge;

// The proper horizontal and vertical increments for a one-pixel offset in multiResBuf
uniform float hInc, vInc;

// The main entry point for our I3D upsampling and interpolation shader.  
void main( void )
{
	vec4 tex[9];

	// Identify the weights for the nearby fragments.  Here, we're using a tent filter over a 3x3 region (for weights)
	vec2 weight = abs( gl_TexCoord[0].zw*coarseRes - floor(gl_TexCoord[0].zw*coarseRes) ) ;	
	
	// Grab a 3x3 region around the central pixel in the coarser level.
	vec2 coarseClampMin = vec2( leftEdge.y, 0.0 ), coarseClampMax = vec2( rightEdge.y, topEdge.y );
	tex[0] = texture2D( multiResBuf, clamp( gl_TexCoord[0].xy + vec2( 0.0, 0.0 ), coarseClampMin, coarseClampMax ) );
	tex[1] = texture2D( multiResBuf, clamp( gl_TexCoord[0].xy + vec2( 0.0, vInc ), coarseClampMin, coarseClampMax ) );
	tex[2] = texture2D( multiResBuf, clamp( gl_TexCoord[0].xy + vec2( 0.0, -vInc ), coarseClampMin, coarseClampMax ) );
	tex[3] = texture2D( multiResBuf, clamp( gl_TexCoord[0].xy + vec2( hInc, 0.0 ), coarseClampMin, coarseClampMax ) );
	tex[4] = texture2D( multiResBuf, clamp( gl_TexCoord[0].xy + vec2( -hInc, 0.0 ), coarseClampMin, coarseClampMax ) );
	tex[5] = texture2D( multiResBuf, clamp( gl_TexCoord[0].xy + vec2( hInc, vInc ), coarseClampMin, coarseClampMax ) );
	tex[6] = texture2D( multiResBuf, clamp( gl_TexCoord[0].xy + vec2( hInc, -vInc ), coarseClampMin, coarseClampMax ) );
	tex[7] = texture2D( multiResBuf, clamp( gl_TexCoord[0].xy + vec2( -hInc, vInc ), coarseClampMin, coarseClampMax ) );
	tex[8] = texture2D( multiResBuf, clamp( gl_TexCoord[0].xy + vec2( -hInc, -vInc ), coarseClampMin, coarseClampMax ) );
	
	// Grab a 3x3 region around the central pixel in the finer level.
	vec2 fineClampMin = vec2( leftEdge.x, 0.0 ), fineClampMax = vec2( rightEdge.x, topEdge.x );
	tex[0] += (1.0-tex[0].a)*texture2D( multiResBuf, clamp(gl_TexCoord[1].xy+vec2( 0.0, 0.0 ),fineClampMin,fineClampMax) ) ;
	tex[1] += (1.0-tex[1].a)*texture2D( multiResBuf, clamp(gl_TexCoord[1].xy+vec2( 0.0, vInc ),fineClampMin,fineClampMax) ) ;
	tex[2] += (1.0-tex[2].a)*texture2D( multiResBuf, clamp(gl_TexCoord[1].xy+vec2( 0.0, -vInc ),fineClampMin,fineClampMax) ) ;
	tex[3] += (1.0-tex[3].a)*texture2D( multiResBuf, clamp(gl_TexCoord[1].xy+vec2( hInc, 0.0 ),fineClampMin,fineClampMax) ) ;
	tex[4] += (1.0-tex[4].a)*texture2D( multiResBuf, clamp(gl_TexCoord[1].xy+vec2( -hInc, 0.0 ),fineClampMin,fineClampMax) ) ;
	tex[5] += (1.0-tex[5].a)*texture2D( multiResBuf, clamp(gl_TexCoord[1].xy+vec2( hInc, vInc ),fineClampMin,fineClampMax) ) ;
	tex[6] += (1.0-tex[6].a)*texture2D( multiResBuf, clamp(gl_TexCoord[1].xy+vec2( hInc, -vInc ),fineClampMin,fineClampMax) ) ;
	tex[7] += (1.0-tex[7].a)*texture2D( multiResBuf, clamp(gl_TexCoord[1].xy+vec2( -hInc, vInc ),fineClampMin,fineClampMax) ) ;
	tex[8] += (1.0-tex[8].a)*texture2D( multiResBuf, clamp(gl_TexCoord[1].xy+vec2( -hInc, -vInc ),fineClampMin,fineClampMax) ) ;

	
	// Compute weights for each of the 3 x 3 samples.  Weights for all samples (except the central one)
	//    are forced to 0 if the texel is invalid (i.e., tex[?].a == 0).  If the texel is valid (i.e.,
	//    tex[?].a == 1) then the weight is left alone.  Corner texels are only considered valid if
	//    the corner sample exists (i.e., tex[?].a == 0) AND at least one of the two adjacent samples
	//    (in a cardinal direction) also is valid.
	float weights[9];
	weights[0] = 1.0;
	weights[1] = tex[1].a * weight.y;
	weights[2] = tex[2].a * (1.0-weight.y);
	weights[3] = tex[3].a * weight.x;
	weights[4] = tex[4].a * (1.0-weight.x);
	weights[5] = min( 1.0, tex[1].a+tex[3].a )*tex[5].a*weight.x*weight.y;
	weights[6] = min( 1.0, tex[2].a+tex[3].a )*tex[6].a*weight.x*(1.0-weight.y);
	weights[7] = min( 1.0, tex[1].a+tex[4].a )*tex[7].a*(1.0-weight.x)*weight.y ;
	weights[8] = min( 1.0, tex[2].a+tex[4].a )*tex[8].a*(1.0-weight.x)*(1.0-weight.y);

	// To correctly normalize the weighted sum, we have to find the sum of the weights
	float weightSum = weights[0]+weights[1]+weights[2]+
	                  weights[3]+weights[4]+weights[5]+
	                  weights[6]+weights[7]+weights[8];

	// Compute the weighted sum of our samples	
	vec4 color = weights[0]*tex[0]+weights[1]*tex[1]+weights[2]*tex[2]+
	             weights[3]*tex[3]+weights[4]*tex[4]+weights[5]*tex[5]+
	             weights[6]*tex[6]+weights[7]*tex[7]+weights[8]*tex[8];
	color /= weightSum;
	color.a = 1.0;  // Identifies this sample as valid in subsequent passes
	
	
	// We only output this color if this fragment was valid in one of the prior
	//    multresolution buffers.  We do not want to interpolate into invalid regions
	//    of the buffer, as this causes multiresolution haloing artifacts (see I3D paper).
	//    This makes sure we only interpolate over valid fragments (i.e., tex[0].a == 1)
	gl_FragColor = tex[0].a * color;
}



