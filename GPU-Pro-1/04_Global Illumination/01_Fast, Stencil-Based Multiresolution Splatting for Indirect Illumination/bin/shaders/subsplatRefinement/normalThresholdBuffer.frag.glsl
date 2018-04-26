// This shader creates the buffer that stores min/max surface normals.
//    The result of this shader is passed through the interative
//    application of normalThresholdMipmap.frag.glsl to get a normal
//    mipmap that is used to refine the illumination shading points.

uniform sampler2D rsmBuffer1;  //  Holds the normal & linear depth of the g-buffer image
                               //   .xyz contains normalized surface normal,
                               //   .w   contains world-space distance from eye to fragment
uniform sampler2D rsmBuffer2;  // Stores the fragment positions from the g-buffer.
                               //    This is used to identify fragments where no illumination
                               //    is possible, because it is behind the light. (Can be a big win)
                           
uniform float offset;          // Offset used to get adjacent texels in the RSM.

void main( void )
{
	vec4 fragNorm = texture2D( rsmBuffer1, gl_TexCoord[0].xy );

	// First, compute the normal variatios in the region
	//    This is computed in a 3x3 region around each texel.  We need to do this 
	//    here, rather than during mipmap creation to avoid problems along
	//    power-of-two pixel block boundaries.
	vec2 norm0 = fragNorm.xy; // texture2D( rsmBuffer1, gl_TexCoord[0].xy ).xy;
	vec2 norm1 = texture2D( rsmBuffer1, gl_TexCoord[0].xy+vec2(offset,0) ).xy;
	vec2 maxNorm = max(norm0,norm1);
	vec2 minNorm = min(norm0,norm1);
	vec2 norm2 = texture2D( rsmBuffer1, gl_TexCoord[0].xy+vec2(0,offset) ).xy;
	maxNorm = max(maxNorm,norm2);
	minNorm = min(minNorm,norm2);
	vec2 norm3 = texture2D( rsmBuffer1, gl_TexCoord[0].xy+vec2(offset,offset) ).xy;
	maxNorm = max(maxNorm,norm3);
	minNorm = min(minNorm,norm3);
	vec2 norm4 = texture2D( rsmBuffer1, gl_TexCoord[0].xy-vec2(offset,0) ).xy;
	maxNorm = max(maxNorm,norm4);
	minNorm = min(minNorm,norm4);
	vec2 norm5 = texture2D( rsmBuffer1, gl_TexCoord[0].xy-vec2(0,offset) ).xy;
	maxNorm = max(maxNorm,norm5);
	minNorm = min(minNorm,norm5);
	vec2 norm6 = texture2D( rsmBuffer1, gl_TexCoord[0].xy-vec2(offset,offset) ).xy;
	maxNorm = max(maxNorm,norm6);
	minNorm = min(minNorm,norm6);
	vec2 norm7 = texture2D( rsmBuffer1, gl_TexCoord[0].xy+vec2(-offset,offset) ).xy;
	maxNorm = max(maxNorm,norm7);
	minNorm = min(minNorm,norm7);
	vec2 norm8 = texture2D( rsmBuffer1, gl_TexCoord[0].xy+vec2(offset,-offset) ).xy;
	maxNorm = max(maxNorm,norm8);
	minNorm = min(minNorm,norm8);
	
	// Here's our depth derivative we'd like to store.
	vec4 minMaxNorms = vec4( minNorm, maxNorm );
	
	// Write out our computed values to the buffer
	gl_FragColor = minMaxNorms; 
}









