// This shader creates the buffer that stores max depth derivative of
//    an input image.  The result of this shader is passed through the 
//    interative application of depthDerivMaxMipmap.frag.glsl to get a 
//    depth derivative mipmap that is used to refine the illumination 
//    shading points.

uniform sampler2D rsmBuffer1;  //  Holds the normal & linear depth of the g-buffer image
                               //   .xyz contains normalized surface normal,
                               //   .w   contains world-space distance from eye to fragment
                               
uniform float offset;          // Offset used to get adjacent texels in the RSM.

uniform float farPlane;        // Distance to the far plane

// Usually, for mipmaps we use a 2x2 region to compute the average.  For the 
//     max derivative mipmap, however, we're using the mipmap to detect discontinuities,
//     which could present a problem if the discontinuity runs right between mipmap
//     texels (at any level).  By using a 3x3 region to determin derivatives we make
//     sure to check discontinuities in these mipmapping cracks.
void main( void )
{
	float depth0 = texture2D( rsmBuffer1, gl_TexCoord[0].xy ).w;
	float depth1 = texture2D( rsmBuffer1, gl_TexCoord[0].xy+vec2(offset,0) ).w;
	float maxDepth = max(depth0,depth1);
	float minDepth = min(depth0,depth1);
	float depth2 = texture2D( rsmBuffer1, gl_TexCoord[0].xy+vec2(0,offset) ).w;
	maxDepth = max(maxDepth,depth2);
	minDepth = min(minDepth,depth2);
	float depth3 = texture2D( rsmBuffer1, gl_TexCoord[0].xy+vec2(offset,offset) ).w;
	maxDepth = max(maxDepth,depth3);
	minDepth = min(minDepth,depth3);
	float depth4 = texture2D( rsmBuffer1, gl_TexCoord[0].xy-vec2(offset,0) ).w;
	maxDepth = max(maxDepth,depth4);
	minDepth = min(minDepth,depth4);
	float depth5 = texture2D( rsmBuffer1, gl_TexCoord[0].xy-vec2(0,offset) ).w;
	maxDepth = max(maxDepth,depth5);
	minDepth = min(minDepth,depth5);
	float depth6 = texture2D( rsmBuffer1, gl_TexCoord[0].xy-vec2(offset,offset) ).w;
	maxDepth = max(maxDepth,depth6);
	minDepth = min(minDepth,depth6);
	float depth7 = texture2D( rsmBuffer1, gl_TexCoord[0].xy+vec2(-offset,offset) ).w;
	maxDepth = max(maxDepth,depth7);
	minDepth = min(minDepth,depth7);
	float depth8 = texture2D( rsmBuffer1, gl_TexCoord[0].xy+vec2(offset,-offset) ).w;
	maxDepth = max(maxDepth,depth8);
	minDepth = min(minDepth,depth8);
	
	gl_FragColor = farPlane*vec4( maxDepth-minDepth )/depth0; // the derivative of the local linear depth
}