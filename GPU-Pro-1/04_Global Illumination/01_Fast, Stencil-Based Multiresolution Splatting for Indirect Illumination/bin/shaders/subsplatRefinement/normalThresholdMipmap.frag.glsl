// This shader is applied in an iterative fashion by the OpenGL program.
//     Each time it take in a level of normal mipmap and outputs
//     the next finer level.  Usually mipmap creation looks at 2x2 regions
//     on the input region.  However, to avoid missing discontinuities that
//     would require closer illumination samples, each level of the mipmap
//     creation computes the maximum depth derivative in a 3x3 region.

// The input of the next finer level of the mipmap
uniform sampler2D inputTex;

// An offset that for accessing neighboring texels in the current mipmap level
//     (euqivalent to 1.0/resolution).  This should probably be
//     a 2D value with variations between x and y, but given this
//     demo uses square buffers this isn't an issue.
uniform float offset;

// OK.  We're looking to create a normal "mipmap" which stores min and max normals 
//      at every texel.  Because we know all normals face the viewer, we store only
//      min and max x & y values.  Usually, you'd just want a 2x2 region to do this 
//      (in which case the texColor[0-3] would be used).  But that leads to 
//      undetected depth discontinuities along the other borders of each texel.  
//      To avoid that, I accumulate max/min depths from a 3x3 region (instead of a 
//      2x2), which solves the problem.
void main( void )
{
	vec2 minNorm, maxNorm;
	
	vec4 texColor0 = texture2D( inputTex, gl_TexCoord[0].xy );
	minNorm = texColor0.xy;
	maxNorm = texColor0.zw;

	vec4 texColor1 = texture2D( inputTex, gl_TexCoord[0].xy+vec2(offset,0) );
	minNorm = min( minNorm, texColor1.xy );
	maxNorm = max( maxNorm, texColor1.zw );
	
	vec4 texColor2 = texture2D( inputTex, gl_TexCoord[0].xy+vec2(offset,offset) );
	minNorm = min( minNorm, texColor2.xy );
	maxNorm = max( maxNorm, texColor2.zw );
	
	vec4 texColor3 = texture2D( inputTex, gl_TexCoord[0].xy+vec2(0,offset) );
	minNorm = min( minNorm, texColor3.xy );
	maxNorm = max( maxNorm, texColor3.zw );
	
	vec4 texColor4 = texture2D( inputTex, gl_TexCoord[0].xy-vec2(offset,0) );
	minNorm = min( minNorm, texColor4.xy );
	maxNorm = max( maxNorm, texColor4.zw );
	
	vec4 texColor5 = texture2D( inputTex, gl_TexCoord[0].xy-vec2(0,offset) );
	minNorm = min( minNorm, texColor5.xy );
	maxNorm = max( maxNorm, texColor5.zw );
	
	vec4 texColor6 = texture2D( inputTex, gl_TexCoord[0].xy-vec2(offset,offset) );
	minNorm = min( minNorm, texColor6.xy );
	maxNorm = max( maxNorm, texColor6.zw );
	
	vec4 texColor7 = texture2D( inputTex, gl_TexCoord[0].xy+vec2(-offset,offset) );
	minNorm = min( minNorm, texColor7.xy );
	maxNorm = max( maxNorm, texColor7.zw );
	
	vec4 texColor8 = texture2D( inputTex, gl_TexCoord[0].xy+vec2(offset,-offset) );
	minNorm = min( minNorm, texColor8.xy );
	maxNorm = max( maxNorm, texColor8.zw );

	gl_FragColor = vec4( minNorm, maxNorm );
}