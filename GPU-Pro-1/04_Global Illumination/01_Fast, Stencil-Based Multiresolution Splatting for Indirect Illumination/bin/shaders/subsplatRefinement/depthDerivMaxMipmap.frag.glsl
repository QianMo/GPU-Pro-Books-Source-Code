// This shader is applied in an iterative fashion by the OpenGL program.
//     Each time it take in a level of depth derivative mipmap and outputs
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

// OK.  We're looking to create a "mipmap" which stores maximum values at every texel.
//      Usually, you'd just want a 2x2 region to do this (in which case the texColor[0-3] 
//      would be used).  But that leads to undetected depth discontinuities along the 
//      other borders of each texel.  To avoid that, I accumulate max/min depths from 
//      a 3x3 region (instead of a 2x2), which solves the problem.
void main( void )
{
	float texColor0 = texture2D( inputTex, gl_TexCoord[0].xy ).r;
	float maxVal = texColor0;
	float texColor1 = texture2D( inputTex, gl_TexCoord[0].xy+vec2(offset,0) ).r;
	maxVal = max(texColor1,maxVal);
	float texColor2 = texture2D( inputTex, gl_TexCoord[0].xy+vec2(offset,offset) ).r;
	maxVal = max(texColor2,maxVal);
	float texColor3 = texture2D( inputTex, gl_TexCoord[0].xy+vec2(0,offset) ).r;
	maxVal = max(texColor3,maxVal);
	float texColor4 = texture2D( inputTex, gl_TexCoord[0].xy-vec2(offset,0) ).r;
	maxVal = max(texColor4,maxVal);
	float texColor5 = texture2D( inputTex, gl_TexCoord[0].xy-vec2(0,offset) ).r;
	maxVal = max(texColor5,maxVal);
	float texColor6 = texture2D( inputTex, gl_TexCoord[0].xy-vec2(offset,offset) ).r;
	maxVal = max(texColor6,maxVal);
	float texColor7 = texture2D( inputTex, gl_TexCoord[0].xy+vec2(-offset,offset) ).r;
	maxVal = max(texColor7,maxVal);
	float texColor8 = texture2D( inputTex, gl_TexCoord[0].xy+vec2(offset,-offset) ).r;
	maxVal = max(texColor8,maxVal);
	
	gl_FragColor = vec4(maxVal);
}