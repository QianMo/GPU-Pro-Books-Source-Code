// This shader takes as input the multiresolution indirect illuination
//    buffer, after it has been upsampled.  However, it still resides
//    in a 2048x1024 flattened mipmap structure, so we cannot directly
//    copy this to the output.  Instead we select the correct segment
//    of this multires buffer (i.e., the left half), multiply it by
//    our G-buffer's fragment albedo (which hasn't been done yet)
//    and output (i.e., blend) the result into the current framebuffer

// This is our multresolution illumination buffer.  At this point
//    in the pipeline, it should already have been upsampled.  Thus
//    only the left half (i.e., x in [0..0.5]) is necessary -- the
//    rest is "garbage" output from our ping-pong upsampling approach
uniform sampler2D multiResBuf;

// This is the G-buffer's color buffer that stores the albedo at every
//    point visible from the eye.  We need to multiply this with the
//    indirect illumination to get the indirect contribution seen from
//    the eye.
uniform sampler2D geomTexColor;

void main( void )
{
	// Get our incident indirect illumination
	vec4 irradiance = max( vec4(0.0), texture2D( multiResBuf,  vec2(gl_TexCoord[0].x*0.5, gl_TexCoord[0].y) ) );
	
	// Get the albedo of the surface
	vec4 geomColor  = texture2D( geomTexColor, gl_TexCoord[0].xy );
	
	// Output the indirect illumination seen by the eye
	gl_FragColor = irradiance * geomColor;
	gl_FragColor.a = 1.0;
}
