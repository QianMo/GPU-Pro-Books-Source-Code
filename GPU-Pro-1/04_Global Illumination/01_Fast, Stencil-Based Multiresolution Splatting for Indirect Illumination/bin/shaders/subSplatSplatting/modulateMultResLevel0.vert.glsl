// This shader takes as input the multiresolution indirect illuination
//    buffer, after it has been upsampled.  However, it still resides
//    in a 2048x1024 flattened mipmap structure, so we cannot directly
//    copy this to the output.  Instead we select the correct segment
//    of this multires buffer (i.e., the left half), multiply it by
//    our G-buffer's fragment albedo (which hasn't been done yet)
//    and output (i.e., blend) the result into the current framebuffer

// This is passed down to the shader, a simple gluOrtho2D matrix
//    to map x,y to [0..1].  It's setup during initialization.
uniform mat4 gluOrtho;

// A simple vertex shader.  Transform by the gluOrtho matrix,
//    and pass the specified texture coordinates to the fragment
//    shader.
void main( void )
{  
	gl_Position = gluOrtho * gl_Vertex;
	gl_TexCoord[0] = gl_MultiTexCoord0;
}