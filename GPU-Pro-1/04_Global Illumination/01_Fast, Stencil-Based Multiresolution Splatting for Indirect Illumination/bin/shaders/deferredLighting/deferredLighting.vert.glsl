// We're doing a deferred shading were all the data
//    is stored in textures.  Thus our vertex shader is dead
//    simple, we just need to do an orthographical projection
//    and make sure the texture ranges from [0..1] over the quad

// This is passed down to the shader, a simple gluOrtho2D matrix
//    to map x,y to [0..1].  It's setup during initialization.
uniform mat4 gluOrtho;  

// Apply our GLU Ortho matrix, pass the vertex data down as
//    texture coordinates (so we don't need in the GL code to
//    explicitly give texture coordinates)
void main( void )
{	
	gl_Position = gluOrtho * gl_Vertex;
	gl_TexCoord[0] = gl_Vertex; 	
}