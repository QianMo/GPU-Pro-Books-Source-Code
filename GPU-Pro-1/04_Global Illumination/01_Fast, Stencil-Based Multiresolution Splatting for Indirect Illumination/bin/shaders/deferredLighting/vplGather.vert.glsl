// We're doing a deferred illumination here, where drawing a 
//    single full-screen quad.  At each fragment on the full screen
//    quad, we look up the g-buffer data for the location and
//    illumination that surface from all of our sampled VPLs.
//    This is blended with the direct illumination that has
//    already been computed and rendered into the framebuffer.


// This is passed down to the shader, a simple gluOrtho2D matrix
//    to map x,y to [0..1].  It's setup during initialization.
uniform mat4 gluOrtho;  

void main( void )
{	
	// Transformations for generating the full-screen quad used to render fragments
	//     Pass the vertex locations (i.e., [0..1] in x and y) down as texture coords
	gl_Position = gluOrtho * gl_Vertex;
	gl_TexCoord[0] = gl_Vertex; 	
}