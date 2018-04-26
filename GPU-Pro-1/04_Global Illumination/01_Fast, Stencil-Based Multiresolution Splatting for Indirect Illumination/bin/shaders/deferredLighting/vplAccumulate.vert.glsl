// We're doing a deferred illumination here, where we're picking 
//    a single virtual point light (stored in the texture vplCompact)
//    and illuminating every fragment in our scene by this single
//    light.  This is accumulated (i.e., blended) with the direct
//    illumination that has already been computed and rendered into
//    the framebuffer.


// This is passed down to the shader, a simple gluOrtho2D matrix
//    to map x,y to [0..1].  It's setup during initialization.
uniform mat4 gluOrtho;  

// A compact list of the sampled VPL locations we'll use to 
//    illuminate the scene
uniform sampler2D vplCompact;

void main( void )
{	
	// Transformations for generating the full-screen quad used to render fragments
	//     Pass the vertex locations (i.e., [0..1] in x and y) down as texture coords
	gl_Position = gluOrtho * gl_Vertex;
	gl_TexCoord[0] = gl_Vertex; 	
	
	// Lookup the VPL position & pass down to the fragment shader. 
	gl_TexCoord[1] = texture2D( vplCompact, vec2( gl_MultiTexCoord0.y, 0.5 ) );
	
	// Lookup the VPL surface normal & pass down to the fragment shader. 
	gl_TexCoord[2] = texture2D( vplCompact, vec2( gl_MultiTexCoord0.y, 1.0 ) );
	
	// Grab the VPL's reflected flux (aka surface albedo, aka color).  Pass to the frag shader.
	gl_TexCoord[3] = texture2D( vplCompact, vec2( gl_MultiTexCoord0.y, 0.0 ) );
}