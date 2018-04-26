// We're doing a deferred illumination here, and this shader
//    is roughly equivalent to the simple vplGather.frag.glsl 
//    shader, only instead of working on a full-screen splat
//    at the highest resolution, it acts on our multiresolution
//    splat.
// Interestingly, because we're using the stencil buffer to
//    selectively render into the correct locations in the
//    multires buffer, this shader looks almost identical to
//    the vplGather.frag.glsl shader.

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