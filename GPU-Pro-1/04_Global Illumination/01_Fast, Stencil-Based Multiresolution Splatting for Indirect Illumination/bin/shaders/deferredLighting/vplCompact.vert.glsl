// This is a simple shader that looks up our VPL samples
//    in our original full-screen RSM and stores them in
//    a compact texture with resolution #samples x 3.
//    (The 3 is because we have 3 render targets in our RSM).

//#version 120 
//#extension GL_EXT_gpu_shader4 : enable

// As most of our shaders, we're using a full-screen quad,
//    so we'll use a standard gluOrtho matrix for this instead
//    of looking at the gl_ModelViewMatrix.
uniform mat4 gluOrtho;

// A simple vertex shader.  Transform by the gluOrtho matrix,
//    and pass the specified texture coordinates to the fragment
//    shader.
void main( void )
{  
	gl_Position = gluOrtho * gl_Vertex;
	gl_TexCoord[0] = gl_MultiTexCoord0;
}