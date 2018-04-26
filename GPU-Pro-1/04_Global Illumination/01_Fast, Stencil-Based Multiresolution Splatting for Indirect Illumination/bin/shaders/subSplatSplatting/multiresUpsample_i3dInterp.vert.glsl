// This shader upsamples the multresolution buffer with the interpolation
//   scene proposed by our I3D 2009 paper.
//
// The interpolation process is iterative, so at any one step, only
//   the illumination at one level and the next-coarsest resolution 
//   are summed together.  This occurs in a coarse-to-fine manner
//   so after the first pass LV_max and LV_max-1 are summed together,
//   the next pass combines those with LV_(max-2), and after all the
//   passes, we have LV_0 + LV_1 + LV_2 + ... + LV_max.

// This is passed down to the shader, a simple gluOrtho2D matrix
//    to map x,y to [0..1].  It's setup during initialization.
uniform mat4 gluOrtho;  

// Transform the quad by the orthographic projection.  In this case,
//    the texture coordinates into the current level of the multires
//    buffer are the x&y coordinates of gl_Vertex and the texture
//    coordinates of the next coarser level are store in the first
void main( void )
{	
	gl_Position = gluOrtho * gl_Vertex;
	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_TexCoord[1] = gl_Vertex;
}