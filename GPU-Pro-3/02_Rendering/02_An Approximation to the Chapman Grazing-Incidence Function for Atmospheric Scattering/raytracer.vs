
// --------------------------------------------------------

void main()
{
	// vertex position
	gl_TexCoord[0] = gl_Vertex;

	// camera position in model space
	gl_TexCoord[1] = gl_ModelViewMatrixInverse * vec4( 0, 0, 0, 1 );

	// z-axis is model space
	gl_TexCoord[2] = gl_ModelViewMatrixInverse * vec4( 0, 0, 1, 0 );

	// fixed function position transform
	gl_Position = ftransform();
}
