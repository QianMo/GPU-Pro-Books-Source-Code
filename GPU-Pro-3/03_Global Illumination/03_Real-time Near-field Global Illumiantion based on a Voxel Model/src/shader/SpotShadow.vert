// QUAD
void main()
{
	gl_Position = ftransform();	// fixed function pipeline functionality
	gl_TexCoord[0] = gl_MultiTexCoord0;
}
