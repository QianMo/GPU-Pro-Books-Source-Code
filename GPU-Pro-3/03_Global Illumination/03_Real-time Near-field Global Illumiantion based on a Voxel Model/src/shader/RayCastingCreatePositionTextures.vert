varying vec4 P;

void main()
{
	gl_FrontColor = gl_Color;	// pass through color
	gl_Position = ftransform();	// fixed function pipeline functionality
	P = gl_ModelViewMatrix * gl_Vertex;
}