
varying vec4 position;
varying vec3 normal;

void main()
{	
	position = gl_Vertex;       // world space for viewer
	normal = gl_Normal;

	gl_Position = ftransform();
}
