
varying vec4 position;
varying vec3 normal;
varying vec4 color;

void main()
{	
	position = gl_Vertex;       // world space position
	normal = gl_Normal;         // world space normal
	color = gl_Color;

	gl_Position = ftransform();
}
