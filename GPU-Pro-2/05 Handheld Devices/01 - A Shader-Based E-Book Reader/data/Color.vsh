
attribute highp   vec2  inVertex;

uniform bool		Rotate;

void RotatePos()
{
	if (Rotate)
	{
		highp float x = gl_Position.x;
		gl_Position.x = -gl_Position.y;
		gl_Position.y = x;
	}
}

void main()
{
	gl_Position = vec4(inVertex, 0.0, 1.0);
	
	RotatePos();
}