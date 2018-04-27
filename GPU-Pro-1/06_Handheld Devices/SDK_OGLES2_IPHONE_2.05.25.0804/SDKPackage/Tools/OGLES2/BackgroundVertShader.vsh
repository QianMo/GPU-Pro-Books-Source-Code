attribute mediump vec2	myVertex;
attribute mediump vec2	myUV;

varying mediump vec2	varCoord;

void main()
{
	gl_Position = vec4(myVertex, 1, 1);
	varCoord = myUV;
}
