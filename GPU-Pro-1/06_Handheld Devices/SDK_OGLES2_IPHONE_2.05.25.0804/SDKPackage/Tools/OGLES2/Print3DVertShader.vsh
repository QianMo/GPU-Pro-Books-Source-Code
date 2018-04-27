attribute highp vec4	myVertex;
attribute mediump vec2	myUV;
attribute lowp vec4		myColour;

uniform highp mat4		myMVPMatrix;

varying lowp vec4		varColour;
varying mediump vec2	texCoord;

void main()
{
	gl_Position = myMVPMatrix * myVertex;
	texCoord = myUV.st;
	varColour = myColour;
}
