uniform sampler2D	sampler2d;

varying lowp vec4		varColour;
varying mediump vec2	texCoord;

void main()
{
	gl_FragColor = varColour * texture2D(sampler2d, texCoord);
}
