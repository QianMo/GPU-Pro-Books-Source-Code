uniform sampler2D inputTex;
uniform float contrib;

varying out vec3 result;


void main()
{
	result = texture2D(inputTex, gl_TexCoord[0].st).rgb * contrib;
}