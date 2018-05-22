uniform sampler2D input;

varying out vec3 result;

void main()
{
	result = max(vec3(0), texture2D(input, gl_TexCoord[0].st).rgb);
}