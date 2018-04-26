

void main( void )
{
	vec4 eyePos = gl_ModelViewMatrix * gl_Vertex;
	vec3 eyeNorm = gl_NormalMatrix * gl_Normal;
	
	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_TexCoord[5] = eyePos;
	gl_TexCoord[6].xyz = eyeNorm;
	
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}