
void main( void )
{	
	vec4 eyeSpacePos  = gl_ModelViewMatrix * gl_Vertex;
	vec3 eyeSpaceNorm = gl_NormalMatrix * gl_Normal;
	gl_TexCoord[5]     = eyeSpacePos / eyeSpacePos.w;
	gl_TexCoord[6].xyz = eyeSpaceNorm;
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}

