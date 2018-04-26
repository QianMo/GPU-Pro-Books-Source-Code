#version 120 
#extension GL_EXT_gpu_shader4 : enable

uniform float lightIntensity;
uniform vec4 sceneAmbient;


void main( void )
{
	vec3 norm = normalize( gl_TexCoord[6].xyz );
	float NdotV = max( 0.0, dot( norm, normalize( -gl_TexCoord[5].xyz ) ) );
	
	gl_FragColor = sceneAmbient;
	gl_FragColor.a = length( gl_TexCoord[5].xyz );
	
}