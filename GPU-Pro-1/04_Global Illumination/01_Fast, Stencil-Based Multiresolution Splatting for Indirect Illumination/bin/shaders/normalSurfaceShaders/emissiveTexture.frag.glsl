#version 120 
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D       wallTex;

uniform vec4 amb, dif;
uniform float lightIntensity;
uniform vec4 sceneAmbient;

void main( void )
{
	vec4 color = lightIntensity * texture2D( wallTex, gl_TexCoord[0].xy );
	
	gl_FragColor = color + sceneAmbient;
	gl_FragColor.a = length( gl_TexCoord[5].xyz );
	
}