#version 120
#extension EXT_gpu_shader4 : enable

uniform sampler2DArray textureArray;

void main( void )
{
	vec4 lv0_color = texture2DArray( textureArray, vec3(         gl_TexCoord[0].xy, 0 ) );
	vec4 lv1_color = texture2DArray( textureArray, vec3( 0.5   * gl_TexCoord[0].xy, 1 ) );
	vec4 lv2_color = texture2DArray( textureArray, vec3( 0.25  * gl_TexCoord[0].xy, 2 ) );
	vec4 lv3_color = texture2DArray( textureArray, vec3( 0.125 * gl_TexCoord[0].xy, 3 ) );
	
	gl_FragColor = lv0_color + 0.25 * lv1_color + 0.0625 * lv2_color + 0.015625 * lv3_color;
}

