#version 120 
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D       theTexture;
uniform float           hasTexture;

uniform vec4 amb, dif, spec, shiny;


void main( void )
{	
	// Output the texture color
	//gl_FragData[0] = (hasTexture > 0.5 ? texture2D( theTexture, gl_TexCoord[0].xy ): dif) * 
    //	                   dot( normalize( gl_TexCoord[6].xyz ), -normalize( gl_TexCoord[5].xyz ) );
	gl_FragData[0] =(hasTexture > 0.5 ? texture2D( theTexture, gl_TexCoord[0].xy ): dif);
	
	// Available for data!
	float NdotV = dot( normalize( gl_TexCoord[6].xyz ), -normalize( gl_TexCoord[5].xyz ) ); 
	gl_FragData[0].w = NdotV < 0 ? -NdotV : NdotV;
	
	// Output the surface normal
	gl_FragData[1].xyz = (NdotV < 0 ? -1.0 : 1.0) * normalize( gl_TexCoord[6].xyz );
	
	// Output the linear world-space depth
	//gl_FragData[1].w   = dot( gl_TexCoord[5].xyz, gl_TexCoord[5].xyz ); 
	gl_FragData[1].w   = length( gl_TexCoord[5].xyz ); 
	
	// Output the eye-space position
	gl_FragData[2].xyz = gl_TexCoord[5].xyz;
	
	// Available for data!
	//gl_FragData[2].w = dot( gl_TexCoord[5].xyz, gl_TexCoord[5].xyz ); 
	gl_FragData[2].w = length( gl_TexCoord[5].xyz ); 
}