
void main( void )
{	
	vec3 norm =  normalize( gl_TexCoord[6].xyz );
	vec3 V    = -normalize( gl_TexCoord[5].xyz );
	float NdotV = max(0.0, dot( norm, V ));

	// We're scaling to change range of the position.  Why the odd
	//    number I no longer remember.  Perhaps for display purposes?
	gl_FragColor.xyz = vec3(0.08,0.08,0.08)*gl_TexCoord[5].xyz+0.5;
	gl_FragColor.w   = NdotV;
}

