
uniform sampler2D depthTex;

void main( void )
{
	float depth = texture2D( depthTex, gl_TexCoord[0].xy ).z; 
	gl_FragColor = vec4( depth );
	gl_FragDepth = depth;
}

