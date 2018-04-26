varying	vec3 lightOut;
varying	vec3 halfOut;
varying	vec3 normalOut;
varying vec3 viewerOut;
varying vec4 colorOut;
varying vec2 uvOut;

uniform sampler2D diffuseTexture;

void main (void)
{	
	const vec4	diffColor = vec4 ( 1.0 );
	const vec4	specColor = vec4 ( 0.7, 0.7, 0.7, 1.0 );
	const float	specPower = 200.0;

	vec3	n     =  normalize ( normalOut );
	vec3	l     =  normalize ( lightOut );
	vec3	h     =  normalize ( halfOut );
	vec4	diff  =  diffColor * max ( dot ( n, l ), 0.0 );
	vec4	spec  =  specColor * pow ( max ( dot ( n, h ), 0.0 ), specPower );

	vec4 diffuseColor = texture2D(diffuseTexture, uvOut);
	
	gl_FragColor = ((diff * diffuseColor) + spec);
}
