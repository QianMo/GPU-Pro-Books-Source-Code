#version 120 
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D       wallTex, spotLight, causticMap0, causticMap1;
uniform sampler2DShadow shadowMap0, shadowMap1, causticDepth0, causticDepth1;
uniform float			useShadowMap, useCausticMap;

uniform float lightsEnabled;
uniform float lightIntensity;
uniform vec4 sceneAmbient;

vec4 IsPointIlluminated( sampler2DShadow sMap, vec4 smapCoord )
{
	return ( all(equal(smapCoord.xyz,clamp(smapCoord.xyz,vec3(0),vec3(1)))) ? 
					shadow2D( sMap, smapCoord.xyz ).x * texture2D( spotLight, smapCoord.xy, 0.0 ): 
					vec4(0.0) );
}

vec4 CausticContribution( sampler2DShadow cDepth, sampler2D cMap, vec4 smapCoord )
{
	return ( all(equal(smapCoord.xyz,clamp(smapCoord.xyz,vec3(0),vec3(1)))) ? 
					shadow2D( cDepth, smapCoord.xyz ).x * texture2D( cMap, smapCoord.xy ): 
					vec4(0.0) );
}

void main( void )
{
	vec4 surfColor = texture2D( wallTex, gl_TexCoord[0].xy );
	vec3 norm = normalize( gl_TexCoord[6].xyz );
	
	// Compute N dot L for the light(s)
	vec3 toLight0 = normalize( gl_LightSource[0].position.xyz - gl_TexCoord[5].xyz );
	float NdotL0 = lightIntensity * max( 0.0, dot( norm, toLight0 ) );
	vec3 toLight1 = normalize( gl_LightSource[1].position.xyz - gl_TexCoord[5].xyz );
	float NdotL1 = lightIntensity * max( 0.0, dot( norm, toLight1 ) );
	
	// Compute the shadow map coordinate(s) for the light(s). 
	vec4 smapCoord0 = vec4( dot( gl_EyePlaneS[7], gl_TexCoord[5] ), dot( gl_EyePlaneT[7], gl_TexCoord[5] ),
						   dot( gl_EyePlaneR[7], gl_TexCoord[5] ), dot( gl_EyePlaneQ[7], gl_TexCoord[5] ) );
	smapCoord0 /= smapCoord0.w;
	smapCoord0.z = smapCoord0.z <= 0.0 ? 0.0 : smapCoord0.z;
	vec4 smapCoord1 = vec4( dot( gl_EyePlaneS[6], gl_TexCoord[5] ), dot( gl_EyePlaneT[6], gl_TexCoord[5] ),
						   dot( gl_EyePlaneR[6], gl_TexCoord[5] ), dot( gl_EyePlaneQ[6], gl_TexCoord[5] ) );
	smapCoord1 /= smapCoord1.w;
	smapCoord1.z = smapCoord1.z <= 0.0 ? 0.0 : smapCoord1.z;
	
	// Now accumulate illumination. 
	vec4 lit = vec4( 0.0 );
	
	// Add in direct lighting
	lit += ( (lightsEnabled > 0.1 && useShadowMap > 0) ? 
		     IsPointIlluminated( shadowMap0, smapCoord0 )*NdotL0 : vec4(NdotL0) );
	lit += ( (lightsEnabled > 1.1 && useShadowMap > 0) ? 
	         IsPointIlluminated( shadowMap1, smapCoord1 )*NdotL1 : 
	         (lightsEnabled < 1.1 ? vec4(0.0) : vec4(NdotL1)) );
	
	// Add in caustic map lighting
	lit += ( (lightsEnabled > 0.1 && useCausticMap > 0) ? 
	         CausticContribution( causticDepth0, causticMap0, smapCoord0 )*NdotL0 : vec4(0.0) );
	lit += ( (lightsEnabled > 1.1 && useCausticMap > 0) ? 
	         CausticContribution( causticDepth1, causticMap1, smapCoord1 )*NdotL1 : vec4(0.0) );
	         
	// Make sure to normalize based on # of lights enabled to avoid making the scene too bright
	lit /= lightsEnabled;
	
	// Add in small ambient term (use some sort of OpenGL state here??)
	lit += sceneAmbient; // vec4( 0.05 );
	
	// Output the accumulated light intensity modulated by surface color
	gl_FragColor = lit * surfColor;
	gl_FragColor.a = length( gl_TexCoord[5].xyz );
	
}