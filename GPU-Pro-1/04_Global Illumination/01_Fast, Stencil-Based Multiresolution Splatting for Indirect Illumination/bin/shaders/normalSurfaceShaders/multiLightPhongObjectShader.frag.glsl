#version 120 
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D       spotLight, causticMap0, causticMap1;
uniform sampler2DShadow shadowMap0, shadowMap1, causticDepth0, causticDepth1;
uniform float			useShadowMap, useCausticMap;

uniform vec4 amb, dif, spec, shiny;
uniform float lightIntensity;
uniform float lightsEnabled;
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
	vec3 norm = normalize( gl_TexCoord[6].xyz );
	
	// Compute N dot L for the light(s)
	vec3 toLight0   = normalize( gl_LightSource[0].position.xyz - gl_TexCoord[5].xyz );
	vec3 half0      = normalize( toLight0 + normalize( -gl_TexCoord[5].xyz ) );
	float NdotL0    = max( 0.0, dot( norm, toLight0 ) );
	float specMult0 = max( 0.0, pow(dot( norm, half0 ),shiny.x) );
	vec3 toLight1   = normalize( gl_LightSource[1].position.xyz - gl_TexCoord[5].xyz );
	vec3 half1      = normalize( toLight1 + normalize( -gl_TexCoord[5].xyz ) );
	float NdotL1    = max( 0.0, dot( norm, toLight1 ) );
	float specMult1 = max( 0.0, pow(dot( norm, half1 ),shiny.x) );
	
	// Compute the shadow map coordinate.  This is wasted if not using shadow/caustic mapping.
	//   However, it's relatively cheap to compute.
	vec4 smapCoord0 = vec4( dot( gl_EyePlaneS[7], gl_TexCoord[5] ), dot( gl_EyePlaneT[7], gl_TexCoord[5] ),
						   dot( gl_EyePlaneR[7], gl_TexCoord[5] ), dot( gl_EyePlaneQ[7], gl_TexCoord[5] ) );
	smapCoord0 /= smapCoord0.w;
	smapCoord0.z = smapCoord0.z <= 0.0 ? 0.0 : smapCoord0.z;
	vec4 smapCoord1 = vec4( dot( gl_EyePlaneS[6], gl_TexCoord[5] ), dot( gl_EyePlaneT[6], gl_TexCoord[5] ),
						   dot( gl_EyePlaneR[6], gl_TexCoord[5] ), dot( gl_EyePlaneQ[6], gl_TexCoord[5] ) );
	smapCoord1 /= smapCoord1.w;
	smapCoord1.z = smapCoord1.z <= 0.0 ? 0.0 : smapCoord1.z;
		
	// Now accumulate illumination. 
	
	// Add in direct lighting
	vec4 lit0 = ( (lightsEnabled > 0.1 && useShadowMap > 0) ? 
		     IsPointIlluminated( shadowMap0, smapCoord0 ) : vec4(1.0) );
	vec4 lit1 = ( (lightsEnabled > 1.1 && useShadowMap > 0) ? 
	         IsPointIlluminated( shadowMap1, smapCoord1 ) : 
	         (lightsEnabled < 1.1 ? vec4(0.0) : vec4(1.0)) );
	
	// Add in caustic map lighting
	lit0 += ( (lightsEnabled > 0.1 && useCausticMap > 0) ? 
	         CausticContribution( causticDepth0, causticMap0, smapCoord0 ) : vec4(0.0) );
	lit1 += ( (lightsEnabled > 1.1 && useCausticMap > 0) ? 
	         CausticContribution( causticDepth1, causticMap1, smapCoord1 ) : vec4(0.0) );
	
	// Make sure the total light intensity is constant, no matter how many lights are used
	lit0 /= lightsEnabled;
	lit1 /= lightsEnabled;
	
	// Output the final, accumulated color
	gl_FragColor = 0.5*(sceneAmbient + amb) + lightIntensity * ( lit0 * (dif*NdotL0 + spec*specMult0) +
	                                        lit1 * (dif*NdotL1 + spec*specMult1) );
	gl_FragColor.a = length( gl_TexCoord[5].xyz );
	
}