// A simple deferred shading pass.
//    This takes a fragment position (from the fragPosition texture)
//    a fragment normal (from the fragNormal texture) and the fragment's
//    diffuse albedo (from the fragColor texture) and performs a
//    deferred shading taking into account direct lighting from a single
//    textured spotlight.  A corresponding shadow map is examined to determine
//    which areas are not illuminated from the light.
//
// No indirect illumination is computed here.

// Textures that store the scene's G-buffer (namely fragment position,
//    fragment normal, and surface albedo).
uniform sampler2D       fragPosition, fragNormal, fragColor;

// Textures corresponding to our light in the scene (a textured spotlight
//    and a corresponding shadow map
uniform sampler2D       spotLight;
uniform sampler2DShadow shadowMap0;

// A multiplicative brightness factor for the light intensity, useful if the
//    scene's light is a little too dim (or too bright)
uniform float lightIntensity;

// This function takes in a light-space shadow map coordinate, looks that
//    position up in the shadow map and the light texture, and returns
//    the location's light color if illuminated and black if the location
//    is shadowed.
vec4 IsPointIlluminated( vec4 smapCoord )
{
	return ( all(equal(smapCoord.xyz,clamp(smapCoord.xyz,vec3(0),vec3(1)))) ? 
					shadow2D( shadowMap0, smapCoord.xyz ).x * texture2D( spotLight, smapCoord.xy, 0.0 ): 
					vec4(0.0) );
}

// Main entry into the deferred shading fragment program
void main( void )
{
	// Get our G-buffer data for the current pixel's correponding
	//     geometry, both position (fragPos) and normal (fragNorm)
	vec4 fragPos  = texture2D( fragPosition, gl_TexCoord[0].xy );
	vec4 fragNorm = texture2D( fragNormal, gl_TexCoord[0].xy );
	
	// The fragPosition texture actually includes other data in the
	//    w-component, so our actual eye-space position is here
	vec4 esPos = vec4( fragPos.xyz, 1.0 );
	
	// Compute the direction from our fragment to the light
	vec3 toLight = normalize( gl_LightSource[0].position.xyz - fragPos.xyz );
	
	// Compute the difuse lighting use a simple normal-dot-light direction
	float NdotL = lightIntensity * max( 0.0, dot( normalize( fragNorm.xyz ), toLight ) );
	
	// Compute the shadow map coordinate.  
	vec4 smapCoord = vec4( dot( gl_EyePlaneS[7], esPos ), dot( gl_EyePlaneT[7], esPos ),
						   dot( gl_EyePlaneR[7], esPos ), dot( gl_EyePlaneQ[7], esPos ) );
	smapCoord /= smapCoord.w;
	
	// Make sure the light only projects forwards (not backwards)
	smapCoord.z = smapCoord.z <= 0.0 ? 0.0 : smapCoord.z;
	
	// Determine if this location is lit (look in shadow map).  Modulate by reflectivity
	vec4 lit = IsPointIlluminated( smapCoord ) * NdotL; 
	
	// Modulate the light by the fragment's diffuse albedo and output the result
	gl_FragColor = lit * texture2D( fragColor, gl_TexCoord[0].xy );
	gl_FragColor.a = 1.0;
}