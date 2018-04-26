// We're doing a deferred illumination here, where drawing a 
//    single full-screen quad.  At each fragment on the full screen
//    quad, we look up the g-buffer data for the location and
//    illumination that surface from all of our sampled VPLs.
//    This is blended with the direct illumination that has
//    already been computed and rendered into the framebuffer.

// These are the three buffers in our G-buffer, that store the
//     eye-space fragment position, normal, and surface albedo
//     for all the surfaces visible from the eye.
uniform sampler2D fragPosition, fragNormal, fragColor;

// A compact list of the sampled VPL locations we'll use to 
//    illuminate the scene
uniform sampler2D vplCompact;

// The light intensity allows us to painlessly increase or 
//     decrease light intensity by a multiplicative factor, which
//     is useful of the scene-specified intensity is too dim or
//     too bright.
uniform float lightIntensity;

// A count of the number of VPLs used in this accumulate pass.
//     This is important, because we're using a point-to-disk
//     form factor (i.e., radiosity) approximation to compute
//     physically accurate light reflectance.  This depends on
//     a measure of solid angle, which varies based upon how many
//     VPLs we use.
uniform float lightSamples;

// This offset is used to move from one VPL in the vplCompact
//     texture to another.  Initially our vplCompactTex had
//     exactly the width of the #VPLs.  Currently, it has a width
//     of 4096 allowing a run-time changable # of VPLs, thus
//     offset is 1/4096.  (Otherwise it would be 1/#VPLs). 
//     If: offset = 1/#VPLs then maxKValue=1.0;
//     If: offset = 1/4096  then maxKValue=#VPLs/4096;
uniform float offset, maxKValue;

// The main entry point for our VPL gather deferred rendering pass
//   In this code, subscripts of "i" represent the eye-space fragment
//   and subscripts of "j" represent the light-space fragment (i.e., the VPL)
void main( void )
{
	// Get data for the current fragment that we're gathering illumination for.
	//   Look this up in the G-buffer, getting fragment position and normal.
	vec4 fragPos  = texture2D( fragPosition, gl_TexCoord[0].xy );
	vec4 fragNorm = texture2D( fragNormal, gl_TexCoord[0].xy );
	vec3 color = vec3( 0.0 );	
	vec3 norm_i = normalize( fragNorm.xyz );
	
	// Now march through all of the VPLs one at a time.
	for (float k=0; k<maxKValue; k+=offset)
	   {
			// Grab the location, surface normal, and reflected flux of this VPL
			vec4 lightPos = vec4( texture2D( vplCompact, vec2( k, 0.5 ) ).xyz, 1.0 );
			vec4 norm_j   = texture2D( vplCompact, vec2( k, 1.0 ) );
			vec3 color_j = texture2D( vplCompact, vec2( k, 0.0 ) ).xyz;
			
			// Compute the direction between our current fragment and this VPL
			//    We also need the square of the distance between the two for our
			//    form factor computation.  We enforce a minimum bounds on the 
			//    distance between these two to avoid sigularities that send light 
			//    intensity to infinity
			vec3 toVPL = lightPos.xyz - fragPos.xyz;
			float distToVPLSqr = max( dot( toVPL, toVPL ), 0.2 );
			toVPL = normalize( toVPL );
			
			// Get the distance between the original light and this (jth) VPL.
			//    This has been precomputed and stored in the vplCompactTex as the
			//    w-component of the surface normal.
			float len_Lj_sqr = norm_j.a;
			
			// Compute the angles between the view direction, vector between VPL & fragment,
			//    and light direction.
			float NiDotT = max( 0.0, dot( norm_i, toVPL ) );
			float NjDotT = max( 0.0, dot( norm_j.xyz, -toVPL ) );
			
			// Compute a point-to-disk form factor approximation that determine
			//    how important this VPL's contribution is to this particular
			//    fragment.  This is Equation (1) in our EGSR 2009 paper.
			float numer = len_Lj_sqr * NiDotT * NjDotT; 
			float denom = 1.5 * lightSamples * distToVPLSqr + len_Lj_sqr;
			
			// Add the color contributed by this VPL to our running total.	
			color += (numer / denom) * color_j.xyz;
	   }
	
	// Modulate our total indirect illumination by the current fragment's
	//    surface albedo.  If we've artificially increased or decreased the
	//    light intensity by a multiplicative factor, take that into account.
	vec4 color_i = texture2D( fragColor, gl_TexCoord[0].xy );
	color = (lightIntensity) * color_i.xyz * color;   
	
	// Output the final computed indirect illumination
	gl_FragColor.xyz = color;
	gl_FragColor.a = 1.0;
}