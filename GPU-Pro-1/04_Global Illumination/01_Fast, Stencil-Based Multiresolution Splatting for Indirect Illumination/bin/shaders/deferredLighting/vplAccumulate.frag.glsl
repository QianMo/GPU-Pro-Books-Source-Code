// We're doing a deferred illumination here, where we're picking 
//    a single virtual point light (passed down from the vertex
//    shader in gl_TexCoord[1,2,3]) and illuminating every fragment 
//    in our scene by this single light.  This is accumulated (i.e., 
//    blended) with the direct illumination that has already been 
//    computed and rendered into the framebuffer.


// These are the three buffers in our G-buffer, that store the
//     eye-space fragment position, normal, and surface albedo
//     for all the surfaces visible from the eye.
uniform sampler2D fragPosition, fragNormal, fragColor;

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


// The main entry point for our VPL accumulation deferred rendering pass
//   In this code, subscripts of "i" represent the eye-space fragment
//   and subscripts of "j" represent the light-space fragment (i.e., the VPL)
void main( void )
{
	// Grab data passed down from the vertex shader about the current VPL
	vec4 lightPos   = gl_TexCoord[1]; // Current VPL position
	vec4 norm_j     = gl_TexCoord[2]; // Current VPL surface normal 
	vec4 color_j    = gl_TexCoord[3]; // Current VPL color/reflected flux
	
	// Look up in the deferred render buffers (aka G-buffers) the current
	//     fragment's position and surface normal.
	vec4 fragPos  = texture2D( fragPosition, gl_TexCoord[0].xy );
	vec4 fragNorm = texture2D( fragNormal, gl_TexCoord[0].xy );

	// Compute a direction from the fragment to the "light" (our VPL)
	//   Also remember the distance between the two (needed for our
	//   point-to-disk form factor approximation, below).  We enforce 
	//   a minimum bounds on the distance between these two to avoid 
	//   sigularities that send light intensity to infinity
	vec3 toVPL = lightPos.xyz - fragPos.xyz;
	float distToVPLSqr = max( dot( toVPL, toVPL ), 0.2 );
	toVPL = normalize( toVPL );
	
	// Get the distance (squared) between the original light and the
	//    current VPL.  This has been precomputed and stored in the 
	//    w-component of the VPL's surface normal.
	float len_Lj_sqr = gl_TexCoord[2].a;
	
	// Unitize the fragment normal.
	vec3 norm_i = normalize( fragNorm.xyz );
	
	// Compute the angles (or really dot products) between the viewing
	//   ray and VPL direction and the direction from the original light.
	float NiDotT = max( 0.0, dot( norm_i, toVPL ) );
	float NjDotT = max( 0.0, dot( norm_j.xyz, -toVPL ) );
	
	// Look up the surface albedo at the eye-space fragment
	vec4 color_i = texture2D( fragColor, gl_TexCoord[0].xy );
	
	// Compute a point-to-disk form factor approximation that determine
	//    how important this VPL's contribution is to this particular
	//    fragment.  This is Equation (1) in our EGSR 2009 paper.
	float numer = len_Lj_sqr * NiDotT * NjDotT;
	float denom = 1.5 * lightSamples * distToVPLSqr + len_Lj_sqr;
		
	// We can then compute the final color by multiplying the reflected VPL
	//    flux with the form factor and modulating by the surface albedo.
	vec3 finalColor = ((lightIntensity * numer) / denom) * color_i.xyz * color_j.xyz;
	
	gl_FragColor.xyz = finalColor;
	gl_FragColor.a = 1.0;
}