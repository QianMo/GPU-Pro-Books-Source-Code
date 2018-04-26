// This is a simple shader that looks up our VPL samples
//    in our original full-screen RSM and stores them in
//    a compact texture with resolution #samples x 3.
//    (The 3 is because we have 3 render targets in our RSM).


// Here's the three render targets from our reflective shadow map.
//    One stores the light-space fragment position.  One stores the
//    light-space fragment normal.  One stores the light-space
//    fragment albedo.
uniform sampler2D lightPosition, lightNormal, lightColor;

// Because we're using a spotlight (not a point light), we may need
//    to modulate the fragment color by the spotlight color to get
//    the reflective flux (which we want to store in our compact texture)
uniform sampler2D spotTex;

// We're using a regular sampling of the RSM here, and this offset controls
//    how frequently we sample the RSM.
uniform vec3 vplCountOffsetDelta;

// In order to avoid high-frequency variations in scene color that might
//    occur in the full-resolution RSM, we'll access a mipmapped color
//    coordinate to blur some of these out.  After all, with only a few
//    (e.g., 256) VPL samples we don't expect to get high frequency 
//    illumination anyways.
uniform float mipLevel;

// The main entry for our VPL compaction shader.
void main( void )
{  
	// Which VPL are we currently processing?  That's passed down as a texture coordinate,
	//    because each column in our output texture represents a single VPL.
	float vplID = gl_TexCoord[0].x;
	
	// Find the location of this sample in a regular grid array of VPLs and transform
	//    this to a [0..1] uv coordinate we can use to lookup the VPL in the RSM.
	float i = vplCountOffsetDelta.x*fract(vplID/vplCountOffsetDelta.x);
	float j = floor(vplID/vplCountOffsetDelta.x);
	vec2 RSMcoord = vplCountOffsetDelta.z*vec2( i,j ) + vec2( vplCountOffsetDelta.y );

	// Grab the VPL's position from the position texture.  Before storing the VPL to
	//    texture, transform it from light-space to eye-space.  Doing this here in the
	//    compaction shader allows us to save lots of duplicate transformations when
	//    we use this VPL for illumination.  The w-component of the position is the
	//    distance from original light to this VPL.  Since we need that distance
	//    squared in the illumination shaders, compute that here.
	vec4 lightSpacePosition = texture2D( lightPosition, RSMcoord );
	lightSpacePosition.xyz  = (gl_ModelViewMatrix * vec4( lightSpacePosition.xyz, 1.0 )).xyz; 
	lightSpacePosition.w    = lightSpacePosition.w * lightSpacePosition.w;
	
	// Grab the VPL's normal from the position texture.  Before storing the VPL to
	//    texture, transform it from light-space to eye-space.  Doing this here in the
	//    compaction shader allows us to save lots of duplicate transformations when
	//    we use this VPL for illumination.  The w-component of the position is the
	//    distance from original light to this VPL.  Since we need that distance
	//    squared in the illumination shaders, compute that here.  Note this is
	//    duplicated in both the position and normal texels, and many shaders
	//    need this distance (squared) but not all need both the light position and
	//    normal.  (i.e., this duplication allows fewer texture accesses later)
	vec4 lightSpaceNormal   = texture2D( lightNormal, RSMcoord );
	lightSpaceNormal.xyz    = gl_NormalMatrix * lightSpaceNormal.xyz;
	lightSpaceNormal.w      = lightSpaceNormal.w * lightSpaceNormal.w;
	
	// Grab the VPL's surface albedo from the texture, modulate it by the spotlight
	//    color at the corresponding location.  Make sure to use the appropriate mipmap
	//    level to blur high frequencies.  Multiply the result for the VPL's reflected flux.
	vec4 lightSpaceColor    = texture2DLod( lightColor, RSMcoord, mipLevel );        
	lightSpaceColor *= texture2D( spotTex, RSMcoord )* max(0.0,lightSpaceColor.a);

	// Choose (based upon which texel we are -- y location 0, 1, or 2) which of the 
	//    three VPL values (position, normal, or color) should be output to this texel.
	gl_FragColor = vec4( lightSpaceColor.xyz, 1.0 );
	if ( gl_TexCoord[0].y > 1.5 )
		gl_FragColor = lightSpaceNormal;
	else if ( gl_TexCoord[0].y > 0.5 )
		gl_FragColor = lightSpacePosition;
}