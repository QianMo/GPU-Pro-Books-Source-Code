// This shader determines which texels in the multiresolution buffer are
//    at the correct "refinement level."  Our I3D paper describes an iterative
//    splitting approach to generate "subsplats" which is relatively slow.
//    This stencil approach is a single-pass approach that simultaneously
//    determines which locations in all resolutions of the buffer are 
//    valid "subsplats."  It generates identical results with the iterative
//    approach discussed in our I3D paper.

// The eye-space geometry mipmaps used to determine if the current fragment
//    in the multiresolution buffer is too coarse, too fine, or just the right
//    resolution.  The first is a max-mipmap storing depth derivatives.  The 
//    second is an approximate normal-cone mipmap that stores min and max
//    x&y components of the surface normal.
uniform sampler2D depthMinMax, normMinMax;

// The thresholds for detecting depth and normal discontinuities (which then
//    require additional illumination samples)
uniform float depthThreshold, normThreshold;

// Specifies the maximum mipmap level in our multiresolution buffer (i.e., the
//    coarsest level our texels can get).  A value of 5 or 6 seems to be about
//    (Larger values are simply never used since pixel clusters larger than
//    64x64 rarely occur.  Smaller values limit the fill-rate savings)
uniform float maxMipLevel;

void main( void )
{
	// Get data passed down from program (the current mipmap level and
	//     a [0..1] xy-coordinate inside the mipmap level)
	vec2  mipCoord     = gl_TexCoord[0].xy;
	float fragMipLevel = gl_TexCoord[0].z;
	
	// Compute the depth threhold for the current mipmap level and the next coarsest.
	//    If we used a constant depth threshold, this could simply be passed in from
	//    the user.
	float depthThresh0 = depthThreshold*(3.0/fragMipLevel);
	float depthThresh1 = depthThreshold*(3.0/(fragMipLevel+1));
	
	// All right, look up the depth discontinuties in this fragment (and it's next
	//    coarsest level in the mipmap)
	float depthDeriv0 = texture2DLod( depthMinMax, mipCoord.xy, fragMipLevel ).r;
	float depthDeriv1 = texture2DLod( depthMinMax, mipCoord.xy, fragMipLevel+1 ).r;
	
	// Look up the corresponding normal discontinities.
	vec4  minMaxNorm0 = texture2DLod( normMinMax, mipCoord.xy, fragMipLevel );
	vec4  minMaxNorm1 = texture2DLod( normMinMax, mipCoord.xy, fragMipLevel+1 );
	float normDeriv0 = max( minMaxNorm0.z-minMaxNorm0.x, minMaxNorm0.w-minMaxNorm0.y );
	float normDeriv1 = max( minMaxNorm1.z-minMaxNorm1.x, minMaxNorm1.w-minMaxNorm1.y );
	
	// If the either normal texture value == ( 1, 1, -1, -1 ) then that texel 
	//     is invalid and *must* be subdivided further.  If we encounter an invalid
	//     texel at the finest level, we may *discard* it, since it will not have any
	//     valid illumination.
	bool invalid0 = (normDeriv0 < -1) ? true : false;
	bool invalid1 = (normDeriv1 < -1) ? true : false;
	
	// If we are at the coarsest level of the finest level, we are at boundary cases.
	//    At the corsest level, we must handle the texel either here or at finer
	//    levels).  This means "Check 2" must always fail.  At the finest level,
	//    we must handle the texel either here or at coarser levels.  This means
	//    "Check 1 must always fail.
	bool atCoarsestLevel = ( fragMipLevel+1 > maxMipLevel ) ? true : false;
	bool atFinestLevel   = ( fragMipLevel <= 0.5 ) ? true : false;
	
	// Determine if we need additional refinement in this viscinity.
	bool needRefinment = (depthDeriv0 > depthThresh0) || (normDeriv0 > normThreshold);
	
	// Determine if the coarser level was sufficient.
	bool coarseLevelSufficient = (depthDeriv1 <= depthThresh1) && (normDeriv1 <= normThreshold) && !invalid1;
	
	// Check 1:  Do we need further refinement?  If so, we're not needed.
	if (needRefinment && !atFinestLevel) 
		discard;
		
	// Check 2:  Is the coarser resolution sufficient?  If so, we're not needed.
	if (coarseLevelSufficient && !atCoarsestLevel)
		discard;
		
	// Check 3:  Do we have an invalid texel?  If so it must be refined.  If
	//    we are at the coarsest level, we may safely discard an invalid texel
	//    because it will not contribute valid lighting to the scene (by def.)
	if (invalid0) 
	    discard;

	// OK.  We do not need further refinement and a coarser level does not suffice.
	//    The output color (at this point) does not matter.  We simply need to set the stencil.
	gl_FragColor = vec4( depthDeriv0 ); 
}	


