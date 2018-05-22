
uniform sampler2D earthcolor;
uniform sampler2D oceanmask;

float zoom = 2.00;							// camera focal length

// image postprocess
float exposure = 1.00;						// global image brightness
float saturation = 1.25;					// color saturation control 

// basic parameters
float R = 1.;								// planet radius
float top = 1.011;							// atmosphere top (up to 70 km)
float H = 0.00119;							// density scale-height of atmosphere (not pressure scale height)
vec3 L = vec3( -1, 0, 0 ); 					// light direction (sun)

// Rayleigh extinction coefficients, 
// calculated for a nitrogen gas (polarizability 1.82e-30)
// with temp 290 K, press 101300 Pa, wavelengths 655 (R), 540 (G) and 425 (B) nm,
// and a scale of 1 unit = 6400 km (earth radius).
vec3 beta = vec3( 38.05, 82.36, 214.65 );	
												
// a small absobtion in the orange band is added
// to simulate the effect of ozone
vec3 absorb = vec3( 0.75, 0.85, 1. );			

// devrived values
float LN2 = 0.693147181;
float H50 = H * LN2;
vec3 beta50 = beta * LN2;
float invH50 = 1. / H50;
float X50 = R / H50;

// --------------------------------------------------------

vec4 point_on_surface( vec3 campos, vec3 viewdir )
{
	// intersect a point on an implicit surface
	// this is just an ad-hoc way to add terrain geometry
	// it has nothing to do with the atmospheric scattering itself

	vec4 result = vec4(0);

	float r = R;
	float d = dot( campos, viewdir );

	float d2 = d * d;
	float x2 = dot( campos, campos );
	float r2 = r * r;
	float D = d2 - x2 + r2;

	if( D >= 0. )
	{
		if( x2 > r2 )
		{
			float t = - d - sqrt(D);
			if( t > 0. )
				result = vec4( campos + viewdir * t, 1 );
		}
		else
			result = vec4( campos, 1 );
	}

	return result;
}

// --------------------------------------------------------

float chapman( float X, float h, float coschi )
{
	// this is the approximate Chapman function,
	// corrected for transitive consistency 

	float c = sqrt( X + h );    
    
	if( coschi >= 0. )
	{	
 		return c / ( c * coschi + 1. ) * exp2( -h );
	}
	else
	{
		float x0 = sqrt( 1. - coschi * coschi ) * ( X + h );
		float c0 = sqrt( x0 );    
		return 2. * c0 * exp2( X - x0 ) - c / ( 1. - c * coschi ) * exp2( -h );
	}
}

// --------------------------------------------------------

vec3 transmittance( vec3 r, vec3 viewdir )
{
	// a quick function to get the transmittance
	// looking from point p into infinity

	float rsq = dot(r,r);
	float invrl = inversesqrt( rsq );
	float len = rsq * invrl;
	float x = len * invH50;
	float h = x - X50;
	float coschi = dot( r, viewdir ) * invrl;
	
	return exp2( - beta50 * H50 * chapman( X50, h, coschi ) );
}

// --------------------------------------------------------

void aerial_perspective( 
	out vec3 resultT,
	out vec3 resultS,
	in vec3 r0,
	in vec3 r1,
	in bool infinite )
{
	// compute the full aerial perspective
	// from point r0 to point r1, 
	// all positions relative to the planet center
	// if the infinite flag is true, the ray is followed beyond r1 into open space

	resultT = vec3(1);
	resultS = vec3(0);

	// get the principal integration bounds
	// as the intersection of the viewing ray
	// with the atmosphere boundary
	// t0 = start, t1 = end

	float t0 = 0.;
	float t1 = 0.;
	vec3 dr = r1 - r0;
	vec3 drn = normalize(dr);

	{
		float dp = dot( r0, drn );
		float r0sq = dot( r0, r0 );
		float top2 = top * top;
		float D = dp * dp - r0sq + top2;

		if( D >= 0. )
		{
			t0 = max( 0., - dp - sqrt(D) );
			t1 = - dp + sqrt(D);
		}
		else
			return;
	}

	// the infinite flag means the viewing ray extends
	// beyond the point of x1 (into space)
	// otherwise it ends at x1 (onto ground)

	float infneg = 1.;
	if( !infinite )
	{
		t0 = min( t0, length(dr) );
		t1 = min( t1, length(dr) );
		infneg = -1.;
	}

	// initialization of
	// the integration loop variables

	const int NMAX = 16;
	int N = NMAX; 
	float range = t1 - t0;
	float dt = range / float(N-1);

	float last_amtl = 0.; 
	float last_mray = 0.;

	{
		vec3 r = r0 + drn * t1;
		float rl = length(r);
		float x = rl * invH50;
		float h = x - X50;
		float coschi_sun = dot( r / rl, L );
		float coschi_ray = dot( r / rl, infneg * drn );
		
		last_amtl = H50 * chapman( X50, h, coschi_sun );
		last_mray = infneg * H50 * chapman( X50, h, coschi_ray );
	}

	float costheta = dot( drn, L );
	float phase = 3. / 4. * ( 1. + costheta * costheta );

	// main loop
	// integrate along the ray in reverse order
	// (back to front)

	for( int i = N - 1; i-- > 0; )
	{
		// calculate altitude r along the ray
		float t = float(i) * dt + t0;
		vec3 r = r0 + drn * t;
		float rl = length(r);

		// normalize altitude to 50%-height
		float x = rl * invH50;
		float h = x - X50;

		// calculate local incidence angle with the sunlight
		float coschi_sun = dot( r / rl, L );
		float coschi_ray = dot( r / rl, infneg * drn );

		// calculate the airmass along this segment of the ray
		float mray = infneg * H50 * chapman( X50, h, coschi_ray );
		float msegment = mray - last_mray;

		// calculate inscatter for this segment
		// amtl = airmass to light
		// for simplicity, the sun irradiance is assumed to be 1
		// so this is just the transmittance towards the sun, Tsun
		// for a Rayleigh atmosphere
		float amtl = H50 * chapman( X50, h, coschi_sun );
		vec3 segmentS = phase * exp2( -beta50 * ( amtl + last_amtl ) * .5 );

		// calculate the transmittance for this segment
		vec3 segmentT = exp2( -beta50 * msegment );
		
		// propagate the integration
		// previous inscatter is attenuated by current transmittance, plus the new inscatter
		// previous transmittance is attenuated by current transmittance
		resultS = resultS * segmentT + ( 1. - segmentT ) * segmentS;
		resultT = resultT * segmentT;

		// keep these variables for the next iteration
		last_amtl = amtl;
		last_mray = mray;
	}

	// Factored the average absorbtion color out of the loop
	// This would not be possible with different absorbtion colors at different scale heights
	resultS *= absorb;
}

// --------------------------------------------------------

void main()
{
	vec3 color = vec3(0);
	vec3 vertex = gl_TexCoord[0].xyz;
	vec3 campos = gl_TexCoord[1].xyz;
	vec3 zaxis = gl_TexCoord[2].xyz;
	vec3 viewdir = vertex - campos;
	viewdir = normalize( viewdir + ( zoom - 1. ) * zaxis * dot( viewdir, zaxis ) );

	// raycast a point to the planet surface
	vec4 P = point_on_surface( campos, viewdir );

	if( P.w == 0. )
	{
		// there was no surface intersection
		// this is simply the aerial perspective of empty space
		vec3 T, S;
		aerial_perspective( T, S, campos, campos + viewdir, true );
		color = S;
	}
	else
	{
		// we have hit a surface point
		// get surfance normal and generate spherical texture coordinates
		vec3 N = normalize( P.xyz );
		vec2 uv = vec2(.5) + vec2( .15915494, .31830989 ) * vec2( atan( N.z, -N.x ), asin( -N.y ) );

		// sample the surface texture and 
		// apply inverse gamma 2.2 conversion to the color
		vec3 surfacecolor = texture2D( earthcolor, uv ).xyz;		surfacecolor = pow( surfacecolor, vec3( 2.2 ) );

		// get the direct light color
		// from the transmittance of the sun through the atmosphere
		vec3 lightcolor = transmittance( P.xyz, L );

		// for shading the landmass we use the Lommel-Seeliger law
		vec3 V = -viewdir;
		float dotNV = max( 0., dot( N, V ) );
		float dotNL = max( 0., dot( N, L ) );
		vec3 landcolor = lightcolor * surfacecolor * dotNL / (dotNL + dotNV);

		// for shading the ocean 
		// we obtain the skycolor reflection
		// via the aerial perspective of the reflection vector,
		// and mix it with an approximate fresnel factor
		vec3 T,S;
		aerial_perspective( T, S, P.xyz, P.xyz - reflect( V, N ), true );
		float fresnel = 1. - dotNV;
		fresnel *= fresnel;
		fresnel *= fresnel;
		vec3 oceancolor = mix( landcolor, S, fresnel );

		// we also add the specular reflection of the sun to the ocean
		// uses the micro-facet shading model described in ShaderX7 
		vec3 H = normalize( L + V );
		float dotNH = max( 0., dot( N, H ) );
		float dotLH = max( 0., dot( L, H ) );
		oceancolor += lightcolor * 0.02 * ( 32. + 1. ) / ( 8. ) * pow( dotNH, 32. ) / pow( dotLH, 3. ) * dotNL;

		// final shaded surface color
		surfacecolor = texture2D( oceanmask, uv ).x > 0.45 ? oceancolor : landcolor;

		// finally obtain the aerial perspective of the shaded surface point
		// from the camera's point of view
		aerial_perspective( T, S, campos, P.xyz, false );
		color = S + T * surfacecolor;
	}

	// final tone mapping
	color = 1. - exp( - exposure * color );

	// saturation control
	color = mix( vec3( dot( vec3( .26, .64, .12 ), color ) ), color, saturation );

	// gamma 2.2 conversion and output
	gl_FragColor = vec4( pow( color, vec3( .4545 ) ), 1 );
}
