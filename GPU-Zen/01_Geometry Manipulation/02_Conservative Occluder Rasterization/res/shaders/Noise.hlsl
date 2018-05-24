// noise generators taken from

// Brian Sharpe
// brisharpe CIRCLE_A yahoo DOT com
// http://briansharpe.wordpress.com
// https://github.com/BrianSharpe

void FAST32_hash_3D(
    float3 gridcell,
    float3 v1_mask, // user definable v1 and v2. ( 0's and 1's )
    float3 v2_mask,
    out float4 hash_0,
    out float4 hash_1,
    out float4 hash_2 ) // generates 3 random numbers for each of the 4 3D cell corners. cell corners: v0=0,0,0 v3=1,1,1 the other two are user definable
{
    // gridcell is assumed to be an integer coordinate

    // TODO: these constants need tweaked to find the best possible noise.
    // probably requires some kind of brute force computational searching or something....
    const float2 OFFSET = float2( 50.0, 161.0 );
    const float DOMAIN = 69.0;
    const float3 SOMELARGEFLOATS = float3( 635.298681, 682.357502, 668.926525 );
    const float3 ZINC = float3( 48.500388, 65.294118, 63.934599 );

    // truncate the domain
    gridcell.xyz = gridcell.xyz - floor( gridcell.xyz * ( 1.0 / DOMAIN ) ) * DOMAIN;
    float3 gridcell_inc1 = step( gridcell, float3( DOMAIN - 1.5, DOMAIN - 1.5, DOMAIN - 1.5 ) ) * ( gridcell + 1.0 );

    // compute x*x*y*y for the 4 corners
    float4 P = float4( gridcell.xy, gridcell_inc1.xy ) + OFFSET.xyxy;
    P *= P;
    float4 V1xy_V2xy = lerp( P.xyxy, P.zwzw, float4( v1_mask.xy, v2_mask.xy ) ); // apply mask for v1 and v2
    P = float4( P.x, V1xy_V2xy.xz, P.z ) * float4( P.y, V1xy_V2xy.yw, P.w );

    // get the lowz and highz mods
    float3 lowz_mods = float3( 1.0 / ( SOMELARGEFLOATS.xyz + gridcell.zzz * ZINC.xyz ) );
    float3 highz_mods = float3( 1.0 / ( SOMELARGEFLOATS.xyz + gridcell_inc1.zzz * ZINC.xyz ) );

    // apply mask for v1 and v2 mod values
    v1_mask = ( v1_mask.z < 0.5 ) ? lowz_mods : highz_mods;
    v2_mask = ( v2_mask.z < 0.5 ) ? lowz_mods : highz_mods;

    // compute the final hash
    hash_0 = frac( P * float4( lowz_mods.x, v1_mask.x, v2_mask.x, highz_mods.x ) );
    hash_1 = frac( P * float4( lowz_mods.y, v1_mask.y, v2_mask.y, highz_mods.y ) );
    hash_2 = frac( P * float4( lowz_mods.z, v1_mask.z, v2_mask.z, highz_mods.z ) );
}

void Simplex3D_GetCornerVectors(
    float3 P, // invalue point
    out float3 Pi, // integer grid index for the origin
    out float3 Pi_1, // offsets for the 2nd and 3rd corners. ( the 4th = Pi + 1.0 )
    out float3 Pi_2,
    out float4 v1234_x, // vectors from the 4 corners to the intput point
    out float4 v1234_y,
    out float4 v1234_z )
{
    //
    // Simplex math from Stefan Gustavson's and Ian McEwan's work at...
    // http://github.com/ashima/webgl-noise
    //

    // simplex math constants
    const float SKEWFACTOR = 1.0 / 3.0;
    const float UNSKEWFACTOR = 1.0 / 6.0;
    const float SIMPLEX_CORNER_POS = 0.5;
    const float SIMPLEX_PYRAMID_HEIGHT = 0.70710678118654752440084436210485; // sqrt( 0.5 ) height of simplex pyramid.

    P *= SIMPLEX_PYRAMID_HEIGHT; // scale space so we can have an approx feature size of 1.0 ( optional )

                                 // Find the vectors to the corners of our simplex pyramid
    Pi = floor( P + dot( P, float3( SKEWFACTOR, SKEWFACTOR, SKEWFACTOR ) ) );
    float3 x0 = P - Pi + dot( Pi, float3( UNSKEWFACTOR, UNSKEWFACTOR, UNSKEWFACTOR ) );
    float3 g = step( x0.yzx, x0.xyz );
    float3 l = 1.0 - g;
    Pi_1 = min( g.xyz, l.zxy );
    Pi_2 = max( g.xyz, l.zxy );
    float3 x1 = x0 - Pi_1 + UNSKEWFACTOR;
    float3 x2 = x0 - Pi_2 + SKEWFACTOR;
    float3 x3 = x0 - SIMPLEX_CORNER_POS;

    // pack them into a parallel-friendly arrangement
    v1234_x = float4( x0.x, x1.x, x2.x, x3.x );
    v1234_y = float4( x0.y, x1.y, x2.y, x3.y );
    v1234_z = float4( x0.z, x1.z, x2.z, x3.z );
}

float4 Simplex3D_GetSurfletWeights(
    float4 v1234_x,
    float4 v1234_y,
    float4 v1234_z )
{
    // perlins original implementation uses the surlet falloff formula of (0.6-x*x)^4.
    // This is buggy as it can cause discontinuities along simplex faces. (0.5-x*x)^3 solves this and gives an almost identical curve

    // evaluate surflet. f(x)=(0.5-x*x)^3
    float4 surflet_weights = v1234_x * v1234_x + v1234_y * v1234_y + v1234_z * v1234_z;
    surflet_weights = max( 0.5 - surflet_weights, 0.0 ); // 0.5 here represents the closest distance (squared) of any simplex pyramid corner to any of its planes. ie, SIMPLEX_PYRAMID_HEIGHT^2
    return surflet_weights*surflet_weights*surflet_weights;
}

float SimplexPerlin3D( float3 P )
{
    // calculate the simplex vector and index math
    float3 Pi;
    float3 Pi_1;
    float3 Pi_2;
    float4 v1234_x;
    float4 v1234_y;
    float4 v1234_z;
    Simplex3D_GetCornerVectors( P, Pi, Pi_1, Pi_2, v1234_x, v1234_y, v1234_z );

    // generate the random vectors
    // ( various hashing methods listed in order of speed )
    float4 hash_0;
    float4 hash_1;
    float4 hash_2;
    FAST32_hash_3D( Pi, Pi_1, Pi_2, hash_0, hash_1, hash_2 );
    //SGPP_hash_3D( Pi, Pi_1, Pi_2, hash_0, hash_1, hash_2 );
    hash_0 -= 0.49999;
    hash_1 -= 0.49999;
    hash_2 -= 0.49999;

    // evaluate gradients
    float4 grad_results = rsqrt( hash_0 * hash_0 + hash_1 * hash_1 + hash_2 * hash_2 ) * ( hash_0 * v1234_x + hash_1 * v1234_y + hash_2 * v1234_z );

    // Normalization factor to scale the final result to a strict 1.0->-1.0 range
    // x = sqrt( 0.75 ) * 0.5
    // NF = 1.0 / ( x * ( ( 0.5 ? x*x ) ^ 3 ) * 2.0 )
    // http://briansharpe.wordpress.com/2012/01/13/simplex-noise/#comment-36
    const float FINAL_NORMALIZATION = 37.837227241611314102871574478976;

    // sum with the surflet and return
    return dot( Simplex3D_GetSurfletWeights( v1234_x, v1234_y, v1234_z ), grad_results ) * FINAL_NORMALIZATION;
}