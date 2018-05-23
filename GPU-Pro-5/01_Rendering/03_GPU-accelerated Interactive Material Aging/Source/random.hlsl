#ifndef __RANDOM__
#define __RANDOM__

static const float PI = 3.14159265f;

// Linear congruence generator.
// Generate random unsigned int in [0, 2^24)
uint lcg(inout uint prev)
{
	uint LCG_A = 1664525u;
	uint LCG_C = 1013904223u;
	prev = (LCG_A * prev + LCG_C);
	return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
float rnd(inout uint prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

float2 rnd2(inout uint prev)
{
	float2 value;
	value.x = rnd(prev);
	value.y = rnd(prev);
	return value;
}

// Create ONB from normal.  Resulting W is Parallel to normal
void createONB( float3 n, out float3 U, out float3 V, out float3 W )
{
	W = normalize( n );
	U = cross( W, float3( 0.0f, 1.0f, 0.0f ) );
	if ( abs( U.x) < 0.001f && abs( U.y ) < 0.001f && abs( U.z ) < 0.001f  )
		U = cross( W, float3( 1.0f, 0.0f, 0.0f ) );
	U = normalize( U );
	V = cross( W, U );
}


// Sample Phong lobe relative to U, V, W frame
float3 samplePhongLobe( float exponent, float3 n, inout uint seed )
{
	float3 U,V,W;
	createONB(n, U, V, W);
	float sample_x = rnd(seed);
	float sample_y = rnd(seed);
	const float power = exp( log(sample_y)/(exponent+1.0f) );
	const float phi = sample_x * 2.0f * PI;
	const float scale = sqrt(1.0f - power*power);
  
	const float x = cos(phi)*scale;
	const float y = sin(phi)*scale;
	const float z = power;

	return x*U + y*V + z*W;
}

// Sample Phong lobe relative to U, V, W frame and checks that result is in upper hemisphere of normal.
float3 samplePhongLobeSafe( float exponent, float3 dir, float3 n, inout uint seed )
{
	float3 U,V,W;
	createONB(dir, U, V, W);
	float3 res;
	int attempts = 0;
	float cos_Phi_i = -1;
	[loop] while (cos_Phi_i < 0 && attempts < 24)
	{
		float sample_x = rnd(seed);
		float sample_y = rnd(seed);
		const float power = exp( log(sample_y)/(exponent+1.0f) );
		const float phi = sample_x * 2.0f * PI;
		const float scale = sqrt(1.0f - power*power);
  
		const float x = cos(phi)*scale;
		const float y = sin(phi)*scale;
		const float z = power;

		res = x*U + y*V + z*W;
		attempts++;
		cos_Phi_i = dot(res, n);
	}
	if (attempts == 24)
		return float3(0,0,0);
	return res;
}


#endif