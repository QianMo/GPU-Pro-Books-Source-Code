#include <optix_world.h>
#include <random.h>
#include "OptixTypes.h"

using namespace optix;

// Application input.
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(float,         gammaton_epsilon, , ) = 0.001f;
rtDeclareVariable(float,         integration_step_size, , ) = 10.0f;
rtDeclareVariable(rtObject,      top_object, , );
rtBuffer<uint2, 1>               gammaton_rnd_seeds;
rtBuffer<GammatonHitOptix, 1>           GammatonHitOptixMap;
rtBuffer<GammatonHitShared, 1>          GammatonHitSharedMap_In;
rtBuffer<GammatonHitShared, 1>          GammatonHitSharedMap_Out;
rtDeclareVariable(float3, v1, , ) = {1,0,0};
rtDeclareVariable(float3, v2, , ) = {0,0,1};
rtDeclareVariable(float3, anchor, , ) = {0,0,0};
rtDeclareVariable(uint, RayIndexOffset, , ) = 0;
rtDeclareVariable(float, initial_velocity, , ) = 10.0f;
rtDeclareVariable(uint2, StartMaterial, , ) = {0,0};
rtDeclareVariable(float, LobeExponent, , ) = 100.0f;

// Intrinsic input.
rtDeclareVariable(uint, launch_index, rtLaunchIndex, );

// Generates a random position on a disc.
__device__ float3 generateRandomPosition()
{
	uint2&   seed = gammaton_rnd_seeds[launch_index];
	float2 area_sample = rnd_from_uint2(seed);

	// generate random polar coordinate
	float radius = area_sample.x;	// [0..1]
	float angle = area_sample.y * 2*M_PIf; // [0..2Pi]
	// convert to cartesian space (points in a unit sphere, concentrated at the center)
	float x = radius * cos(angle);
	float y = radius * sin(angle);

	return anchor + v1 * x + v2 * y;
}

// Generates a phong lobe sampled emission direction.
__device__ float3 generateRandomDirection()
{
	uint2&   seed = gammaton_rnd_seeds[launch_index];
	float3 ray_direction = normalize(cross(v2, v1));

	float2 dir_sample = rnd_from_uint2(seed);	 // uses lcg to generate two further random numbers
	return normalize(sample_phong_lobe(dir_sample, LobeExponent, normalize(v1), normalize(v2), ray_direction ));
}

// Entry program that continues a midair gammaton or emits a new gammaton.
RT_PROGRAM void spray_source()
{
	// initialize payload
	GammatonPRD prd;
	prd.ray_index = launch_index + RayIndexOffset;
	prd.ray_depth = 0;	
	
	// get gammaton data (ping pong)
	GammatonHitShared& hitshared_in  = GammatonHitSharedMap_In [prd.ray_index];
	GammatonHitShared& hitshared_out = GammatonHitSharedMap_Out[prd.ray_index];
	hitshared_out.flags = hitshared_in.flags;

	float3 ray_origin, ray_direction;
	// if the gammaton is alive we continue the ray.                  sanity check  -> is NaN ?   (can happen if (0,0,0) velocity got normalized)
	if (IS_ALIVE(hitshared_in.flags) && hitshared_in.velocity.x == hitshared_in.velocity.x)
	{
		// continue from last position
		ray_origin = GammatonHitOptixMap[prd.ray_index].position;
		ray_direction = normalize(hitshared_in.velocity);
		prd.speed = length(hitshared_in.velocity);
		hitshared_out.carriedMaterial = hitshared_in.carriedMaterial;
	}
	else  // else emit a new ray
	{
		ray_origin = generateRandomPosition();
		ray_direction = generateRandomDirection();
		prd.speed = initial_velocity;
		hitshared_out.carriedMaterial = StartMaterial;		
	}

	// Set state
	SET_ALIVE(hitshared_out.flags);
	SET_BOUNCE(hitshared_out.flags);

	hitshared_out.randomSeed = hitshared_in.randomSeed;	// pass through the random seed.
	float maxDist = prd.speed * integration_step_size;	// s = v * t
	optix::Ray ray = optix::Ray(ray_origin, ray_direction, RayType_Gammaton, gammaton_epsilon, maxDist );	
	rtTrace( top_object, ray, prd );
}

RT_PROGRAM void spray_reset()
{
	int ray_index = launch_index + RayIndexOffset;

	// store position in area, only accessable by OptiX
	GammatonHitOptixMap[ray_index].position = generateRandomPosition();

	// store other data in buffer that is shared with D3D
	// "GammatonHitSharedMap_In" is the buffer that is read in the next "area_source" launch
	GammatonHitShared& hitshared = GammatonHitSharedMap_In[ray_index];
	hitshared.velocity = generateRandomDirection() * initial_velocity;
	hitshared.carriedMaterial = StartMaterial;		
	SET_ALIVE(hitshared.flags);
	SET_BOUNCE(hitshared.flags);
	SET_HIT(hitshared.flags);
}
