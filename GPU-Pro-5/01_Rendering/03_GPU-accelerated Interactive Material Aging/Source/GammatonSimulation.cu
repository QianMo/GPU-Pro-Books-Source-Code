#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <random.h>
#include "OptixTypes.h"

using namespace optix;

// Application input.
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(float,         gammaton_epsilon, , ) = 0.001f;
rtDeclareVariable(float,         integration_step_size, , ) = 10.0f;
rtDeclareVariable(float3,        gravity, , ) = {0, -9.81f, 0};
rtDeclareVariable(float,         minimum_velocity, , ) = 0.05f;
rtDeclareVariable(float3,        world_aabb_min, , ) = {-1500, -48.5653877, -1500}; // defaults
rtDeclareVariable(float3,        world_aabb_max, , ) = { 1500,  384.388849f,  1500};

// Intrinsic input.
rtDeclareVariable(GammatonPRD,	prd,		  rtPayload, );
rtDeclareVariable(optix::Ray,	ray,          rtCurrentRay, );
rtDeclareVariable(float,		t_hit,        rtIntersectionDistance, );

// Input from Intersection program.
rtDeclareVariable(float2, texcoord,		attribute texcoord, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

// Buffers.
rtBuffer<GammatonHitOptix, 1>           GammatonHitOptixMap;
rtBuffer<GammatonHitShared, 1>          GammatonHitSharedMap_Out;
rtBuffer<PlantletShared, 1>				PlantletSharedMap;

__device__ static const float PLANT_PROBABILITY = 0.001;
__device__ static const float EPSILON = 0.1;	// numerical epsilon

// checks whether a gammaton is too slow on a nearly horizontal plane
__device__ bool isTooSlow(float3 velocity, float3 world_shading_normal)
{
	float speed = length(velocity);
	return 	speed != speed // NaN check
			|| (  speed < minimum_velocity && abs(world_shading_normal.y) > 0.8  )
			|| speed < abs(gravity.y) * 1.5;
}

// plants with small probability a plantlet.
__device__ void plantletPlacement(uint seed, float3 pnew)
{
	float random = rnd(seed);
	if (random < PLANT_PROBABILITY)
	{
		PlantletShared& plantlet = PlantletSharedMap[prd.ray_index];
		plantlet.Position = pnew;
		plantlet.Dummy = prd.ray_index;
		plantlet.AtlasCoord = texcoord;
	}
}

// invoked when the gammaton ray intersects with geometry.
// - adds gravity to the gammaton
// - stores the position and hit texture coordinate
// - might place a plantlet
RT_PROGRAM void gammaton_closest_hit()
{	
	// Get the data stored for this gammaton (getting a reference, so we have write access too)
	GammatonHitShared& hitshared = GammatonHitSharedMap_Out[prd.ray_index];
	
	// there is a detail we left out in the article: the normal must be transformed to world space.
	float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );

	// get velocity and apply offset to position
	float3 v = ray.direction * prd.speed;				// velocity
	float3 phit = ray.origin + t_hit * ray.direction;
	float3 pnew = phit + world_shading_normal * EPSILON;	// move up by some constant value

	// update the velocity based on the traveled distance (apply gravity)
	float s = length(pnew - ray.origin);
	float dt = s / prd.speed;	// time = way / speed
	v += dt * gravity;

	hitshared.texcoord = texcoord;
	hitshared.velocity = normalize(v) * prd.speed;

	if (isTooSlow(v, world_shading_normal))   // too slow?
		SET_DEAD(hitshared.flags);	              // set inactive
	
	SET_HIT(hitshared.flags);

	// Remember position, since d3d will setup a new direction. The gammaton source will launch from this position.
	GammatonHitOptix& gammaState = GammatonHitOptixMap[prd.ray_index];
	gammaState.position = pnew;

	// plant with probability
	plantletPlacement(hitshared.randomSeed, pnew);
}

// Checks if a particle is outside the simulation domain (world bounding box)
__device__ bool leftDomain(float3 position)
{
	return (world_aabb_min.x > position.x || world_aabb_max.x < position.x ||
			world_aabb_min.y > position.y || world_aabb_max.y < position.y ||
			world_aabb_min.z > position.z || world_aabb_max.z < position.z);
}

// invoked when a gammaton ray does not intersect with geometry.
// - adds gravity to the gammaton 
// - if the recursion depth is reached
//      stores the gammaton as midair 
//   else launches recursively a new ray
RT_PROGRAM void gammaton_miss()
{
	// compute position of gammaton
	float dt = integration_step_size;
	float3 vold = ray.direction * prd.speed;

	// apply gravity force
	float3 pnew = ray.origin + dt * vold;
	float3 vnew = vold + dt * gravity;

	GammatonHitShared& hitshared = GammatonHitSharedMap_Out[prd.ray_index];
	
	if (leftDomain(pnew)) {   // if outside bounding box
		SET_DEAD(hitshared.flags);
		return;
	}
	
	// Floating particle moved over edge.
	if (IS_FLOAT(hitshared.flags))	{  // float
		vnew = make_float3(0,0,0);     // let gammaton fall
		SET_BOUNCE(hitshared.flags);   // free fall
	}
	
	// gammaton still alive after maximum depth
	prd.ray_depth++;
	if (prd.ray_depth >= MAX_GAMMATON_DEPTH) {
		GammatonHitOptixMap[prd.ray_index].position = pnew;
		hitshared.velocity = vnew;
		SET_ALIVE(hitshared.flags);
		SET_MIDAIR(hitshared.flags);
		return;
	}	

	// launch next ray
	prd.speed = length(vnew);
	float maxDist = dt * prd.speed;
	optix::Ray ray = optix::Ray(pnew, normalize(vnew), RayType_Gammaton, gammaton_epsilon, maxDist);			
	rtTrace( top_object, ray, prd );
}
