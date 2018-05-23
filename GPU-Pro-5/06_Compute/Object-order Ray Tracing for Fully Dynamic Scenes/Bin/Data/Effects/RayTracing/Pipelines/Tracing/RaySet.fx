#ifndef BE_RAYTRACING_RAY_SET_H
#define BE_RAYTRACING_RAY_SET_H

/// Ray constants.
struct RaySetLayout
{
	uint RayCount;
	uint RayLinkCount;
	uint ActiveRayCount;
	uint MaxRayLinkCount;
};

#ifdef BE_TRACING_SETUP
	cbuffer RaySetConstants : register(b4)
#else
	cbuffer prebound(RaySetConstants) : register(b4)
#endif
{
	RaySetLayout RaySet;
}

#endif