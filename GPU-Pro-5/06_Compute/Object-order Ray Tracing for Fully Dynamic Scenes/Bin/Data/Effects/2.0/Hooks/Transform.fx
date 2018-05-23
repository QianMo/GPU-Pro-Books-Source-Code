#ifndef BE_HOOK_TRANSFORM
#define BE_HOOK_TRANSFORM

//// For public re-use ////

/// Basic transform hook state.
struct TransformHookState
{
	float4 WorldPos;
	float3 WorldNormal;
	float4 WVPPos;
};

/// Gets the world-spaec position from the given hook state.
float4 GetWorldPosition(TransformHookState s) { return s.WorldPos; }
/// Gets the projection-space position from the given hook state.
float4 GetWVPPosition(TransformHookState s) { return s.WVPPos; }
/// Gets the world-space normal from the given hook state.
float3 GetWorldNormal(TransformHookState s) { return s.WorldNormal; }

/// Minimal transform hook vertex.
struct TransformHookVertex
{
	float4 Position;
	float3 Normal;
};

/// Allows for position-only hook usage.
TransformHookVertex TransformHookPositionOnly(float4 position)
{
	TransformHookVertex v = { position, float3(0.0f, 0.0f, 1.0f) };
	return v;
}

//// Default transform hook ////

#ifdef _HOOKED

#include <Engine/Perspective.fx>
#include <Engine/Renderable.fx>

#if !defined(BE_RENDERABLE_INCLUDE_WORLD) || !defined(BE_RENDERABLE_INCLUDE_PROJ)

struct DefaultTransformHookState
{
#ifdef BE_RENDERABLE_INCLUDE_WORLD
	float4 WorldPos;
	float3 WorldNormal;
#endif
#ifdef BE_RENDERABLE_INCLUDE_PROJ
	float4 WVPPos;
#endif
};

#ifdef BE_RENDERABLE_INCLUDE_WORLD
	float4 GetWorldPosition(DefaultTransformHookState s) { return s.WorldPos; }
	float3 GetWorldNormal(DefaultTransformHookState s) { return s.WorldNormal; }
#endif
#ifdef BE_RENDERABLE_INCLUDE_PROJ
	float4 GetWVPPosition(DefaultTransformHookState s) { return s.WVPPos; }
#endif

#else

struct DefaultTransformHookState : TransformHookState { };

#endif

DefaultTransformHookState DefaultTransformHook(float4 pos, float3 normal)
{
	DefaultTransformHookState state;

#ifdef BE_RENDERABLE_INCLUDE_WORLD
	state.WorldPos = mul(pos, World);
	state.WorldNormal = normalize( mul((float3x3) WorldInverse, normal) );
#endif
#ifdef BE_RENDERABLE_INCLUDE_PROJ
	state.WVPPos = mul(pos, WorldViewProj);
#endif

	return state;
}

#hookfun Transform(vertex) -> DefaultTransformHookState = DefaultTransformHook(vertex.Position, vertex.Normal)

#endif

#endif