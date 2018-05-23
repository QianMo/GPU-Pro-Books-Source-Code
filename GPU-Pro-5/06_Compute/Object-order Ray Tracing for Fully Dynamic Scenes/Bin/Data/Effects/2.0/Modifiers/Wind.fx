#hookconfig
{
	#redefine BE_RENDERABLE_INCLUDE_WORLD
	#redefine BE_RENDERABLE_INCLUDE_PROJ
}

#include <Engine/Perspective.fx>
#include <Engine/Renderable.fx>

#include <Hooks/Transform.fx>

#hookdef SetupConstants
{
	float AnchorHeight
	<
		String UIName = "Anchor Height";
	> = 10.0f;

	float FreeHeight
	<
		String UIName = "Free Height";
	> = 0.0f;

	float3 Amplitude
	<
		String UIName = "Amplitude";
	> = 0.0f;
}

float3 WindPerturbation(float3 pos)
{
	float randOffset = dot(World[3].xyz, 1.0f);
	float randPeriod = rcp( (AnchorHeight - FreeHeight) / 80.0f + 0.3f * frac(randOffset * 53.0f) );
	float sint, cost;
	sincos(randOffset + randPeriod * Perspective.Time, sint, cost);
	cost = 1.0f - sqrt( 1.0f - pow(sint * 0.5, 2) );
	float sinhy = sinh( clamp( (AnchorHeight - pos.y) * 2.0f * rcp(AnchorHeight - FreeHeight), 0.0f, 2.0f) );

	float detailOffset = dot(normalize(rcp(Amplitude + 0.01f)), pos);

	float detail = 0.02f * sin(randOffset + 6.33f * Perspective.Time - 0.6f * detailOffset);
	detail += 0.03f * sin(randOffset + 5.77f * Perspective.Time - 0.5f * detailOffset);
//	detail += 0.027f * sin(randOffset + 4.11f * Perspective.Time - 0.09f * pos.y);
	detail *= 0.75f + 0.25f * sin(randOffset + 7.11f * Perspective.Time - 0.09f * pos.y);

	return float2(cost, sint + (0.5f + 0.5f * abs(sint)) * detail).yxy * Amplitude * sinhy;
}

float3 WindTransformWorldPosition(float3 pos)
{
	pos.xyz += WindPerturbation(pos);
	return pos;
}

float3 WindTransformWorldNormal(float3 normal, float3 pos)
{
	normal.y += dot(WindPerturbation(pos).xz, normal.xz) * 10.0f * rcp(AnchorHeight - FreeHeight);
	return normal;
}

TransformHookState WindTransform(#hookstate Transform baseState)
{
	TransformHookState state;

	float4 basePos = GetWorldPosition(baseState);
	float3 baseNormal = GetWorldNormal(baseState);

	state.WorldPos = float4(WindTransformWorldPosition(basePos.xyz), basePos.w);
	state.WorldNormal = normalize( WindTransformWorldNormal(baseNormal, basePos.xyz) );

	state.WVPPos = mul(state.WorldPos, Perspective.ViewProj);

	return state;
}

#hookfun Transform(vertex) -> TransformHookState = WindTransform(#hookcall Transform(vertex))
