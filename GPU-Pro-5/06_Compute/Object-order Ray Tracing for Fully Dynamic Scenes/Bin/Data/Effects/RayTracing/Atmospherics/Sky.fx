/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "2.0/Atmospherics/Sky.fx"
#include "Pipelines/Tracing/Ray.fx"
#include "Pipelines/Tracing/Scene.fx"
#include <Utility/Math.fx>

static const uint LightingGroupSize = 512;

[numthreads(LightingGroupSize,1,1)]
void CSMain(uint dispatchIdx : SV_DispatchThreadID)
{
	if (dispatchIdx >= RaySet.RayCount)
		return;

	TracedGeometry geometry = TracedGeometryBuffer[dispatchIdx];

	if (geometry.Depth < MissedRayDepth)
		return;

	RayDesc ray = RayDescriptionBuffer[dispatchIdx];

	float3 direction = normalize(ray.Dir);
	float3 color = EvaluateSkyColor(direction, SkyColor, HazeColor, true, GroundColor);

	TracedLightUAV[dispatchIdx].Color = PackTracedLight( float4(color, 0.0f) );
}

[numthreads(1,1,1)]
void CSTracingGroup(uint dispatchIdx : SV_DispatchThreadID)
{
	GroupDispatchUAV.Store3(
			0,
			uint3( ceil_div(RaySet.RayCount, LightingGroupSize), 1, 1 )
		);
}

technique11 TracingDefault <
	string PipelineStage = "TraceLightingPipelineStage";
	string RenderQueue = "DefaultRenderQueue";
	bool EnableTracing = true;
>
{
	pass Sky < string TracingCSGroupPass = "Group"; >
	{
		// Workaround: W/O VS, input layout creation fails
		SetVertexShader( CompileShader(vs_5_0, VSMain()) );
		SetComputeShader( CompileShader(cs_5_0, CSMain()) );
	}

	pass Group < bool Normal = false; >
	{
		SetComputeShader( CompileShader(cs_5_0, CSTracingGroup()) );
	}
}
