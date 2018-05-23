#include "Pipelines/LPR/Pipeline.fx"

#define BE_TRACING_SETUP

#include "Pipelines/Tracing/Scene.fx"

/// Scene approximation stage
PipelineStage VoxelRepPipelineStage
<
	int Layer = 40;
	bool Normal = false;
	string Setup = "TracingPipelineSetup";
>;

/// Ray tracing pipeline stage
PipelineStage TracingPipelineStage
<
	int Layer = 50;
	bool Normal = false;
	string Setup = "TracingPipelineSetup";
>;

/// Alternative ray tracing stage that allows for experimentation in isolated steps w/o breaking the overall tracing flow
PipelineStage PartialTracingPipelineStage
<
	int Layer = 50;
	bool Normal = false;
	string Setup = "TracingPipelineSetup";
>;

/// Shading of the ray tracing results
PipelineStage TraceLightingPipelineStage
<
	int Layer = 60;
	bool Normal = false;
	string Setup = "TracingPipelineSetup";
>;

// The rest is bound manually by the tracing pipeline
static const String PSTracingResources[] = { "PerspectiveConstants" };

/// Read-only depth-stencil state.
DepthStencilState DisableDepthStencilState
{
	DepthEnable = False;
};

// Setup of stages
technique11 TracingPipelineSetup <
	bool EnableProcessing = true;
>
{
	pass <
		string PipelineStage = "VoxelRepPipelineStage";
		
		string VSBindResources[] = VSDefaultResources;
		string GSBindResources[] = GSDefaultResources;
		string PSBindResources[] = PSTracingResources;
		string CSBindResources[] = PSTracingResources;
	>
	{
		SetRasterizerState( DoublesidedRasterizerState );
		SetDepthStencilState( DisableDepthStencilState, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetHullShader( NULL );
		SetDomainShader( NULL );
		SetGeometryShader( NULL );
	}

	pass <
		string PipelineStage = "TracingPipelineStage";

		string VSBindResources[] = VSDefaultResources;
		string GSBindResources[] = GSDefaultResources;
		string PSBindResources[] = PSTracingResources;
		string CSBindResources[] = PSTracingResources;
	>
	{
		SetRasterizerState( DoublesidedRasterizerState );
		SetDepthStencilState( DisableDepthStencilState, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetHullShader( NULL );
		SetDomainShader( NULL );
		SetGeometryShader( NULL );
	}

	pass <
		string PipelineStage = "PartialTracingPipelineStage";

		string VSBindResources[] = VSDefaultResources;
		string GSBindResources[] = GSDefaultResources;
		string PSBindResources[] = PSTracingResources;
		string CSBindResources[] = PSTracingResources;
	>
	{
		SetRasterizerState( DoublesidedRasterizerState );
		SetDepthStencilState( DisableDepthStencilState, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetHullShader( NULL );
		SetDomainShader( NULL );
		SetGeometryShader( NULL );
	}

	pass <
		string PipelineStage = "TraceLightingPipelineStage";
		
		string VSBindResources[] = VSDefaultResources;
		string GSBindResources[] = GSDefaultResources;
		string PSBindResources[] = PSTracingResources;
		string CSBindResources[] = PSTracingResources;
	>
	{
		SetRasterizerState( DoublesidedRasterizerState );
		SetDepthStencilState( DisableDepthStencilState, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetHullShader( NULL );
		SetDomainShader( NULL );
		SetGeometryShader( NULL );
	}
}
