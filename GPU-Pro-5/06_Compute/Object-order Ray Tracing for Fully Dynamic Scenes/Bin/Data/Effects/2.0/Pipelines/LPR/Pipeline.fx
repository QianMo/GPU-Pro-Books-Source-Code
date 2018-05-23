#define BE_PERSPECTIVE_SETUP
#define BE_SCENE_SETUP

#include "Engine/Pipe.fx"
#include "Engine/Pipeline.fx"
#include "Pipelines/LPR/Scene.fx"
#include "Engine/Feedback.fx"
#include "Engine/Perspective.fx"
#include "Engine/Shadow.fx"

/// Geometry pipeline stage.
PipelineStage GeometryPipelineStage
<
	int Layer = 10;
	string Setup = "PipelineSetup";
>;

/// Shadow pipeline stage.
PipelineStage ShadowPipelineStage
<
	int Layer = 20;
	bool Normal = false;
	string Setup = "PipelineSetup";
>;

/// Lighting pipeline stage.
PipelineStage LightingPipelineStage
<
	int Layer = 30;
	string Setup = "PipelineSetup";
	bool Conditional = true;
>;

/// Default pipeline stage.
PipelineStage DefaultPipelineStage
<
	int Layer = 100;
	string Setup = "PipelineSetup";
>;

/// Default pipeline stage.
PipelineStage OverlayPipelineStage
<
	int Layer = 10000;
	string Setup = "PipelineSetup";
	bool Conditional = true;
>;

/// Default pipeline stage.
PipelineStage ProcessingPipelineStage
<
	int Layer = 1000;
>;

/// ID pipeline stage.
PipelineStage ObjectIDPipelineStage
<
	int Layer = 100;
	bool Normal = false;
	string Setup = "PipelineSetup";
>;

/// Background render queue.
RenderQueue BackRenderQueue
<
	int Layer = 10;
>;

/// Default render queue.
RenderQueue DefaultRenderQueue
<
	int Layer = 30;
>;

/// Alpha render queue.
RenderQueue AlphaRenderQueue
<
	int Layer = 50;
	bool DepthSort = true;
	bool Backwards = true;
	string Setup = "PipelineSetup";
>;

/// Default rasterizer state enabling multi-sampling.
RasterizerState DefaultRasterizerState
{
	MultisampleEnable = true;
	AntiAliasedLineEnable = true;
};

/// Double=sided rasterizer state enabling multi-sampling.
RasterizerState DoublesidedRasterizerState
{
	MultisampleEnable = true;
	AntiAliasedLineEnable = true;
	CullMode = None;
};

/// Default depth-stencil state allowing for additive rendering.
DepthStencilState DefaultDepthStencilState
{
	DepthFunc = Less_Equal;
};

/// Read-only depth-stencil state.
DepthStencilState ReadOnlyDepthStencilState
{
	DepthWriteMask = Zero;
	DepthFunc = Less_Equal;
};

/// Alpha blend state.
BlendState AlphaBlendState
{
	BlendEnable[0] = true;
	SrcBlend[0] = One;
	DestBlend[0] = Inv_Src_Alpha;
};

/// Additive blend state.
BlendState DoubleAdditiveBlendState
{
	BlendEnable[0] = true;
	SrcBlend[0] = One;
	DestBlend[0] = One;

	BlendEnable[1] = true;
	SrcBlend[1] = One;
	DestBlend[1] = One;
};

static const String VSDefaultResources[] = { "PerspectiveConstants" };
static const String GSDefaultResources[] = { "PerspectiveConstants" };
static const String PSDefaultResources[] = { "PerspectiveConstants" };

static const String PSLightingResources[] = { "PerspectiveConstants", "SceneGeometryTexture", "SceneDiffuseTexture", "SceneSpecularTexture" };

technique11 PipelineSetup <
	bool EnableProcessing = true;
>
{
	pass <
		string PipelineStage = "GeometryPipelineStage";
		
		string Color0 = "SceneGeometryTarget";
		bool bClearColorOnce0 = true;
		float4 ClearColor0 = float4(10000.0f, 0.0f, 0.0f, -1.0f);
		bool bKeepColor0 = true;
		
		string Color1 = "SceneDiffuseTarget";
		bool bClearColorOnce1 = true;
		bool bKeepColor1 = true;

		string Color2 = "SceneSpecularTarget";
		bool bClearColorOnce2 = true;
		bool bKeepColor2 = true;
		
		string DepthStencil = "SceneDepthBuffer";
		bool bClearDepthOnce = true;
		bool bClearStencilOnce = true;
		bool bKeepDepthStencil = true;

		string VSBindResources[] = VSDefaultResources;
		string GSBindResources[] = GSDefaultResources;
		string PSBindResources[] = PSDefaultResources;
	>
	{
		SetRasterizerState( DefaultRasterizerState );
		SetDepthStencilState( DefaultDepthStencilState, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetHullShader( NULL );
		SetDomainShader( NULL );
		SetGeometryShader( NULL );
	}

	pass <
		string PipelineStage = "ShadowPipelineStage";

/*		string Color0 = "SceneShadowTarget";
		bool bClearColorOnce0 = true;
		float4 ClearColor0 = 10000.0f;
		bool bKeepColor0 = true;
*/
		bool Multisampled = false;
		
		string DepthStencil = "SceneShadowTarget"; // "SceneDepthBuffer";
		bool bClearDepthOnce = true;
		bool bClearStencilOnce = true;
		bool bKeepDepthStencil = true;

		string VSBindResources[] = VSDefaultResources;
		string GSBindResources[] = GSDefaultResources;
		string PSBindResources[] = PSDefaultResources;
	>
	{
		SetRasterizerState( DefaultRasterizerState );
		SetDepthStencilState( DefaultDepthStencilState, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetHullShader( NULL );
		SetDomainShader( NULL );
		SetGeometryShader( NULL );
	}

	pass <
		string PipelineStage = "LightingPipelineStage";

		string Color0 = "SceneTarget";
		bool bClearColorOnce0 = true;
		float4 ClearColor0 = 0.0f;
		bool bKeepColor0 = true;

		string DepthStencil = "SceneDepthBuffer";
		bool bKeepDepthStencil = true;

		string VSBindResources[] = VSDefaultResources;
		string GSBindResources[] = GSDefaultResources;
		string PSBindResources[] = PSLightingResources;
		string ForceTextureBinding[] = PSLightingResources;
	>
	{
		SetRasterizerState( NULL );
		SetDepthStencilState( ReadOnlyDepthStencilState, 0 );
		SetBlendState( DoubleAdditiveBlendState, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetHullShader( NULL );
		SetDomainShader( NULL );
		SetGeometryShader( NULL );
	}

	pass <
		string PipelineStage = "DefaultPipelineStage";

		string Color0 = "SceneTarget";
		bool bClearColorOnce0 = true;
		bool bKeepColor0 = true;

		string DepthStencil = "SceneDepthBuffer";
		bool bKeepDepthStencil = true;

		string VSBindResources[] = VSDefaultResources;
		string GSBindResources[] = GSDefaultResources;
		string PSBindResources[] = PSDefaultResources;
	>
	{
		SetRasterizerState( DefaultRasterizerState );
		SetDepthStencilState( DefaultDepthStencilState, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetHullShader( NULL );
		SetDomainShader( NULL );
		SetGeometryShader( NULL );
	}
	
	pass <
		string PipelineStage = "DefaultPipelineStage";
		string RenderQueue = "AlphaRenderQueue";
	>
	{
		SetBlendState( AlphaBlendState, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );
		
	}
	
	pass <
		string PipelineStage = "OverlayPipelineStage";

		string Color0 = "FinalTarget";
		bool bClearColorOnce0 = true;
		bool bKeepColor0 = true;

		string DepthStencil = "SceneDepthBuffer";
		bool bClearDepth = true;
		bool bKeepDepthStencil = true;

		string VSBindResources[] = VSDefaultResources;
		string GSBindResources[] = GSDefaultResources;
		string PSBindResources[] = PSDefaultResources;
	>
	{
		SetRasterizerState( DefaultRasterizerState );
		SetDepthStencilState( DefaultDepthStencilState, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetHullShader( NULL );
		SetDomainShader( NULL );
		SetGeometryShader( NULL );
	}
	
	pass <
		string PipelineStage = "ObjectIDPipelineStage";

		string Color0 = "ObjectIDTarget";
		float4 ClearColor0 = 4294960000.0f;
		bool bClearColorOnce0 = true;
		bool bKeepColor0 = true;

		string DepthStencil = "SceneDepthBuffer";
		bool bClearDepthOnce = true;
		bool bClearStencilOnce = true;
		bool bKeepDepthStencil = true;

		string VSBindResources[] = VSDefaultResources;
		string GSBindResources[] = GSDefaultResources;
		string PSBindResources[] = PSDefaultResources;
	>
	{
		SetRasterizerState( DefaultRasterizerState );
		SetDepthStencilState( DefaultDepthStencilState, 0 );
		SetBlendState( NULL, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff );

		SetHullShader( NULL );
		SetDomainShader( NULL );
		SetGeometryShader( NULL );
	}
}
