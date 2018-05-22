#include <stdafx.h>
#include <DEMO.h>
#include <GLOBAL_ILLUM.h>
#include <DIRECTIONAL_LIGHT.h>

bool DIRECTIONAL_LIGHT::Create(const VECTOR3D &direction,const COLOR &color,float multiplier)
{
	this->direction = direction.GetNormalized();
	this->color = color;
	this->multiplier = multiplier;
	hasShadow = true;
	
	frustumRatio = 0.18f; // portion of complete shadow frustum, that will be used
	
	shadowTexMatrix.Set(0.5f,0.0f,0.0f,0.0f,
		                  0.0f,0.5f,0.0f,0.0f,
											0.0f,0.0f,1.0f,0.0f,
											0.5f,0.5f,0.0f,1.0f);

	CalculateFrustum();
	CalculateMatrices();

	// shader for direct illumination
	lightShader = DEMO::resourceManager->LoadShader("shaders/dirLight.sdr");
	if(!lightShader)
		return false;

	// shader for shadow map generation
	shadowMapShader = DEMO::resourceManager->LoadShader("shaders/shadowMapDir.sdr");
	if(!shadowMapShader)
		return false;

	// shader for illumination of fine resolution voxel-grid
	lightGridShaders[FINE_GRID] = DEMO::resourceManager->LoadShader("shaders/dirLightGrid.sdr",1); // (Permutation 1 = FINE_GRID)
	if(!lightGridShaders[FINE_GRID])
		return false;

	// shader for illumination of coarse resolution voxel-grid
	lightGridShaders[COARSE_GRID] = DEMO::resourceManager->LoadShader("shaders/dirLightGrid.sdr");
	if(!lightGridShaders[COARSE_GRID])
		return false;

  UNIFORM_LIST uniformList;
	uniformList.AddElement("direction",VEC3_DT);
	uniformList.AddElement("multiplier",FLOAT_DT);
	uniformList.AddElement("color",VEC4_DT);
	uniformList.AddElement("shadowViewProjMatrix",MAT4_DT);
	uniformList.AddElement("shadowViewProjTexMatrix",MAT4_DT);
	uniformList.AddElement("invShadowMapSize",FLOAT_DT);
	uniformBuffer = DEMO::renderer->CreateUniformBuffer(LIGHT_UB_BP,uniformList);
	if(!uniformBuffer)
		return false;
	
	UpdateUniformBuffer();

	RASTERIZER_DESC rasterDesc;
	noneCullRS = DEMO::renderer->CreateRasterizerState(rasterDesc);
	if(!noneCullRS)
		return false;
	rasterDesc.cullMode = BACK_CULL;
	backCullRS = DEMO::renderer->CreateRasterizerState(rasterDesc);
	if(!backCullRS)
		return false;

	DEPTH_STENCIL_DESC depthStencilDesc;
	defaultDSS = DEMO::renderer->CreateDepthStencilState(depthStencilDesc);
	if(!defaultDSS)
		return false;
	depthStencilDesc.depthTest = false;
	depthStencilDesc.depthMask = false;

	// only illuminate actual geometry, not sky
	depthStencilDesc.stencilTest = true; 
  depthStencilDesc.stencilRef = 1;
	depthStencilDesc.stencilPassOp = KEEP_STENCIL_OP;

	noDepthTestDSS = DEMO::renderer->CreateDepthStencilState(depthStencilDesc);
	if(!noDepthTestDSS)
		return false;
	
	BLEND_DESC blendDesc;
	blendDesc.colorMask = 0;
	noColorBS = DEMO::renderer->CreateBlendState(blendDesc);
	if(!noColorBS)
		return false;
	blendDesc.colorMask = ALL_COLOR_MASK;
	blendDesc.blend = true;
	blendBS = DEMO::renderer->CreateBlendState(blendDesc);
	if(!blendBS)
		return false;

	// render direct illumination only into accumulation render-target of GBuffer
  RT_CONFIG_DESC rtcDesc;
	rtcDesc.numColorBuffers = 1;
	rtConfig = DEMO::renderer->CreateRenderTargetConfig(rtcDesc);
	if(!rtConfig)
		return false;

  // cache pointer to GLOBAL_ILLUM post-processor
	globalIllumPP = (GLOBAL_ILLUM*)DEMO::renderer->GetPostProcessor("GLOBAL_ILLUM");
	if(!globalIllumPP)
		return false; 
	
	index = DEMO::renderer->GetNumLights();

	return true;
}

lightTypes DIRECTIONAL_LIGHT::GetLightType() const 
{
	return DIRECTIONAL_LT;
}

void DIRECTIONAL_LIGHT::CalculateFrustum()
{
	// get corners of camera frustum in view-space  
	CAMERA *camera = DEMO::renderer->GetCamera(MAIN_CAMERA_ID);
	VECTOR3D frustumCornersVS[8] =
	{
		VECTOR3D(-1.0f,1.0f,0.0f),
		VECTOR3D(1.0f,1.0f,0.0f),
		VECTOR3D(1.0f,-1.0f,0.0f),
		VECTOR3D(-1.0f,-1.0f,0.0f),
		VECTOR3D(-1.0f,1.0f,1.0f),
		VECTOR3D(1.0f,1.0f,1.0f),
		VECTOR3D(1.0f,-1.0f,1.0f),
		VECTOR3D(-1.0f,-1.0f,1.0f)
	};
	for(int i=0;i<8;i++)
		frustumCornersVS[i] = camera->GetInvProjMatrix()*frustumCornersVS[i];

  // get corners of shadow frustum in view space
	VECTOR3D shadowFrustumCornersVS[8];
	for(int j=0;j<4;j++)
	{
		VECTOR3D cornerRay = frustumCornersVS[j+4]-frustumCornersVS[j];
		VECTOR3D farCornerRay = cornerRay*frustumRatio;
		shadowFrustumCornersVS[j] = frustumCornersVS[j];
		shadowFrustumCornersVS[j+4] = frustumCornersVS[j]+farCornerRay;
	}

	// calculate radius of bounding-sphere for the shadow frustum
	VECTOR3D centerVS(0.0f,0.0f,0.0f);
	for(int j=0;j<8;j++)
		centerVS += shadowFrustumCornersVS[j];
	centerVS /= 8.0f;
	frustumRadius = 0.0f;
	for(int j=0;j<8;j++)
	{
		float distance = (shadowFrustumCornersVS[j]-centerVS).GetLength();
		if(distance>frustumRadius)
			frustumRadius = distance;
	}

	// calculate shadowProjMatrix
	shadowProjMatrix.SetOrtho(-frustumRadius,frustumRadius,-frustumRadius,frustumRadius,0.2f,frustumRadius*2.0f); 
}

void DIRECTIONAL_LIGHT::CalculateMatrices()
{
	// get light-position
	CAMERA *camera = DEMO::renderer->GetCamera(MAIN_CAMERA_ID);
	float distance = camera->GetNearFarClipDistance()*frustumRatio*0.5f;
	VECTOR3D center = camera->GetPosition()+camera->GetDirection()*distance;
	VECTOR3D position = center-(direction*frustumRadius);

	// calculate shadowViewProjMatrix 
	MATRIX4X4 transMatrix,rotMatrix,shadowViewMatrix;
	transMatrix.SetTranslation(position);
	rotMatrix.SetRotation(direction);
	shadowViewMatrix = transMatrix*rotMatrix;
	shadowViewMatrix = shadowViewMatrix.GetInverse();
	shadowViewProjMatrix = shadowProjMatrix*shadowViewMatrix;

	// prevent flickering of shadow map when main camera is moving
	VECTOR3D shadowOrigin(0.0f,0.0f,0.0f);
	shadowOrigin = shadowViewProjMatrix*shadowOrigin;
	float shadowMapSize = (float)DEMO::renderer->GetRenderTarget(SHADOW_MAP_RT_ID)->GetWidth();
	invShadowMapSize = 1.0f/shadowMapSize;
	shadowOrigin *= shadowMapSize*0.5f;
	VECTOR3D roundedOrigin = shadowOrigin.GetFloored();
	VECTOR3D roundOffset = roundedOrigin-shadowOrigin;   
	roundOffset *= 2.0f*invShadowMapSize;
	MATRIX4X4 roundedShadowProjMatrix = shadowProjMatrix;
	roundedShadowProjMatrix.entries[12] += roundOffset.x;
	roundedShadowProjMatrix.entries[13] += roundOffset.y;
	shadowViewProjMatrix = roundedShadowProjMatrix*shadowViewMatrix;

	// calculate shadowViewProjTexMatrix
	shadowViewProjTexMatrix = shadowTexMatrix*shadowViewProjMatrix;
}

DX11_UNIFORM_BUFFER* DIRECTIONAL_LIGHT::GetUniformBuffer() const
{
	return uniformBuffer;
}

void DIRECTIONAL_LIGHT::UpdateUniformBuffer()
{
	float *uniformBufferData = direction;
	uniformBuffer->Update(uniformBufferData);
}

void DIRECTIONAL_LIGHT::Update()
{	
	if(!active)
    return;
	CalculateMatrices(); 
	UpdateUniformBuffer();
}

void DIRECTIONAL_LIGHT::SetDirection(VECTOR3D &direction)
{
	this->direction = direction.GetNormalized();
}

void DIRECTIONAL_LIGHT::SetColor(COLOR &color)
{
	this->color = color;
}

void DIRECTIONAL_LIGHT::SetMultiplier(float multiplier)
{
	this->multiplier = multiplier;
}

void DIRECTIONAL_LIGHT::SetupShadowMapSurface(SURFACE *surface)
{
	surface->renderTarget = DEMO::renderer->GetRenderTarget(SHADOW_MAP_RT_ID);
	surface->renderOrder = SHADOW_RO;
	surface->light = this;
  surface->rasterizerState = backCullRS;	
	surface->depthStencilState = defaultDSS;
	surface->blendState = noColorBS;
  surface->shader = shadowMapShader; 
}

void DIRECTIONAL_LIGHT::AddLitSurface()
{
	if(!active)
		return;
	MESH *screenQuadMesh = DEMO::renderer->GetMesh(SCREEN_QUAD_MESH_ID);
	SURFACE surface;
	surface.renderTarget = DEMO::renderer->GetRenderTarget(GBUFFER_RT_ID);
	surface.renderTargetConfig = rtConfig;
	surface.renderOrder = ILLUM_RO;
	surface.camera = DEMO::renderer->GetCamera(MAIN_CAMERA_ID);
	surface.vertexBuffer = screenQuadMesh->vertexBuffer; 
	surface.primitiveType = screenQuadMesh->primitiveType;
	surface.firstIndex = 0;
	surface.numElements = screenQuadMesh->vertexBuffer->GetVertexCount();
	surface.colorTexture = DEMO::renderer->GetRenderTarget(GBUFFER_RT_ID)->GetTexture(1); // albedoGloss
	surface.normalTexture = DEMO::renderer->GetRenderTarget(GBUFFER_RT_ID)->GetTexture(2); // normalDepth 
	surface.specularTexture = DEMO::renderer->GetRenderTarget(SHADOW_MAP_RT_ID)->GetDepthStencilTexture(); // shadow map
	surface.light = this;
	surface.rasterizerState = noneCullRS;
	surface.depthStencilState = noDepthTestDSS;
	surface.blendState = blendBS;
	surface.shader = lightShader;
	surface.renderMode = NON_INDEXED_RM;
	DEMO::renderer->AddSurface(surface);
}

void DIRECTIONAL_LIGHT::AddGridSurfaces()
{
	if(!active)
		return;

	// illuminate fine and coarse resolution voxel-grid of GLOBAL_ILLUM post-processor
	for(int i=0;i<2;i++)
	{
		SURFACE surface;
		surface.renderOrder = GRID_ILLUM_RO;
		surface.light = this;
		surface.colorTexture = DEMO::renderer->GetRenderTarget(SHADOW_MAP_RT_ID)->GetDepthStencilTexture();
		surface.shader = lightGridShaders[i];
		globalIllumPP->SetupLightGridSurface(surface,(gridTypes)i);
		DEMO::renderer->AddSurface(surface); 
	}
}