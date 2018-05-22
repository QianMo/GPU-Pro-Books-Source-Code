#include <stdafx.h>
#include <DEMO.h>
#include <GLOBAL_ILLUM.h>
#include <POINT_LIGHT.h>

bool POINT_LIGHT::Create(const VECTOR3D &position,float radius,const COLOR &color,float multiplier)
{
	this->position = position;
	this->radius = radius;
	this->color = color;
	this->multiplier = multiplier;
	hasShadow = false;
	cameraInVolume = false;

	CalculateMatrices();

	// shader for direct illumination
	lightShader = DEMO::resourceManager->LoadShader("shaders/pointLight.sdr");
	if(!lightShader)
		return false;

	// shader for illumination of fine resolution voxel-grid
	lightGridShaders[FINE_GRID] = DEMO::resourceManager->LoadShader("shaders/pointLightGrid.sdr",1); // (Permutation 1 = FINE_GRID)
	if(!lightGridShaders[FINE_GRID])
		return false;

	// shader for illumination of coarse resolution voxel-grid
	lightGridShaders[COARSE_GRID] = DEMO::resourceManager->LoadShader("shaders/pointLightGrid.sdr");
	if(!lightGridShaders[COARSE_GRID])
		return false;
	
  UNIFORM_LIST uniformList;
	uniformList.AddElement("position",VEC3_DT);
	uniformList.AddElement("radius",FLOAT_DT);
	uniformList.AddElement("color",VEC4_DT);
	uniformList.AddElement("worldMatrix",MAT4_DT);
	uniformList.AddElement("multiplier",FLOAT_DT);
	uniformBuffer = DEMO::renderer->CreateUniformBuffer(LIGHT_UB_BP,uniformList);
	if(!uniformBuffer)
		return false;

	UpdateUniformBuffer();

  performUpdate = false;

	RASTERIZER_DESC rasterDesc;
	rasterDesc.cullMode = BACK_CULL;
	backCullRS = DEMO::renderer->CreateRasterizerState(rasterDesc);
	if(!backCullRS)
		return false;
	rasterDesc.cullMode = FRONT_CULL;
	frontCullRS = DEMO::renderer->CreateRasterizerState(rasterDesc);
	if(!frontCullRS)
	  return false;

	DEPTH_STENCIL_DESC depthStencilDesc;
	depthStencilDesc.depthMask = false;

	// only illuminate actual geometry, not sky
	depthStencilDesc.stencilTest = true;
	depthStencilDesc.stencilRef = 1;
	depthStencilDesc.stencilPassOp = KEEP_STENCIL_OP;

	noDepthWriteDSS = DEMO::renderer->CreateDepthStencilState(depthStencilDesc);
	if(!noDepthWriteDSS)
		return false;
	depthStencilDesc.depthTest = false;
	noDepthTestDSS = DEMO::renderer->CreateDepthStencilState(depthStencilDesc);
	if(!noDepthTestDSS)
		return false;

	BLEND_DESC blendDesc;
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

lightTypes POINT_LIGHT::GetLightType() const
{
	return POINT_LT;
}

bool POINT_LIGHT::IsSphereInVolume(const VECTOR3D &position,float radius)
{
  float distance = (this->position-position).GetLength();
  if(distance<=(this->radius+radius))
	  return true;
	return false;
}

void POINT_LIGHT::CalculateMatrices()
{
	if(!performUpdate)
		return;

	// slightly increase radius to compensate for low tessellation of the used sphere geometry
	float dilatedRadius = radius+10.0f;

	// calculate worldMatrix of sphere geometry 
	VECTOR3D scale(dilatedRadius,dilatedRadius,dilatedRadius);
	MATRIX4X4 transMatrix,scaleMatrix;
	transMatrix.SetTranslation(position);
	scaleMatrix.SetScale(scale);
	worldMatrix = transMatrix*scaleMatrix;
}

DX11_UNIFORM_BUFFER* POINT_LIGHT::GetUniformBuffer() const
{
	return uniformBuffer;
}

void POINT_LIGHT::UpdateUniformBuffer()
{
	if(!performUpdate)
		return;
	float *uniformBufferData = position;
	uniformBuffer->Update(uniformBufferData);
}

void POINT_LIGHT::Update()
{
	if(!active)
	  return;
	cameraInVolume = IsSphereInVolume(DEMO::renderer->GetCamera(MAIN_CAMERA_ID)->GetPosition(),10.0f);
	CalculateMatrices(); 
	UpdateUniformBuffer();
	performUpdate = false;
}

void POINT_LIGHT::SetPosition(VECTOR3D &position)
{
	if(this->position==position)
		return;
	this->position = position;
	performUpdate = true;
}

void POINT_LIGHT::SetRadius(float radius)
{
	if(IS_EQUAL(this->radius,radius))
		return;
	this->radius = radius;
	performUpdate = true;
}

void POINT_LIGHT::SetColor(COLOR &color)
{
	if(this->color==color)
		return;
	this->color = color;
	performUpdate = true;
}

void POINT_LIGHT::SetMultiplier(float multiplier)
{
	if(IS_EQUAL(this->multiplier,multiplier))
		return;
	this->multiplier = multiplier;
	performUpdate = true;
}

void POINT_LIGHT::SetupShadowMapSurface(SURFACE *surface)
{
}

void POINT_LIGHT::AddLitSurface()
{
	if(!active)
		return;
	MESH *sphereMesh = DEMO::renderer->GetMesh(UNIT_SPHERE_MESH_ID);
	SURFACE surface;
	surface.renderTarget = DEMO::renderer->GetRenderTarget(GBUFFER_RT_ID);
	surface.renderTargetConfig = rtConfig;
	surface.renderOrder = ILLUM_RO;
	surface.camera = DEMO::renderer->GetCamera(MAIN_CAMERA_ID);
	surface.vertexBuffer = sphereMesh->vertexBuffer; 
	surface.indexBuffer = sphereMesh->indexBuffer;
	surface.primitiveType = sphereMesh->primitiveType;
	surface.firstIndex = 0;
	surface.numElements = sphereMesh->indexBuffer->GetIndexCount();
	surface.colorTexture = DEMO::renderer->GetRenderTarget(GBUFFER_RT_ID)->GetTexture(1); // albedoGloss
	surface.normalTexture = DEMO::renderer->GetRenderTarget(GBUFFER_RT_ID)->GetTexture(2); // normalDepth
	surface.light = this;

	// When camera not in light volume, do depth-testing + back-face culling, otherwise disable
	// depth-testing and do front-face culling.
  if(!cameraInVolume) 
	{	
		surface.rasterizerState = backCullRS;
		surface.depthStencilState = noDepthWriteDSS;
	}
	else 
	{	
		surface.rasterizerState = frontCullRS;
		surface.depthStencilState = noDepthTestDSS;
	}

	surface.blendState = blendBS;
	surface.shader = lightShader;
	DEMO::renderer->AddSurface(surface); 
}

void POINT_LIGHT::AddGridSurfaces()
{
	if(!active)
		return;

	// illuminate fine and coarse resolution voxel-grid of GLOBAL_ILLUM post-processor
  for(int i=0;i<2;i++)
	{
		SURFACE surface;
		surface.renderOrder = GRID_ILLUM_RO;
		surface.light = this;
		surface.shader = lightGridShaders[i];
		globalIllumPP->SetupLightGridSurface(surface,(gridTypes)i); 
		DEMO::renderer->AddSurface(surface); 
	}
}