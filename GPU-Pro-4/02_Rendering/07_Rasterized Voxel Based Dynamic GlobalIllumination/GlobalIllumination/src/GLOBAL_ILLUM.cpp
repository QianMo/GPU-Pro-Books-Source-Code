#include <stdafx.h>
#include <DEMO.h>
#include <GLOBAL_ILLUM.h>

bool GLOBAL_ILLUM::Create()
{	
	// commonly used objects
	{
		sceneRT = DEMO::renderer->GetRenderTarget(GBUFFER_RT_ID);
		if(!sceneRT)
			return false;
		
		BLEND_DESC blendDesc;
		blendDesc.blend = true;
		blendBS = DEMO::renderer->CreateBlendState(blendDesc);
		if(!blendBS)
			return false;

		UNIFORM_LIST uniformList;
		uniformList.AddElement("gridViewProjMatrices",MAT4_DT,6);
    uniformList.AddElement("gridCellSizes",VEC4_DT);
		uniformList.AddElement("gridPositions",VEC4_DT,2);
		uniformList.AddElement("snappedGridPositions",VEC4_DT,2);
		gridUniformBuffer = DEMO::renderer->CreateUniformBuffer(CUSTOM_UB_BP,uniformList);
		if(!gridUniformBuffer)
			return false;

		gridHalfExtents[FINE_GRID] = 1000.0f;
		gridHalfExtents[COARSE_GRID] = 1600.0f;

		gridCellSizes.x = gridHalfExtents[FINE_GRID]/16.0f;
		gridCellSizes.y = 1.0f/gridCellSizes.x;
		gridCellSizes.z = gridHalfExtents[COARSE_GRID]/16.0f;
		gridCellSizes.w = 1.0f/gridCellSizes.z;

		// for generating the voxel-grids, the scene geometry is rendered with orthographic projection
    for(int i=0;i<2;i++)
		{
			gridProjMatrices[i].SetOrtho(-gridHalfExtents[i],gridHalfExtents[i],-gridHalfExtents[i],
			                             gridHalfExtents[i],0.2f,2.0f*gridHalfExtents[i]); 
		}
	}

	// objects used for generating the voxel-grids
	{
		gridRT = DEMO::renderer->CreateRenderTarget(64,64,1,TEX_FORMAT_RGB8);
		if(!gridRT)
			return false;

		gridSBs[FINE_GRID] = DEMO::renderer->CreateStructuredBuffer(CUSTOM_SB0_BP,32*32*32,6*sizeof(float));
		if(!gridSBs[FINE_GRID])
			return false;
		gridSBs[COARSE_GRID] = DEMO::renderer->CreateStructuredBuffer(CUSTOM_SB1_BP,32*32*32,6*sizeof(float));
		if(!gridSBs[COARSE_GRID])
			return false;
	
		RT_CONFIG_DESC desc;
		desc.numStructuredBuffers = 1;
		desc.structuredBuffers[0] = gridSBs[FINE_GRID];
		gridRTCs[FINE_GRID] = DEMO::renderer->CreateRenderTargetConfig(desc);
		if(!gridRTCs[FINE_GRID])
			return false;
		desc.numStructuredBuffers = 1;
		desc.structuredBuffers[0] = gridSBs[COARSE_GRID];
		gridRTCs[COARSE_GRID] = DEMO::renderer->CreateRenderTargetConfig(desc);
		if(!gridRTCs[COARSE_GRID])
			return false;
		
		gridFillShaders[FINE_GRID] = DEMO::resourceManager->LoadShader("shaders/gridFill.sdr",1); // (Permutation 1 = FINE_GRID)
		if(!gridFillShaders[0])
		  return false;
		gridFillShaders[COARSE_GRID] = DEMO::resourceManager->LoadShader("shaders/gridFill.sdr");
		if(!gridFillShaders[1])
			return false;
		
		RASTERIZER_DESC rasterDesc;
		gridRS = DEMO::renderer->CreateRasterizerState(rasterDesc);
		if(!gridRS)
			return false;

		// disable depth-write and depth-test, in order to fully "voxelize" scene geometry 
		DEPTH_STENCIL_DESC depthStencilDesc;
		depthStencilDesc.depthTest = false;
		depthStencilDesc.depthMask = false;
		gridDSS = DEMO::renderer->CreateDepthStencilState(depthStencilDesc);
		if(!gridDSS)
			return false;

		// disable color-write since instead of outputting the rasterized voxel information into the bound render-target,
		// it will be written into a 3D structured buffer
		BLEND_DESC blendDesc;
		blendDesc.colorMask = 0;
		gridBS = DEMO::renderer->CreateBlendState(blendDesc);
		if(!gridBS)
			return false;
	}

	// objects used for illuminating the voxel-grids
	{
		// Create for each grid resolution a 32x32x32 2D texture array with 3 attached render-targets (one for each second order spherical
		// harmonics coefficients for each color channel). Since this render-targets will be further used for the light propagation step,
		// for each grid resolution two MRTs are required (for iterative rendering).  
		for(int i=0;i<2;i++)
		{
			lightRTs[FINE_GRID][i] = DEMO::renderer->CreateRenderTarget(32,32,32,TEX_FORMAT_RGBA16F,false,3,DEMO::renderer->GetSampler(LINEAR_SAMPLER_ID),true);
			if(!lightRTs[FINE_GRID][i])
				return false;			
			lightRTs[COARSE_GRID][i] = DEMO::renderer->CreateRenderTarget(32,32,32,TEX_FORMAT_RGBA16F,false,3,DEMO::renderer->GetSampler(LINEAR_SAMPLER_ID),true);
			if(!lightRTs[COARSE_GRID][i])
				return false;	
		}
	}

	// objects used for the light propagation
	{
		// configure corresponding render-target, to perform light propagation in the compute shader
		RT_CONFIG_DESC desc;
		desc.computeTarget = true;
		desc.numColorBuffers = 3;
		lightPropagateRTC = DEMO::renderer->CreateRenderTargetConfig(desc);
		if(!lightPropagateRTC)
			return false;

		lightPropagateShaders[FINE_GRID][0] = DEMO::resourceManager->LoadShader("shaders/lightPropagate.sdr",1); // (Permutation 1 = FINE_GRID)
		if(!lightPropagateShaders[0])
			return false;
		lightPropagateShaders[FINE_GRID][1] = DEMO::resourceManager->LoadShader("shaders/lightPropagate.sdr",3); // (Permutation 3 = FINE_GRID + USE_OCCLUSION)
		if(!lightPropagateShaders[1])
			return false;
		lightPropagateShaders[COARSE_GRID][0] = DEMO::resourceManager->LoadShader("shaders/lightPropagate.sdr");
		if(!lightPropagateShaders[0])
			return false;
		lightPropagateShaders[COARSE_GRID][1] = DEMO::resourceManager->LoadShader("shaders/lightPropagate.sdr",2); // (Permutation 2 = USE_OCCLUSION)
		if(!lightPropagateShaders[1])
			return false;
	}

	// objects used for generating the indirect illumination
	{
		// only render into the accumulation render-target of the GBuffer
		RT_CONFIG_DESC desc;
		desc.numColorBuffers = 1;
		outputRTC = DEMO::renderer->CreateRenderTargetConfig(desc);
		if(!outputRTC)
			return false;

		globalIllumShader = DEMO::resourceManager->LoadShader("shaders/globalIllum.sdr");
		if(!globalIllumShader)
			return false;
		globalIllumNoTexShader = DEMO::resourceManager->LoadShader("shaders/globalIllum.sdr",1); // (Permutation 1 = NO_TEXTURE)
		if(!globalIllumNoTexShader)
			return false;

		// only illuminate actual scene geometry, not sky
		DEPTH_STENCIL_DESC depthStencilDesc;
		depthStencilDesc.stencilTest = true;
		depthStencilDesc.stencilRef = 1;
		depthStencilDesc.stencilPassOp = KEEP_STENCIL_OP;
		stencilTestDSS = DEMO::renderer->CreateDepthStencilState(depthStencilDesc);
		if(!stencilTestDSS)
			return false;
	}

	// objects used for visualizing the voxel-grids
	{
		gridVisShader = DEMO::resourceManager->LoadShader("shaders/gridVis.sdr");
		if(!gridVisShader)
			return false;
	}

	// objects used for clearing the voxel-grids
	{
		clearRT = DEMO::renderer->CreateRenderTarget(0,0,0,TEX_FORMAT_NONE,false,0);
		if(!clearRT)
			return false;
 
		// use compute shader for clearing the voxel-grids
		RT_CONFIG_DESC desc;
		desc.computeTarget = true;
		desc.numStructuredBuffers = 2;
		desc.structuredBuffers[0] = gridSBs[FINE_GRID];		
		desc.structuredBuffers[1] = gridSBs[COARSE_GRID];		
		clearRTC = DEMO::renderer->CreateRenderTargetConfig(desc);
		if(!clearRTC)
			return false;

		clearShader = DEMO::resourceManager->LoadShader("shaders/gridClear.sdr");
		if(!clearShader)
			return false;
	}

	Update();

	return true;
}

DX11_RENDER_TARGET* GLOBAL_ILLUM::GetOutputRT() const
{
	return sceneRT;
}

void GLOBAL_ILLUM::Update()
{
	// update data for each grid
	for(int i=0;i<2;i++)
		UpdateGrid(i);

	// update uniform-buffer
	float *uniformBufferData = gridViewProjMatrices[0];
	gridUniformBuffer->Update(uniformBufferData);
}

void GLOBAL_ILLUM::UpdateGrid(int gridType)
{
	CAMERA *camera = DEMO::renderer->GetCamera(MAIN_CAMERA_ID);
 
	// calculate center of grid
	VECTOR3D tmpGridPosition;
	if(gridType==FINE_GRID)
	  tmpGridPosition = camera->GetPosition()+(camera->GetDirection()*0.5f*gridHalfExtents[FINE_GRID]);
	else
	  tmpGridPosition = camera->GetPosition()+(camera->GetDirection()*(gridHalfExtents[COARSE_GRID]-0.5f*gridHalfExtents[FINE_GRID]));
	gridPositions[gridType].Set(tmpGridPosition);

	// calculate center of grid that is snapped to the grid-cell extents
	VECTOR3D tmpSnappedGridPosition;
	tmpSnappedGridPosition = tmpGridPosition;
	tmpSnappedGridPosition *= gridCellSizes.GetElements()[gridType*2+1];
	tmpSnappedGridPosition.Floor();
	tmpSnappedGridPosition *= gridCellSizes.GetElements()[gridType*2];
	snappedGridPositions[gridType].Set(tmpSnappedGridPosition);

	// back to front viewProjMatrix
	{
		VECTOR3D translate = -tmpSnappedGridPosition-VECTOR3D(0.0f,0.0f,gridHalfExtents[gridType]);
		MATRIX4X4 gridTransMatrix;
		gridTransMatrix.SetTranslation(translate);
		gridViewProjMatrices[gridType*3] = gridProjMatrices[gridType]*gridTransMatrix;
	}

	// right to left viewProjMatrix
	{
		VECTOR3D translate = -tmpSnappedGridPosition-VECTOR3D(gridHalfExtents[gridType],0.0f,0.0f);
		MATRIX4X4 gridTransMatrix;
		gridTransMatrix.SetTranslation(translate);
		MATRIX4X4 gridXRotMatrix;
		gridXRotMatrix.SetRotation(VECTOR3D(0.0f,1.0f,0.0f),90.0f);
		MATRIX4X4 gridViewMatrix;
		gridViewMatrix = gridXRotMatrix*gridTransMatrix;
		gridViewProjMatrices[gridType*3+1] = gridProjMatrices[gridType]*gridViewMatrix;
	}

	// top to down viewProjMatrix
	{
		VECTOR3D translate = -tmpSnappedGridPosition-VECTOR3D(0.0f,gridHalfExtents[gridType],0.0f);
		MATRIX4X4 gridTransMatrix;
		gridTransMatrix.SetTranslation(translate);
		MATRIX4X4 gridYRotMatrix;
		gridYRotMatrix.SetRotation(VECTOR3D(1.0f,0.0f,0.0f),90.0f);
		MATRIX4X4 gridViewMatrix;
		gridViewMatrix = gridYRotMatrix*gridTransMatrix;
		gridViewProjMatrices[gridType*3+2] = gridProjMatrices[gridType]*gridViewMatrix;
	}
}

void GLOBAL_ILLUM::PerformGridLightingPass()
{
	// let each active light illuminate the voxel-grids
	for(int i=0;i<DEMO::renderer->GetNumLights();i++) 
	{
		ILIGHT *light = DEMO::renderer->GetLight(i);
		if(light->IsActive())
			light->AddGridSurfaces();
	}
}

void GLOBAL_ILLUM::PerformLightPropagatePass(int index,gridTypes gridType)
{
	// Propagate virtual point-lights to 6 neighbor grid-cells in the compute shader. In the 
	// first iteration no occlusion is used, in order to initially let the light distribute. 
	// From the second iteration on we use the geometry occlusion, in order to avoid light leaking. 
	SURFACE surface;
	surface.renderTarget = lightRTs[gridType][1-currentLightRTIndices[gridType]];
	surface.renderTargetConfig = lightPropagateRTC;
	surface.renderOrder = GLOBAL_ILLUM_RO;
  if((useOcclusion)&&(index>0))
	{
		surface.shader = lightPropagateShaders[gridType][1];
		surface.customSBs[gridType] = gridSBs[gridType];
	}
	else
    surface.shader = lightPropagateShaders[gridType][0];
	surface.customTextures[0] = lightRTs[gridType][currentLightRTIndices[gridType]]->GetTexture();
	surface.customTextures[1] = lightRTs[gridType][currentLightRTIndices[gridType]]->GetTexture(1);
	surface.customTextures[2] = lightRTs[gridType][currentLightRTIndices[gridType]]->GetTexture(2);
	surface.vertexBuffer = NULL;
	surface.rasterizerState = NULL;
	surface.depthStencilState = NULL;
	surface.blendState = NULL;
	surface.renderMode = COMPUTE_RM;
	surface.numThreadGroupsX = 4;
	surface.numThreadGroupsY = 4;
	surface.numThreadGroupsZ = 4;
	DEMO::renderer->AddSurface(surface); 

	currentLightRTIndices[gridType] = 1-currentLightRTIndices[gridType];
}

void GLOBAL_ILLUM::PerformGlobalIllumPass()
{
	// Use normal-/ depth buffer (of GBuffer) to perform actual indirect illumination of each visible pixel.
	// By using the stencil buffer, we prevent that the sky is illuminated, too.
	SURFACE surface;
	surface.renderTarget = sceneRT;
	surface.renderTargetConfig = outputRTC;
	surface.renderOrder = GLOBAL_ILLUM_RO;
	surface.camera = DEMO::renderer->GetCamera(MAIN_CAMERA_ID);
  surface.normalTexture = sceneRT->GetTexture(2); // normalDepth
	surface.customTextures[0] = lightRTs[FINE_GRID][currentLightRTIndices[FINE_GRID]]->GetTexture();
	surface.customTextures[1] = lightRTs[FINE_GRID][currentLightRTIndices[FINE_GRID]]->GetTexture(1);
	surface.customTextures[2] = lightRTs[FINE_GRID][currentLightRTIndices[FINE_GRID]]->GetTexture(2);
	surface.customTextures[3] = lightRTs[COARSE_GRID][currentLightRTIndices[COARSE_GRID]]->GetTexture();
	surface.customTextures[4] = lightRTs[COARSE_GRID][currentLightRTIndices[COARSE_GRID]]->GetTexture(1);
	surface.customTextures[5] = lightRTs[COARSE_GRID][currentLightRTIndices[COARSE_GRID]]->GetTexture(2);
	surface.customUB = gridUniformBuffer;
	DEMO::renderer->SetupPostProcessSurface(surface);
	surface.depthStencilState = stencilTestDSS;
	if(mode==DEFAULT_GIM)
  {
	  surface.colorTexture = sceneRT->GetTexture(1); // albedoGloss
	  surface.blendState = blendBS;
		surface.shader = globalIllumShader;  
	}
	else
    surface.shader = globalIllumNoTexShader; 
	DEMO::renderer->AddSurface(surface); 
}

void GLOBAL_ILLUM::PerformGridVisPass()
{
  // Use depth buffer (of GBuffer) to reconstruct position of the pixels and extract/ visualize 
	// the corresponding voxels.
	SURFACE surface;
	surface.renderTarget = sceneRT;
	surface.renderTargetConfig = outputRTC;
	surface.renderOrder = GLOBAL_ILLUM_RO;
	surface.camera = DEMO::renderer->GetCamera(MAIN_CAMERA_ID);
	surface.normalTexture = sceneRT->GetTexture(2); // normalDepth
  surface.customSBs[0] = gridSBs[FINE_GRID];
	surface.customSBs[1] = gridSBs[COARSE_GRID];
	surface.customUB = gridUniformBuffer;
	surface.shader = gridVisShader;
	DEMO::renderer->SetupPostProcessSurface(surface);
  surface.depthStencilState = stencilTestDSS;
	DEMO::renderer->AddSurface(surface); 
}

void GLOBAL_ILLUM::PerformClearPass()
{
	// Clear voxel-grids in compute shader.
	SURFACE surface;
	surface.renderTarget = clearRT;
	surface.renderTargetConfig = clearRTC;
	surface.renderOrder = GLOBAL_ILLUM_RO;
	surface.shader = clearShader;  
	surface.vertexBuffer = NULL;
	surface.rasterizerState = NULL;
	surface.depthStencilState = NULL;
	surface.blendState = NULL;
	surface.renderMode = COMPUTE_RM;
	surface.numThreadGroupsX = 4;
	surface.numThreadGroupsY = 4;
	surface.numThreadGroupsZ = 4;
	DEMO::renderer->AddSurface(surface); 
}

void GLOBAL_ILLUM::AddSurfaces()
{
	Update();
  if((mode==DEFAULT_GIM)||(mode==INDIRECT_ILLUM_ONLY_GIM))
	{
		PerformGridLightingPass();

		// This demo uses 10 light propagation iterations. 
		for(int i=0;i<10;i++)
			PerformLightPropagatePass(i,FINE_GRID);
		for(int i=0;i<10;i++)
			PerformLightPropagatePass(i,COARSE_GRID);	

		PerformGlobalIllumPass();
	} 
  else if(mode==VISUALIZE_GIM)
		PerformGridVisPass();
	PerformClearPass();
	currentLightRTIndices[FINE_GRID] = 0;
	currentLightRTIndices[COARSE_GRID] = 0;
}

void GLOBAL_ILLUM::SetupGridSurface(SURFACE &surface,gridTypes gridType)
{
	// Setup passed scene geometry surface to generate the voxel-grid. Actually for each grid
	// the scene geometry is rendered in 2 orthographic views without depth-testing into a small
	// (64x64) render-target. Instead of writing out the results in the fragment shader into the 
  // render-target, they are written into a 3D structured buffer.
	surface.renderTarget = gridRT;
	surface.renderTargetConfig = gridRTCs[gridType];
	surface.rasterizerState = gridRS;
	surface.depthStencilState = gridDSS;
	surface.blendState = gridBS;
	surface.customUB = gridUniformBuffer;
	surface.shader = gridFillShaders[gridType];
}

void GLOBAL_ILLUM::SetupLightGridSurface(SURFACE &surface,gridTypes gridType)
{
	// Setup passed surface to illuminate the voxel-grid. This is done by rendering for each
	// light a full-screen quad (32x32 pixels) with 32 instances into the corresponding
	// 32x32x32 2D texture array. By using additive hardware blending the results are combined.
	if(!surface.light)
		return;
	surface.renderTarget = lightRTs[gridType][0];
	surface.customUB = gridUniformBuffer;
	surface.customSBs[gridType] = gridSBs[gridType];
	surface.numInstances = 32;
	DEMO::renderer->SetupPostProcessSurface(surface);
	surface.blendState = blendBS;
}



