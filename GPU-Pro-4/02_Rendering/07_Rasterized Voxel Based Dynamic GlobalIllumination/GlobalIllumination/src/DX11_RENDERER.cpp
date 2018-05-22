#include <stdafx.h>
#include <DEMO.h>
#include <DX11_RENDERER.h>

void DX11_RENDERER::Destroy()
{
	SAFE_DELETE_PLIST(samplers);
	SAFE_DELETE_PLIST(rasterizerStates);
	SAFE_DELETE_PLIST(depthStencilStates);
	SAFE_DELETE_PLIST(blendStates);
	SAFE_DELETE_PLIST(renderTargetConfigs);
	SAFE_DELETE_PLIST(renderTargets);
	SAFE_DELETE_PLIST(vertexBuffers);
	SAFE_DELETE_PLIST(indexBuffers);
	SAFE_DELETE_PLIST(uniformBuffers);
	SAFE_DELETE_PLIST(structuredBuffers);
	SAFE_DELETE_PLIST(cameras);
	SAFE_DELETE_PLIST(lights);
	SAFE_DELETE_PLIST(meshes);
	SAFE_DELETE_PLIST(postProcessors);
	surfaces.Erase();

	// reset device-context to default settings
	if(deviceContext)
		deviceContext->ClearState();

	// release swap-chain
	SAFE_RELEASE(swapChain);

	// release device-context
	SAFE_RELEASE(deviceContext);

	// release device
	SAFE_RELEASE(device);
}

bool DX11_RENDERER::Create()
{
	// create device/ device-context/ swap chain
	DXGI_SWAP_CHAIN_DESC desc;
	ZeroMemory(&desc,sizeof(DXGI_SWAP_CHAIN_DESC));
	desc.BufferDesc.Width = SCREEN_WIDTH;
	desc.BufferDesc.Height = SCREEN_HEIGHT;
	desc.BufferDesc.RefreshRate.Numerator = 60;
	desc.BufferDesc.RefreshRate.Denominator = 1;
	desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	desc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	desc.BufferCount = 2;
	desc.OutputWindow = DEMO::window->GetHWnd();
	desc.Windowed = TRUE;
	desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

	UINT createDeviceFlags = D3D11_CREATE_DEVICE_SINGLETHREADED;
#ifdef _DEBUG
	createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif 

	if(D3D11CreateDeviceAndSwapChain(NULL,D3D_DRIVER_TYPE_HARDWARE,NULL,createDeviceFlags,NULL,0,
    D3D11_SDK_VERSION,&desc,&swapChain,&device,NULL,&deviceContext)!=S_OK)
	{
		MessageBox(NULL,"Failed to create a DX11 device!","ERROR",MB_OK|MB_ICONEXCLAMATION);  
		Destroy();
		return false;
	}

	// check, if required feature-level is supported
	D3D_FEATURE_LEVEL featureLevel = device->GetFeatureLevel();
	if(featureLevel < D3D_FEATURE_LEVEL_11_0)
	{ 
		MessageBox(NULL,"Feature Level 11.0 not supported!","ERROR",MB_OK|MB_ICONEXCLAMATION); 
		Destroy();
		return false;
	}

	if(!CreateDefaultObjects())
		return false;

	// pre-allocate some surfaces, to prevent initial stutters
	surfaces.Resize(256);

	return true;
}

bool DX11_RENDERER::CreateDefaultObjects()
{
	// create frequently used samplers

	// LINEAR_SAMPLER
	SAMPLER_DESC samplerDesc;
	if(!CreateSampler(samplerDesc))
		return false;

	// TRILINEAR_SAMPLER
	samplerDesc.filter = MIN_MAG_MIP_LINEAR_FILTER;
	samplerDesc.adressU = REPEAT_TEX_ADDRESS;
	samplerDesc.adressV = REPEAT_TEX_ADDRESS;
	samplerDesc.adressW = REPEAT_TEX_ADDRESS;
	if(!CreateSampler(samplerDesc))
		return false;

	// SHADOW_MAP_SAMPLER
	samplerDesc.filter = COMP_MIN_MAG_LINEAR_FILTER;
	samplerDesc.adressU = CLAMP_TEX_ADRESS;
	samplerDesc.adressV = CLAMP_TEX_ADRESS;
	samplerDesc.adressW = CLAMP_TEX_ADRESS;
	samplerDesc.compareFunc = LEQUAL_COMP_FUNC;
	if(!CreateSampler(samplerDesc))
		return false;


	// create frequently used render-targets

	// BACK_BUFFER_RT 
	if(!CreateBackBufferRT())
		return false;

	// GBUFFER_RT	
	// 1. frameBufferTextures[0]:
	//    accumulation buffer
	// 2. frameBufferTextures[1]:
	//    RGB-channel: albedo, Alpha-channel: specular intensity
	// 3. frameBufferTextures[2]:
	//    RGB-channel: normal, Alpha-channel: depth 
	if(!CreateRenderTarget(SCREEN_WIDTH,SCREEN_HEIGHT,1,TEX_FORMAT_RGBA16F,true,3))
		return false;

	// SHADOW_MAP_RT
	if(!CreateRenderTarget(1024,1024,1,TEX_FORMAT_DEPTH24,true,0,GetSampler(SHADOW_MAP_SAMPLER_ID)))
		return false;

	// create render-states, frequently used by post-processors
	RASTERIZER_DESC rasterDesc;
	noneCullRS = CreateRasterizerState(rasterDesc);
	if(!noneCullRS)
		return false;

	DEPTH_STENCIL_DESC depthStencilDesc;
	depthStencilDesc.depthTest = false;
	depthStencilDesc.depthMask = false;
	noDepthTestDSS = CreateDepthStencilState(depthStencilDesc);
	if(!noDepthTestDSS)
		return false;

	BLEND_DESC blendDesc;
	defaultBS = CreateBlendState(blendDesc);
	if(!defaultBS)
		return false;


	// create frequently used cameras

	// MAIN_CAMERA
	if(!CreateCamera(80.0f,0.2f,5000.0f))
		return false;


	// create frequently used meshes

	// SCREEN_QUAD_MESH
	if(!CreateScreenQuadMesh())
		return false;

	// UNIT_SPHERE_MESH
	if(!CreateUnitSphere())
		return false;


	return true;
}

bool DX11_RENDERER::CreateScreenQuadMesh()
{
  // Create mesh with only 2 vertices, the geometry shader will generate the actual full-screen quad. 
	VERTEX_ELEMENT_DESC vertexLayout[2] = { POSITION_ELEMENT,R32G32B32_FLOAT_EF,0,
		                                      TEXCOORDS_ELEMENT,R32G32_FLOAT_EF,3};
	MESH *screenQuadMesh = CreateMesh(LINES_PRIMITIVE,vertexLayout,2,false,2,0);
	if(!screenQuadMesh)
		return false;

	QUAD_VERTEX screenQuadVertices[2]; 
	screenQuadVertices[0].position.Set(-1.0f,-1.0f,0.0f);	
	screenQuadVertices[0].texCoords.Set(0.0f,1.0f);
	screenQuadVertices[1].position.Set(1.0f,1.0f,0.0f);
	screenQuadVertices[1].texCoords.Set(1.0f,0.0f);
	screenQuadMesh->vertexBuffer->AddVertices(2,(float*)screenQuadVertices);
	screenQuadMesh->vertexBuffer->Update();

	return true;
}

bool DX11_RENDERER::CreateUnitSphere()
{
	// Create low tessellated unit sphere, used for example for rendering deferred point-lights.
	int tesselation = 18;
	int numSphereVertices = 2+(((tesselation/2)-2)*(tesselation/2)*4);
	int numSphereIndices = ((tesselation)*6)+(((tesselation/2)-2)*(tesselation/2)*12);
	VERTEX_ELEMENT_DESC vertexLayout[1] = { POSITION_ELEMENT,R32G32B32_FLOAT_EF,0 };
	MESH *sphereMesh = CreateMesh(TRIANGLES_PRIMITIVE,vertexLayout,1,false,numSphereVertices,numSphereIndices);
	if(!sphereMesh)
		return false;

	VECTOR3D *sphereVertices = new VECTOR3D[numSphereVertices];
	if(!sphereVertices)
		return false;
	int *sphereIndices = new int[numSphereIndices];
	if(!sphereIndices)
	{
		SAFE_DELETE_ARRAY(sphereVertices);
		return false;
	}

	// create vertices
	int vertexIndex = 0;
	VECTOR3D vertex;
	VECTOR4D theta;
	sphereVertices[vertexIndex++].Set(0.0f,-1.0f,0.0f);
	for(int i=1;i<tesselation/2-1;i++)
	{
		theta.x = ((i*TWOPI)/tesselation)-PIDIV2;	
		float sinThetaX = sin(theta.x);
		float cosThetaX = cos(theta.x);
		theta.y = (((i+1)*TWOPI)/tesselation)-PIDIV2;
		float sinThetaY = sin(theta.y);
		float cosThetaY = cos(theta.y);
		for(int j=0;j<tesselation;j+=2)
		{
			theta.z = (j*TWOPI)/tesselation;
			float sinThetaZ = sin(theta.z);
      float cosThetaZ = cos(theta.z);
			theta.w = ((j+1)*TWOPI)/tesselation;	
			float sinThetaW = sin(theta.w);
			float cosThetaW = cos(theta.w);
			vertex.x = cosThetaX*cosThetaZ;
			vertex.y = sinThetaX;
			vertex.z = cosThetaX*sinThetaZ;
			sphereVertices[vertexIndex++] = vertex;
			vertex.x = cosThetaY*cosThetaZ;
			vertex.y = sinThetaY;
			vertex.z = cosThetaY*sinThetaZ;
			sphereVertices[vertexIndex++] = vertex;
			vertex.x = cosThetaY*cosThetaW;
			vertex.y = sinThetaY;
			vertex.z = cosThetaY*sinThetaW;
			sphereVertices[vertexIndex++] = vertex;
			vertex.x = cosThetaX*cosThetaW;
			vertex.y = sinThetaX;
			vertex.z = cosThetaX*sinThetaW;
			sphereVertices[vertexIndex++] = vertex;			
		}
	}
	sphereVertices[vertexIndex++].Set(0.0f,1.0f,0.0f);

	// create lower cap indices
	int index = 0;
	vertexIndex = 1;
	for(int i=0;i<tesselation;i++)
	{
		sphereIndices[index++] = 0;
		sphereIndices[index++] = vertexIndex;
		if((i % 2)==0)
			vertexIndex += 3;
		else
			vertexIndex += 1;
		sphereIndices[index++] = (i<(tesselation-1)) ? vertexIndex : 1;
	}

	// create middle sphere indices
	vertexIndex = 1;
	for(int i=1;i<tesselation/2-1;i++)
	{
		int startIndex = vertexIndex;
		for(int j=0;j<tesselation/2;j++)
		{
			sphereIndices[index++] = vertexIndex++;
			sphereIndices[index++] = vertexIndex++;
			sphereIndices[index++] = vertexIndex;
			sphereIndices[index++] = vertexIndex++;
			sphereIndices[index++] = vertexIndex++;
			sphereIndices[index++] = vertexIndex-4;
			int nextIndex = (j==(tesselation/2-1)) ? startIndex : vertexIndex;
			sphereIndices[index++] = vertexIndex-1;
			sphereIndices[index++] = vertexIndex-2;
			sphereIndices[index++] = nextIndex+1;
			sphereIndices[index++] = nextIndex+1;
			sphereIndices[index++] = nextIndex;
			sphereIndices[index++] = vertexIndex-1;
		}
	}

	// create upper cap indices	
	int lastIndex = vertexIndex;
	vertexIndex -= 2;
	for(int i=0;i<tesselation;i++)
	{
		sphereIndices[index++] = lastIndex;
		sphereIndices[index++] = vertexIndex;
		if((i % 2)==0)
			vertexIndex -= 1;
		else
			vertexIndex -= 3;
		sphereIndices[index++] = (i<(tesselation-1)) ? vertexIndex : (lastIndex-2);
	}

	sphereMesh->vertexBuffer->AddVertices(numSphereVertices,(float*)sphereVertices);
	sphereMesh->vertexBuffer->Update();
	sphereMesh->indexBuffer->AddIndices(numSphereIndices,sphereIndices);
	sphereMesh->indexBuffer->Update();

	SAFE_DELETE_ARRAY(sphereVertices);
	SAFE_DELETE_ARRAY(sphereIndices);

	return true;
}

DX11_SAMPLER* DX11_RENDERER::CreateSampler(const SAMPLER_DESC &desc)
{
	for(int i=0;i<samplers.GetSize();i++)
	{
		if(samplers[i]->GetDesc()==desc)
			return samplers[i];
	}
	DX11_SAMPLER *sampler = new DX11_SAMPLER;
	if(!sampler)
		return NULL;
	if(!sampler->Create(desc))
	{
		SAFE_DELETE(sampler);
		return NULL;
	}
	samplers.AddElement(&sampler);
	return sampler;
}

DX11_SAMPLER* DX11_RENDERER::GetSampler(int ID) const
{
	if((ID<0)||(ID>=samplers.GetSize()))
		return NULL;
	return samplers[ID];
}

DX11_RASTERIZER_STATE* DX11_RENDERER::CreateRasterizerState(const RASTERIZER_DESC &desc)
{
	for(int i=0;i<rasterizerStates.GetSize();i++)
	{
		if(rasterizerStates[i]->GetDesc()==desc)
			return rasterizerStates[i];
	}
	DX11_RASTERIZER_STATE *rasterizerState = new DX11_RASTERIZER_STATE;
	if(!rasterizerState)
		return NULL;
	if(!rasterizerState->Create(desc))
	{
		SAFE_DELETE(rasterizerState);
		return NULL;
	}
	rasterizerStates.AddElement(&rasterizerState);
	return rasterizerState;
}

DX11_DEPTH_STENCIL_STATE* DX11_RENDERER::CreateDepthStencilState(const DEPTH_STENCIL_DESC &desc)
{
	for(int i=0;i<depthStencilStates.GetSize();i++)
	{
		if(depthStencilStates[i]->GetDesc()==desc)
			return depthStencilStates[i];
	}
	DX11_DEPTH_STENCIL_STATE *depthStencilState = new DX11_DEPTH_STENCIL_STATE;
	if(!depthStencilState)
		return NULL;
	if(!depthStencilState->Create(desc))
	{
		SAFE_DELETE(depthStencilState);
		return NULL;
	}
	depthStencilStates.AddElement(&depthStencilState);
	return depthStencilState;
}

DX11_BLEND_STATE* DX11_RENDERER::CreateBlendState(const BLEND_DESC &desc)
{
	for(int i=0;i<blendStates.GetSize();i++)
	{
		if(blendStates[i]->GetDesc()==desc)
			return blendStates[i];
	}
	DX11_BLEND_STATE *blendState = new DX11_BLEND_STATE;
	if(!blendState)
		return NULL;
	if(!blendState->Create(desc))
	{
		SAFE_DELETE(blendState);
		return NULL;
	}
	blendStates.AddElement(&blendState);
	return blendState;
}

RENDER_TARGET_CONFIG* DX11_RENDERER::CreateRenderTargetConfig(const RT_CONFIG_DESC &desc)
{
	for(int i=0;i<renderTargetConfigs.GetSize();i++)
	{
		if(renderTargetConfigs[i]->GetDesc()==desc)
			return renderTargetConfigs[i];
	}
	RENDER_TARGET_CONFIG *renderTargetConfig = new RENDER_TARGET_CONFIG;
	if(!renderTargetConfig)
		return NULL;
	if(!renderTargetConfig->Create(desc))
	{
		SAFE_DELETE(renderTargetConfig);
		return NULL;
	}
	renderTargetConfigs.AddElement(&renderTargetConfig);
	return renderTargetConfig;
}

DX11_RENDER_TARGET* DX11_RENDERER::CreateBackBufferRT()
{
	DX11_RENDER_TARGET *backBuffer = new DX11_RENDER_TARGET;
	if(!backBuffer)
		return NULL;
	backBuffer->CreateBackBuffer();
	renderTargets.AddElement(&backBuffer);
	return backBuffer;
}

DX11_RENDER_TARGET* DX11_RENDERER::CreateRenderTarget(int width,int height,int depth,texFormats format,bool depthStencil,int numColorBuffers,
																											DX11_SAMPLER *sampler,bool useUAV)
{
	DX11_RENDER_TARGET *renderTarget = new DX11_RENDER_TARGET;
	if(!renderTarget)
		return NULL;
	if(!renderTarget->Create(width,height,depth,format,depthStencil,numColorBuffers,sampler,useUAV))
	{
		SAFE_DELETE(renderTarget);
		return NULL;
	}
	renderTargets.AddElement(&renderTarget);
	return renderTarget;
}

DX11_RENDER_TARGET* DX11_RENDERER::GetRenderTarget(int ID) const
{
	if((ID<0)||(ID>=renderTargets.GetSize()))
		return NULL;
	return renderTargets[ID];
}

DX11_VERTEX_BUFFER* DX11_RENDERER::CreateVertexBuffer(const VERTEX_ELEMENT_DESC *vertexElementDescs,int numVertexElementDescs, 
																											bool dynamic,int maxVertexCount)
{
	DX11_VERTEX_BUFFER *vertexBuffer = new DX11_VERTEX_BUFFER;
	if(!vertexBuffer)
		return NULL;
	if(!vertexBuffer->Create(vertexElementDescs,numVertexElementDescs,dynamic,maxVertexCount))
	{
		SAFE_DELETE(vertexBuffer);
		return NULL;
	}
	vertexBuffers.AddElement(&vertexBuffer);
	return vertexBuffer;
}

DX11_INDEX_BUFFER* DX11_RENDERER::CreateIndexBuffer(bool dynamic,int maxIndexCount)
{
	DX11_INDEX_BUFFER *indexBuffer = new DX11_INDEX_BUFFER;
	if(!indexBuffer)
		return NULL;
	if(!indexBuffer->Create(dynamic,maxIndexCount))
	{
		SAFE_DELETE(indexBuffer);
		return NULL;
	}
	indexBuffers.AddElement(&indexBuffer);
	return indexBuffer;
}

DX11_UNIFORM_BUFFER* DX11_RENDERER::CreateUniformBuffer(uniformBufferBP bindingPoint,const UNIFORM_LIST &uniformList)
{
	DX11_UNIFORM_BUFFER *uniformBuffer = new DX11_UNIFORM_BUFFER;
	if(!uniformBuffer)
		return NULL;
	if(!uniformBuffer->Create(bindingPoint,uniformList))
	{
		SAFE_DELETE(uniformBuffer);
		return NULL;
	}
	uniformBuffers.AddElement(&uniformBuffer);
	return uniformBuffer;
}

DX11_STRUCTURED_BUFFER* DX11_RENDERER::CreateStructuredBuffer(int bindingPoint,int elementCount,int elementSize)
{
	DX11_STRUCTURED_BUFFER *structuredBuffer = new DX11_STRUCTURED_BUFFER;
	if(!structuredBuffer)
		return NULL;
	if(!structuredBuffer->Create(bindingPoint,elementCount,elementSize))
	{
		SAFE_DELETE(structuredBuffer);
		return NULL;
	}
	structuredBuffers.AddElement(&structuredBuffer);
	return structuredBuffer;
}

CAMERA* DX11_RENDERER::CreateCamera(float fovy,float nearClipDistance,float farClipDistance)
{
	CAMERA *camera = new CAMERA;
	if(!camera)
		return NULL;
	if(!camera->Init(fovy,nearClipDistance,farClipDistance))
	{
		SAFE_DELETE(camera);
		return NULL;
	}
	cameras.AddElement(&camera);
	return camera;
}

CAMERA* DX11_RENDERER::GetCamera(int ID) const
{
	if((ID<0)||(ID>=cameras.GetSize()))
		return NULL;
	return cameras[ID];
}

POINT_LIGHT* DX11_RENDERER::CreatePointLight(const VECTOR3D &position,float radius,const COLOR &color,float multiplier)
{
	POINT_LIGHT *pointLight = new POINT_LIGHT;
	if(!pointLight)
		return false;
	if(!pointLight->Create(position,radius,color,multiplier))
	{
		SAFE_DELETE(pointLight);
		return NULL;
	}
	lights.AddElement((ILIGHT**)(&pointLight));
	return pointLight;
}

DIRECTIONAL_LIGHT* DX11_RENDERER::CreateDirectionalLight(const VECTOR3D &direction,const COLOR &color,float multiplier)
{
	DIRECTIONAL_LIGHT *directionalLight = new DIRECTIONAL_LIGHT;
	if(!directionalLight)
		return false;
	if(!directionalLight->Create(direction,color,multiplier))
	{
		SAFE_DELETE(directionalLight);
		return NULL;
	}
	lights.AddElement((ILIGHT**)(&directionalLight));
	return directionalLight;
}

ILIGHT* DX11_RENDERER::GetLight(int index) const
{
	if((index<0)||(index>=lights.GetSize()))
		return NULL;
	return lights[index];
}

MESH* DX11_RENDERER::CreateMesh(primitiveTypes primitiveType,const VERTEX_ELEMENT_DESC *vertexElementDescs,
																int numVertexElementDescs,bool dynamic,int numVertices,int numIndices)
{
	MESH *mesh = new MESH;
	if(!mesh)
		return NULL;
	if(!mesh->Create(primitiveType,vertexElementDescs,numVertexElementDescs,dynamic,numVertices,numIndices))
	{
		SAFE_DELETE(mesh);
		return NULL;
	}
	meshes.AddElement(&mesh);
	return mesh;
}

MESH* DX11_RENDERER::GetMesh(int ID) const
{
	if((ID<0)||(ID>=meshes.GetSize()))
		return NULL;
	return meshes[ID];
}

IPOST_PROCESSOR* DX11_RENDERER::GetPostProcessor(const char *name) const
{
	if(!name)
		return NULL;
	for(int i=0;i<postProcessors.GetSize();i++)
	{
		if(strcmp(name,postProcessors[i]->GetName())==0)
			return postProcessors[i];
	}
	return NULL;
}

int DX11_RENDERER::GetNumLights() const
{
	return lights.GetSize();
}

void DX11_RENDERER::UpdateLights()
{
	for(int i=0;i<lights.GetSize();i++) 
	{
		if(lights[i]->IsActive())
			lights[i]->Update();
	}
}

void DX11_RENDERER::SetupPostProcessSurface(SURFACE &surface)
{
	MESH* screenQuadMesh = GetMesh(SCREEN_QUAD_MESH_ID);
	surface.vertexBuffer = screenQuadMesh->vertexBuffer;
	surface.primitiveType = screenQuadMesh->primitiveType;
	surface.firstIndex = 0;
	surface.numElements = screenQuadMesh->vertexBuffer->GetVertexCount();
	surface.rasterizerState = noneCullRS;
	surface.depthStencilState = noDepthTestDSS;
	surface.blendState = defaultBS;
	surface.renderMode = NON_INDEXED_RM;
}

void DX11_RENDERER::AddSurface(SURFACE &surface)
{
	int index = surfaces.AddElements(1,&surface);
	surfaces[index].ID = index;
}

void DX11_RENDERER::ClearFrame()
{
	surfaces.Clear(); 
	for(int i=0;i<renderTargets.GetSize();i++)
		renderTargets[i]->Reset();
	frameCleared = true;
}

void DX11_RENDERER::ExecutePostProcessors()
{
	for(int i=0;i<postProcessors.GetSize();i++)
	{
		if(postProcessors[i]->IsActive())
			postProcessors[i]->AddSurfaces();
	}
}

void DX11_RENDERER::SetRenderStates(SURFACE &surface)
{ 
	if((surface.rasterizerState!=lastSurface.rasterizerState)||(frameCleared))
	{
		if(surface.rasterizerState)
		  surface.rasterizerState->Set();
		lastSurface.rasterizerState = surface.rasterizerState;
	} 

	if((surface.depthStencilState!=lastSurface.depthStencilState)||(frameCleared))
	{   
		if(surface.depthStencilState)
		  surface.depthStencilState->Set();
		lastSurface.depthStencilState = surface.depthStencilState;
	}

	if((surface.blendState!=lastSurface.blendState)||(frameCleared))
	{   
		if(surface.blendState)
		  surface.blendState->Set();
		lastSurface.blendState = surface.blendState;
	}

	if((surface.renderTarget!=lastSurface.renderTarget)||(surface.renderTargetConfig!=lastSurface.renderTargetConfig)||
		 (frameCleared))
	{
		surface.renderTarget->Bind(surface.renderTargetConfig);
		lastSurface.renderTarget = surface.renderTarget;	  
		lastSurface.renderTargetConfig = surface.renderTargetConfig;
	}

	if((surface.vertexBuffer!=lastSurface.vertexBuffer)||(frameCleared))
	{
		if(surface.vertexBuffer)
			surface.vertexBuffer->Bind();
		lastSurface.vertexBuffer = surface.vertexBuffer;
	}

	if((surface.indexBuffer!=lastSurface.indexBuffer)||(frameCleared))
	{
		if(surface.indexBuffer)
			surface.indexBuffer->Bind();
		lastSurface.indexBuffer = surface.indexBuffer;
	}

	if((surface.shader!=lastSurface.shader)||(frameCleared))
	{
		surface.shader->Bind();
		lastSurface.shader = surface.shader;
	}

	if((surface.light!=lastSurface.light)||(frameCleared))
	{   
		if(surface.light)
		{
			if(surface.renderOrder==SHADOW_RO)
				surface.renderTarget->Clear(DEPTH_CLEAR_BIT);
		}
		lastSurface.light = surface.light;
	}
}

void DX11_RENDERER::SetShaderParams(SURFACE &surface)
{
	// set camera uniform-buffer
	if(surface.camera)
    surface.shader->SetUniformBuffer(surface.camera->GetUniformBuffer());

	// set light uniform-buffer
  if(surface.light)
		surface.shader->SetUniformBuffer(surface.light->GetUniformBuffer());

	// set custom uniform-buffer
	if(surface.customUB)
	  surface.shader->SetUniformBuffer(surface.customUB);

	// set custom structured buffers
  for(int i=0;i<NUM_CUSTOM_STRUCTURED_BUFFERS;i++)
	{
		if(surface.customSBs[i])
			surface.shader->SetStructuredBuffer(surface.customSBs[i]);
	}

	// set color texture
	if(surface.colorTexture)
		surface.shader->SetTexture(COLOR_TEX_BP,surface.colorTexture);

	// set normal texture
	if(surface.normalTexture)
		surface.shader->SetTexture(NORMAL_TEX_BP,surface.normalTexture);

	// set specular texture
	if(surface.specularTexture)
		surface.shader->SetTexture(SPECULAR_TEX_BP,surface.specularTexture);

	// set custom textures
  for(int i=0;i<NUM_CUSTOM_TEXURES;i++)
	{
		textureBP bindingPoint = (textureBP)(CUSTOM0_TEX_BP+i);
		if(surface.customTextures[i])
			surface.shader->SetTexture(bindingPoint,surface.customTextures[i]);
	}
}

// compare-function passed to qsort
int CompareSurfaces(const void *a,const void *b)
{
	SURFACE *sA = (SURFACE*)a;
	SURFACE *sB = (SURFACE*)b;

	// interleave shadow-map generation + direct illumination + illumination of voxel-grid
	if(((sA->renderOrder>=SHADOW_RO)&&(sA->renderOrder<=GRID_ILLUM_RO))&&
		((sB->renderOrder>=SHADOW_RO)&&(sB->renderOrder<=GRID_ILLUM_RO)))
	{
		if(sA->light->GetIndex()<sB->light->GetIndex())
			return -1;
		else if(sA->light->GetIndex()>sB->light->GetIndex())
			return 1;
	}

	if(sA->renderOrder<sB->renderOrder)
		return -1;
	else if(sA->renderOrder>sB->renderOrder)
		return 1;
	if(sA->GetID()<sB->GetID())
		return -1;
	else if(sA->GetID()>sB->GetID())
		return 1;
	return 0;
} 

void DX11_RENDERER::DrawSurfaces()
{
	ExecutePostProcessors();
	surfaces.Sort(CompareSurfaces);
	for(int i=0;i<surfaces.GetSize();i++)
	{ 
		if(i>0) 
			frameCleared = false;
		SetRenderStates(surfaces[i]);	
		SetShaderParams(surfaces[i]);
		switch(surfaces[i].renderMode)
		{
		case INDEXED_RM:
			DrawIndexedElements(surfaces[i].primitiveType,surfaces[i].numElements,surfaces[i].firstIndex,surfaces[i].numInstances);
			break;

		case NON_INDEXED_RM:
			DrawElements(surfaces[i].primitiveType,surfaces[i].numElements,surfaces[i].firstIndex,surfaces[i].numInstances);
			break;

		case COMPUTE_RM:
			Dispatch(surfaces[i].numThreadGroupsX,surfaces[i].numThreadGroupsY,surfaces[i].numThreadGroupsZ);
			break;
		}
	}
	swapChain->Present(VSYNC_ENABLED,0);
}

void DX11_RENDERER::DrawIndexedElements(primitiveTypes primitiveType,int numElements,int firstIndex,int numInstances)
{
	deviceContext->IASetPrimitiveTopology((D3D11_PRIMITIVE_TOPOLOGY)primitiveType);
	if(numInstances<2)
		deviceContext->DrawIndexed(numElements,firstIndex,0);
	else
		deviceContext->DrawIndexedInstanced(numElements,numInstances,firstIndex,0,0);
	UnbindShaderResources();
}

void DX11_RENDERER::DrawElements(primitiveTypes primitiveType,int numElements,int firstIndex,int numInstances)
{
	deviceContext->IASetPrimitiveTopology((D3D11_PRIMITIVE_TOPOLOGY)primitiveType);
	if(numInstances<2)
		deviceContext->Draw(numElements,firstIndex);

	else
		deviceContext->DrawInstanced(numElements,numInstances,firstIndex,0);
	UnbindShaderResources();
}

void DX11_RENDERER::Dispatch(int numThreadGroupsX,int numThreadGroupsY,int numThreadGroupsZ)
{
	deviceContext->Dispatch(numThreadGroupsX,numThreadGroupsY,numThreadGroupsZ);
	UnbindShaderResources();

	ID3D11UnorderedAccessView *unorderedAccessViews[MAX_NUM_COLOR_BUFFERS] = 
	  { NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL };
	deviceContext->CSSetUnorderedAccessViews(0,MAX_NUM_COLOR_BUFFERS,unorderedAccessViews, NULL);
}

void DX11_RENDERER::UnbindShaderResources()
{
	ID3D11ShaderResourceView *shaderResourceViews[] = 
	  { NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL };
	int numViews = NUM_TEXTURE_BP+NUM_STRUCTURED_BUFFER_BP;
	deviceContext->VSSetShaderResources(0,numViews,shaderResourceViews);
	deviceContext->GSSetShaderResources(0,numViews,shaderResourceViews);
	deviceContext->PSSetShaderResources(0,numViews,shaderResourceViews);
	deviceContext->CSSetShaderResources(0,numViews,shaderResourceViews);
}

void DX11_RENDERER::SaveScreenshot() const
{
	// try to find a not existing path for screenshot	
	char filePath[DEMO_MAX_FILEPATH];
	for(int i=0;i<1000;i++)
	{
		sprintf(filePath,"../Data/screenshots/screen%d.bmp",i);
		if(!DEMO::fileManager->FilePathExists(filePath))
			break;
		if(i==999)
			return;
	}

	// save content of back-buffer to bitmap file
	ID3D11Texture2D *backBufferTexture = NULL;
	if(swapChain->GetBuffer(0,__uuidof( ID3D11Texture2D),(LPVOID*)&backBufferTexture)!=S_OK)
		return;
	wchar_t wideFilePath[DEMO_MAX_FILEPATH];
	MultiByteToWideChar(CP_ACP,0,filePath,DEMO_MAX_FILEPATH,wideFilePath,DEMO_MAX_FILEPATH);
	D3DX11SaveTextureToFileW(deviceContext,backBufferTexture,D3DX11_IFF_BMP,wideFilePath);
	SAFE_RELEASE(backBufferTexture);
}
