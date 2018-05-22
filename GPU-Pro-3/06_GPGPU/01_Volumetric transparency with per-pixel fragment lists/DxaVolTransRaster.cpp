#include "DXUT.h"
#include "DxaVolTransRaster.h"
#include "Mesh/GeometryLoader.h"
#include "Mesh/Material.h"
#include "Mesh/VertexStream.h"
#include "Mesh/Indexed.h"
#include "Mesh/Instanced.h"
#include "Mesh/Importer.h"
#include <assimp.hpp>      // C++ importer interface
#include <aiScene.h>       // Output data structure
#include <aiPostProcess.h> // Post processing flags
#include "Sas/SasButton.h"
	
/// Base functor to toggle a bool
class TogglerFunctor : public SasControl::Button::Functor
{
	CDXUTButton* button;
	bool& variable;
	std::wstring trueText;
	std::wstring falseText;
public:
	TogglerFunctor(bool& variable, std::wstring trueText, std::wstring falseText)
		:variable(variable), trueText(trueText), falseText(falseText)
	{
		button = NULL;
	}
	void setButton(CDXUTButton* button) {this->button = button;}
	void operator()()
	{
		variable = !variable;
		if(variable)
			button->SetText(trueText.c_str());
		else
			button->SetText(falseText.c_str());
	}
};

Dxa::VolTransRaster::VolTransRaster(ID3D11Device* device)
	:Dxa::Sas(device)
{
	nParticles = 300;
	camera.SetViewParams( &D3DXVECTOR3(0, 3, -10), &D3DXVECTOR3(0, 3, 0) );
	camera.SetScalers(0.00999999, 800);

	displayParticles = false;
	freeze = false;
	drawSpheresOnly = 0;
}

void Dxa::VolTransRaster::loadEffect()
{
	ID3DBlob* compiledEffect = NULL;
	ID3DBlob* compilationErrors = NULL;
	HRESULT hr = D3DX11CompileFromFileW(
			L"fx/voltransmain.fx", NULL, NULL, NULL,
			"fx_5_0", 0, 0, NULL, &compiledEffect, &compilationErrors, NULL);
	if(hr != S_OK)
	{
		if(compilationErrors != NULL)
			MessageBoxA( NULL, 
				(LPSTR)compilationErrors->GetBufferPointer(),
				"Failed to create effect from file!", MB_OK);
		else
			MessageBoxA( NULL, 
				"File cound not be opened",
				"Failed to create effect from file!", MB_OK);
		exit(-1);
	}
	#pragma endregion compile effect file with error handling
	#pragma region 6.6
	hr = D3DX11CreateEffectFromMemory(
		compiledEffect->GetBufferPointer(), 
		compiledEffect->GetBufferSize(),
		0, device, &effect);
	if(hr != S_OK)
	{
		MessageBoxA( NULL, 
			"CreateEffectFromMemory failed",
			"Failed to create effect from file!", MB_OK);
		exit(-1);
	}
	#pragma endregion create effect with error handling
}

Mesh::Cast::P Dxa::VolTransRaster::loadMeshTransparent(const char* geometryFilename)
{
	Mesh::Geometry::P meshGeometry = Mesh::GeometryLoader::createGeometryFromFile(device, geometryFilename);//importObject(geometryFilename);
	Mesh::Cast::P castMesh = Mesh::Cast::make();

	ID3DX11EffectPass* storePass = effect->GetTechniqueByName("volumetricTransparency")->GetPassByName("storeFragments");
	D3DX11_PASS_DESC storePassDesc;
	storePass->GetDesc(&storePassDesc);
	castMesh->add(storeRole,
		Mesh::Shaded::make(
			binder->bind(device, storePassDesc, meshGeometry),
			Mesh::Material::make(storePass, D3DX11_EFFECT_PASS_KEEP_UAV_COUNTERS))); // keep uav counters when setting this material
	return castMesh;
}

Mesh::Cast::P Dxa::VolTransRaster::loadMeshOpaque(const char* geometryFilename, bool isShadowCaster)
{
	Mesh::Geometry::P meshGeometry = Mesh::GeometryLoader::createGeometryFromFile(device, geometryFilename);
	Mesh::Cast::P castMesh = Mesh::Cast::make();

	ID3DX11EffectPass* deferPass = effect->GetTechniqueByName("volumetricTransparency")->GetPassByName("defer");
	D3DX11_PASS_DESC deferPassDesc;
	deferPass->GetDesc(&deferPassDesc);
	Mesh::Material::P material = Mesh::Material::make(deferPass, 0);
	material->saveVariable( effect->GetVariableByName("kdTexture") );
	material->saveVariable( effect->GetVariableByName("kdColor") );
	castMesh->add(deferRole,
		Mesh::Shaded::make(
			binder->bind(device, deferPassDesc, meshGeometry),
			material));

	if(isShadowCaster)
	{
		ID3DX11EffectPass* shadowPass = effect->GetTechniqueByName("volumetricTransparency")->GetPassByName("storeShadowFragments");
		D3DX11_PASS_DESC shadowPassDesc;
		shadowPass->GetDesc(&shadowPassDesc);
		castMesh->add(storeRole,
			Mesh::Shaded::make(
				binder->bind(device, shadowPassDesc, meshGeometry),
				Mesh::Material::make(shadowPass, D3DX11_EFFECT_PASS_KEEP_UAV_COUNTERS)));  // keep uav counters when setting this material
	}

	return castMesh;
}

ID3D11ShaderResourceView* Dxa::VolTransRaster::loadTexture(const char* filename)
{
	if(filename == NULL || filename[0] == '\0')
		return NULL;
	ID3D11ShaderResourceView* srv;
	SrvDirectory::iterator iSrv = srvs.find(filename);
	if(iSrv == srvs.end())
	{
		HRESULT hr = D3DX11CreateShaderResourceViewFromFileA( device, filename, NULL, NULL, &srv, NULL);

		if(hr != S_OK)
		{
			srv = NULL;
			//MessageBoxA( NULL, "Failed to load texture.", filename, MB_OK);
		}
		srvs[filename] = srv;
	}
	else
		srv = iSrv->second;
	return srv;
}

Mesh::Geometry::P Dxa::VolTransRaster::importObject(const char* objectFileName)
{
	Assimp::Importer importer;
	const aiScene* assImpScene = importer.ReadFile( objectFileName,
		aiProcess_CalcTangentSpace		| 
		aiProcess_Triangulate			|
		aiProcess_JoinIdenticalVertices	|
		aiProcess_SortByPType);

	return Mesh::Importer::fromAiMesh(device, assImpScene->mMeshes[0]);
}

void Dxa::VolTransRaster::importScene(const char* sceneFileName)
{
	ID3DX11EffectPass* deferPass = effect->GetTechniqueByName("volumetricTransparency")->GetPassByName("defer");
	D3DX11_PASS_DESC deferPassDesc;
	deferPass->GetDesc(&deferPassDesc);

	ID3DX11EffectPass* storePass = effect->GetTechniqueByName("volumetricTransparency")->GetPassByName("storeFragments");
	D3DX11_PASS_DESC storePassDesc;
	storePass->GetDesc(&storePassDesc);

	ID3DX11EffectPass* fluidPass = effect->GetTechniqueByName("volumetricTransparency")->GetPassByName("storeFluidFragments");
	D3DX11_PASS_DESC fluidPassDesc;
	fluidPass->GetDesc(&fluidPassDesc);

	Assimp::Importer importer;

	const aiScene* assImpScene = importer.ReadFile( sceneFileName,
		aiProcess_CalcTangentSpace		| 
		aiProcess_Triangulate			|
		aiProcess_JoinIdenticalVertices	|
		aiProcess_SortByPType);

	if(assImpScene->HasMeshes())
	{
		for(int iMesh=0; iMesh < assImpScene->mNumMeshes; iMesh++)
		{
			Mesh::Geometry::P meshGeometry = Mesh::Importer::fromAiMesh(device, assImpScene->mMeshes[iMesh]);
			aiMaterial* assImpMaterial = assImpScene->mMaterials[assImpScene->mMeshes[iMesh]->mMaterialIndex];

			aiString name;
			assImpMaterial->Get(AI_MATKEY_NAME, name);
			aiColor3D kd(0.f,0.f,0.f);
			assImpMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, kd);
			aiColor3D ks(0.f,0.f,0.f);
			assImpMaterial->Get(AI_MATKEY_COLOR_SPECULAR, kd);
			aiColor3D ka(0.f,0.f,0.f);
			assImpMaterial->Get(AI_MATKEY_COLOR_AMBIENT, kd);

			aiString kdTexturePath;
			assImpMaterial->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), kdTexturePath);
			ID3D11ShaderResourceView* kdSrv = loadTexture(kdTexturePath.data);
			aiString kaTexturePath;
			assImpMaterial->Get(AI_MATKEY_TEXTURE_AMBIENT(0), kaTexturePath);
			ID3D11ShaderResourceView* kaSrv = loadTexture(kaTexturePath.data);
			aiString alphaTexturePath;
			assImpMaterial->Get(AI_MATKEY_TEXTURE_OPACITY(0), alphaTexturePath);
			ID3D11ShaderResourceView* alphaSrv = loadTexture(alphaTexturePath.data);
			aiString normalTexturePath;
			assImpMaterial->Get(AI_MATKEY_TEXTURE_HEIGHT(0), normalTexturePath);
			ID3D11ShaderResourceView* normalSrv = loadTexture(normalTexturePath.data);

			int aiShadingModel;
			assImpMaterial->Get(AI_MATKEY_SHADING_MODEL, aiShadingModel);
			
			if(aiShadingModel == 3)
			{
				Mesh::Material::P material = Mesh::Material::make(deferPass, 0);
				effect->GetVariableByName("kdColor")->AsVector()->SetFloatVector((float*)&kd);
				material->saveVariable( effect->GetVariableByName("kdColor") );
				effect->GetVariableByName("kdTexture")->AsShaderResource()->SetResource(kdSrv);
				material->saveVariable( effect->GetVariableByName("kdTexture") );
				effect->GetVariableByName("normalTexture")->AsShaderResource()->SetResource(normalSrv);
				material->saveVariable( effect->GetVariableByName("normalTexture") );
				effect->GetVariableByName("alphaTexture")->AsShaderResource()->SetResource(alphaSrv);
				material->saveVariable( effect->GetVariableByName("alphaTexture") );

				Mesh::Cast::P castMesh = Mesh::Cast::make();
				castMesh->add(deferRole,
					Mesh::Shaded::make(
						binder->bind(device, deferPassDesc, meshGeometry),
					material));

				Entity entity;
				D3DXMatrixIdentity(&entity.modelMatrix);
				entity.mesh = castMesh;
				entity.transparentMaterialIndex = 0;
				entities.push_back(entity);
			}
			else if(aiShadingModel == 9)
			{
				Mesh::Cast::P castMesh = Mesh::Cast::make();
				castMesh->add(storeRole,
					Mesh::Shaded::make(
						binder->bind(device, storePassDesc, meshGeometry),
						Mesh::Material::make(storePass, D3DX11_EFFECT_PASS_KEEP_UAV_COUNTERS))); // keep uav counters when setting this material

				Entity entity;
				D3DXMatrixIdentity(&entity.modelMatrix);
				entity.mesh = castMesh;
				entity.transparentMaterialIndex = 1 + (iMesh%2);
				entities.push_back(entity);
			}
			else
			{
				Mesh::Cast::P castMesh = Mesh::Cast::make();
				castMesh->add(storeRole,
					Mesh::Shaded::make(
						binder->bind(device, fluidPassDesc, meshGeometry),
						Mesh::Material::make(fluidPass, D3DX11_EFFECT_PASS_KEEP_UAV_COUNTERS))); // keep uav counters when setting this material

				Entity entity;
				D3DXMatrixIdentity(&entity.modelMatrix);
				entity.mesh = castMesh;
				entity.transparentMaterialIndex = 3;
				entities.push_back(entity);
			}
		}
	}
}

HRESULT Dxa::VolTransRaster::createResources()
{
	// gui
	Dxa::Sas::createResources();

	TogglerFunctor* freezeToggler = new TogglerFunctor(freeze, L"Play", L"Pause");
	freezeToggler->setButton(addButton(L"Pause", 0, 50, 150, 20, VK_BACK, false, freezeToggler)); 

	TogglerFunctor* particlesToggler = new TogglerFunctor(displayParticles, L"Switch to VolOIT", L"Swich to VolParticles");
	particlesToggler->setButton(addButton(L"Swich to VolParticles", 0, 80, 150, 20, VK_BACK, false, particlesToggler)); 

	TogglerFunctor* spheresOnlyToggler = new TogglerFunctor(drawSpheresOnly, L"Switch to spheres", L"Switch to dist imp");
	spheresOnlyToggler->setButton(addButton(L"Switch to dist imp", 0, 110, 150, 20, VK_BACK, false, spheresOnlyToggler)); 

	// textures
	D3DX11CreateShaderResourceViewFromFile( device, L"Media/jajdesert.dds", NULL, NULL, &envTextureSrv, NULL);
	D3DX11CreateShaderResourceViewFromFile( device, L"Media/puff2.dds", NULL, NULL, &puffTextureSrv, NULL);
	ID3D11ShaderResourceView* axeSrv;
	D3DX11CreateShaderResourceViewFromFile( device, L"Media/tiles.jpg", NULL, NULL, &axeSrv, NULL);
	srvs["axeTexture"] = axeSrv;

	// scene
	binder = new Mesh::Binder();

	// sponza
	importScene("sponza/sponge.obj");

	// ming heads
	Mesh::Cast::P mingMesh = loadMeshTransparent("media/ming.gbs");

	D3DXMATRIX tmpm; 

	Entity t0;
	D3DXMatrixScaling(&t0.modelMatrix, 6, 6, 6);
	D3DXMatrixRotationY(&tmpm, 0.0); t0.modelMatrix *= tmpm;
	D3DXMatrixTranslation(&tmpm, -500, 100, 0); (&tmpm, 1.58); t0.modelMatrix *= tmpm;
	t0.mesh = mingMesh;
	t0.transparentMaterialIndex = 5;
	entities.push_back(t0);

	Entity t1;
	D3DXMatrixScaling(&t1.modelMatrix, 5, 5, 5);
	D3DXMatrixRotationY(&tmpm, 0.0); t1.modelMatrix *= tmpm;
	D3DXMatrixTranslation(&tmpm, 50, 150, 50); (&tmpm, 1.58); t1.modelMatrix *= tmpm;

	t1.mesh = mingMesh;
	t1.transparentMaterialIndex = 6;
	entities.push_back(t1);

	effect->GetVariableByName("kdTexture")->AsShaderResource()->SetResource(axeSrv);

	/// slab with GpuPro cutout
	Mesh::Cast::P opakMesh = loadMeshOpaque("media/gpupro_adj.gbs", true);
	Entity opak;
	D3DXMatrixScaling(&opak.modelMatrix, 10, 10, 10);
	D3DXMatrixRotationX(&tmpm, 1.58); opak.modelMatrix *= tmpm;
	D3DXMatrixTranslation(&tmpm, -500, 260, 0); (&tmpm, 1.58); opak.modelMatrix *= tmpm;
	opak.mesh = opakMesh;
	opak.transparentMaterialIndex = 0;
	entities.push_back(opak);

	// full viewport quad
	Mesh::VertexStream::P quadVertices = Mesh::VertexStream::make(device, Mesh::VertexStreamDesc());
	Mesh::Indexed::P quadGeometry = Mesh::Indexed::make(device, Mesh::IndexBufferDesc(), quadVertices);

	quadMesh = Mesh::Cast::make();

	// quad will be rendered as background
	ID3DX11EffectPass* backgroundPass = effect->GetTechniqueByName("volumetricTransparency")->GetPassByName("background");
	D3DX11_PASS_DESC backgroundPassDesc;
	backgroundPass->GetDesc(&backgroundPassDesc);
	quadMesh->add(deferRole,
		Mesh::Shaded::make(
			binder->bind(device, backgroundPassDesc, quadGeometry),
			Mesh::Material::make(backgroundPass, 0)));

	// quad will be rendered to evaluate fragment lists
	ID3DX11EffectPass* sortPass = effect->GetTechniqueByName("volumetricTransparency")->GetPassByName("sortAndRender");
	D3DX11_PASS_DESC sortPassDesc;
	sortPass->GetDesc(&sortPassDesc);
	quadMesh->add(sortRole,
		Mesh::Shaded::make(
			binder->bind(device, sortPassDesc, quadGeometry),
			Mesh::Material::make(sortPass, 0)));

	/// 16x16 tiles
	Mesh::Instanced::P tileSetGeometry = Mesh::Instanced::make(device, 256, NULL, 0, quadGeometry);

	tileSet = Mesh::Cast::make();

	// tiles will be rendered to raycast particles
	ID3DX11EffectPass* smokePass = effect->GetTechniqueByName("smoke")->GetPassByName("smoke");
	D3DX11_PASS_DESC smokePassDesc;
	smokePass->GetDesc(&smokePassDesc);
	tileSet->add(smokeRole,
		Mesh::Shaded::make(
			binder->bind(device, smokePassDesc, tileSetGeometry),
			Mesh::Material::make(smokePass, 0)));
	ID3DX11EffectPass* smokeSpheresPass = effect->GetTechniqueByName("smoke")->GetPassByName("smokeSpheres");
	D3DX11_PASS_DESC smokeSpheresPassDesc;
	smokeSpheresPass->GetDesc(&smokeSpheresPassDesc);
	tileSet->add(sphereRole,
		Mesh::Shaded::make(
			binder->bind(device, smokeSpheresPassDesc, tileSetGeometry),
			Mesh::Material::make(smokeSpheresPass, 0)));

	// initilize particles
	for(int ip = 0; ip < nParticles; ip++)
	{
		particles[ip].reborn();
	}

	unsigned int iniz[256];
	for(unsigned int q=0; q<256; q++)
		iniz[q] = 0;
	D3D11_SUBRESOURCE_DATA srd;
	srd.pSysMem = iniz;

	D3D11_BUFFER_DESC tileParticleBufferDesc;
	tileParticleBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	tileParticleBufferDesc.ByteWidth = sizeof(unsigned int) * 256 * 4;
	tileParticleBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	tileParticleBufferDesc.MiscFlags = 0;
	tileParticleBufferDesc.Usage = D3D11_USAGE_DYNAMIC;

	device->CreateBuffer(&tileParticleBufferDesc, &srd, &tileParticleCountBuffer);
	effect->GetConstantBufferByName("tileParticleCountBuffer")->SetConstantBuffer(tileParticleCountBuffer);

	D3D11_TEXTURE2D_DESC particleTextureDesc;
	particleTextureDesc.ArraySize = 1;
	particleTextureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	particleTextureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	particleTextureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	particleTextureDesc.MipLevels = 1;
	particleTextureDesc.MiscFlags = 0;
	particleTextureDesc.Usage = D3D11_USAGE_DYNAMIC;
	particleTextureDesc.Width = 1024;
	particleTextureDesc.Height = 256;
	particleTextureDesc.SampleDesc.Count = 1;
	particleTextureDesc.SampleDesc.Quality = 0;

	device->CreateTexture2D(&particleTextureDesc, NULL, &particleTexture);

	D3D11_SHADER_RESOURCE_VIEW_DESC particleSrvDesc;
	particleSrvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	particleSrvDesc.Texture2D.MipLevels = 1;
	particleSrvDesc.Texture2D.MostDetailedMip = 0;
	particleSrvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;

	device->CreateShaderResourceView(particleTexture, &particleSrvDesc, &particleSrv);

	effect->GetVariableByName("particleTexture")->AsShaderResource()->SetResource(particleSrv);

	return S_OK;
}

HRESULT Dxa::VolTransRaster::releaseResources()
{
	tileParticleCountBuffer->Release();
	particleTexture->Release();
	particleSrv->Release();

	SrvDirectory::iterator i = srvs.begin();
	SrvDirectory::iterator e = srvs.end();
	while(i != e)
	{
		if(i->second)
			i->second->Release();
		i++;
	}

	envTextureSrv->Release();
	puffTextureSrv->Release();

	quadMesh.reset();
	entities.clear();

	delete binder;

	return Dxa::Sas::releaseResources();
}

HRESULT Dxa::VolTransRaster::createSwapChainResources()
{
	Dxa::Sas::createSwapChainResources();
	camera.SetProjParams(1.58, (float)backbufferSurfaceDesc.Width / backbufferSurfaceDesc.Height, 1, 10000 );

	D3DXVECTOR2 frameDimensions(backbufferSurfaceDesc.Width, backbufferSurfaceDesc.Height);
	effect->GetVariableByName("frameDimensions")->AsVector()->SetFloatVector((float*)&frameDimensions);

	D3D11_BUFFER_DESC bufferDesc;
	ZeroMemory( &bufferDesc, sizeof( D3D11_BUFFER_DESC) );
	bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;

	// Create Fragment and Link buffer.
	bufferDesc.StructureByteStride = sizeof(unsigned int) * 3;
	bufferDesc.ByteWidth = backbufferSurfaceDesc.Width * backbufferSurfaceDesc.Height * 16 * bufferDesc.StructureByteStride;
	bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	device->CreateBuffer( &bufferDesc, NULL, &fragmentLinkBuffer );

	// Create Start Offset buffer
	bufferDesc.StructureByteStride = sizeof(unsigned int);
	bufferDesc.ByteWidth = backbufferSurfaceDesc.Width * backbufferSurfaceDesc.Height * bufferDesc.StructureByteStride;
	bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
	device->CreateBuffer( &bufferDesc, NULL, &startOffsetBuffer );

	// Create Unordered Access Views
	D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
	uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	uavDesc.Buffer.FirstElement = 0;

	uavDesc.Format = DXGI_FORMAT_UNKNOWN;
	uavDesc.Buffer.NumElements = backbufferSurfaceDesc.Width * backbufferSurfaceDesc.Height * 16;
	uavDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_COUNTER;
	device->CreateUnorderedAccessView( fragmentLinkBuffer, &uavDesc, &fragmentLinkUav );

	uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
	uavDesc.Buffer.NumElements = backbufferSurfaceDesc.Width * backbufferSurfaceDesc.Height;
	uavDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
	device->CreateUnorderedAccessView( startOffsetBuffer, &uavDesc, &startOffsetUav );

	// Create Shader Resource Views
	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.FirstElement = 0;

	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.Buffer.NumElements = backbufferSurfaceDesc.Width * backbufferSurfaceDesc.Height * 16;
	device->CreateShaderResourceView( fragmentLinkBuffer, &srvDesc, &fragmentLinkSrv );

	srvDesc.Format = DXGI_FORMAT_R32_UINT;
	srvDesc.Buffer.NumElements = backbufferSurfaceDesc.Width * backbufferSurfaceDesc.Height;
	device->CreateShaderResourceView( startOffsetBuffer, &srvDesc, &startOffsetSrv );

// Texture and views for the opaque scene

	D3D11_TEXTURE2D_DESC opaqueTextureDesc;
	ZeroMemory( &opaqueTextureDesc, sizeof(opaqueTextureDesc) );
	opaqueTextureDesc.Width = backbufferSurfaceDesc.Width;
	opaqueTextureDesc.Height = backbufferSurfaceDesc.Height;
	opaqueTextureDesc.MipLevels = 1;
	opaqueTextureDesc.ArraySize = 1;
	opaqueTextureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	opaqueTextureDesc.SampleDesc.Count = 1;
	opaqueTextureDesc.Usage = D3D11_USAGE_DEFAULT;
	opaqueTextureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;	
	device->CreateTexture2D( &opaqueTextureDesc, NULL, &opaqueTexture );

	D3D11_RENDER_TARGET_VIEW_DESC opaqueRtvDesc;
	opaqueRtvDesc.Format = opaqueTextureDesc.Format;
	opaqueRtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	opaqueRtvDesc.Texture2D.MipSlice = 0;
	device->CreateRenderTargetView( opaqueTexture, &opaqueRtvDesc, &opaqueRtv );

	D3D11_SHADER_RESOURCE_VIEW_DESC opaqueSrvDesc;
	opaqueSrvDesc.Format = opaqueTextureDesc.Format;
	opaqueSrvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	opaqueSrvDesc.Texture2D.MostDetailedMip = 0;
	opaqueSrvDesc.Texture2D.MipLevels = 1;

	device->CreateShaderResourceView( opaqueTexture, &opaqueSrvDesc, &opaqueSrv );

	D3D11_TEXTURE2D_DESC nearPlaneTextureDesc;
	ZeroMemory( &nearPlaneTextureDesc, sizeof(nearPlaneTextureDesc) );
	nearPlaneTextureDesc.Width = backbufferSurfaceDesc.Width;
	nearPlaneTextureDesc.Height = backbufferSurfaceDesc.Height;
	nearPlaneTextureDesc.MipLevels = 1;
	nearPlaneTextureDesc.ArraySize = 1;
	nearPlaneTextureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	nearPlaneTextureDesc.SampleDesc.Count = 1;
	nearPlaneTextureDesc.Usage = D3D11_USAGE_DEFAULT;
	nearPlaneTextureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;	
	device->CreateTexture2D( &nearPlaneTextureDesc, NULL, &nearPlaneTexture );
	device->CreateTexture2D( &nearPlaneTextureDesc, NULL, &nearPlaneIrradianceTexture );

	D3D11_RENDER_TARGET_VIEW_DESC nearPlaneRtvDesc;
	nearPlaneRtvDesc.Format = nearPlaneTextureDesc.Format;
	nearPlaneRtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	nearPlaneRtvDesc.Texture2D.MipSlice = 0;
	device->CreateRenderTargetView( nearPlaneTexture, &nearPlaneRtvDesc, &nearPlaneRtv );
	device->CreateRenderTargetView( nearPlaneIrradianceTexture, &nearPlaneRtvDesc, &nearPlaneIrradianceRtv );


	D3D11_SHADER_RESOURCE_VIEW_DESC nearPlaneSrvDesc;
	nearPlaneSrvDesc.Format = nearPlaneTextureDesc.Format;
	nearPlaneSrvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	nearPlaneSrvDesc.Texture2D.MostDetailedMip = 0;
	nearPlaneSrvDesc.Texture2D.MipLevels = 1;
	device->CreateShaderResourceView( nearPlaneTexture, &nearPlaneSrvDesc, &nearPlaneSrv );
	device->CreateShaderResourceView( nearPlaneIrradianceTexture, &nearPlaneSrvDesc, &nearPlaneIrradianceSrv );

	return S_OK;
}

HRESULT Dxa::VolTransRaster::releaseSwapChainResources()
{
	opaqueTexture->Release();
	opaqueRtv->Release();
	opaqueSrv->Release();

	nearPlaneTexture->Release();
	nearPlaneRtv->Release();
	nearPlaneSrv->Release();

	nearPlaneIrradianceTexture->Release();
	nearPlaneIrradianceRtv->Release();
	nearPlaneIrradianceSrv->Release();

	fragmentLinkBuffer->Release();
	startOffsetBuffer->Release();
	fragmentLinkUav->Release();
	startOffsetUav->Release();
	fragmentLinkSrv->Release();
	startOffsetSrv->Release();
	return Dxa::Sas::releaseSwapChainResources();
}

void Dxa::VolTransRaster::animate(double dt, double t)
{
	Dxa::Sas::animate(dt, t);
	camera.FrameMove(dt);
	static double time = 0;
	if(!freeze)
		time += dt;
	effect->GetVariableByName("time")->AsScalar()->SetFloat(time);

	int displayedParticleCount= 1;
	effect->GetVariableByName("displayedParticleCount")->AsScalar()->GetInt(&displayedParticleCount);

	const D3DXVECTOR3& eye = *camera.GetEyePt();
	const D3DXVECTOR3& ahead = *camera.GetWorldAhead();
	const D3DXVECTOR3& right = *camera.GetWorldRight();
	const D3DXVECTOR3& up = *camera.GetWorldUp();

	D3DXMATRIX vpm;
	D3DXMATRIX eyePosTranslationMatrix;
	D3DXMatrixTranslation(&eyePosTranslationMatrix, eye.x, eye.y, eye.z);
	D3DXMatrixInverse(&vpm, NULL, &(eyePosTranslationMatrix * *camera.GetViewMatrix() * *camera.GetProjMatrix()));

	if(!freeze)
		for(int ip = 0; ip < displayedParticleCount; ip++)
		{
			particles[ip].move(dt);
		}
	for(int ip = 0; ip < displayedParticleCount; ip++)
	{
		particles[ip].recalculateDistance(eye);
	}

	// sort particles
	for(int j = 1; j < displayedParticleCount; j++)
	{
		for(int k = 0; k < displayedParticleCount-1; k++)	
		{
			float distk = particles[k].getDistanceFromEye();
			float distk1 = particles[k+1].getDistanceFromEye();
			if( distk > distk1)
			{
				Particle p = particles[k+1];
				particles[k+1] = particles[k];
				particles[k] = p;
			}
		}
	}

	ID3D11DeviceContext* context;
	device->GetImmediateContext(&context);

	D3D11_MAPPED_SUBRESOURCE mappedTileParticleCountBuffer;
	context->Map(tileParticleCountBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedTileParticleCountBuffer);
	unsigned int* tileParticleCounts = (unsigned int*)mappedTileParticleCountBuffer.pData;

	D3D11_MAPPED_SUBRESOURCE mappedTexture;
	context->Map(particleTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedTexture);
	D3DXVECTOR4* particleData = (D3DXVECTOR4*)mappedTexture.pData;

	float radiusScale = 1;
	effect->GetVariableByName("particleRadiusScale")->AsScalar()->GetFloat(&radiusScale);

	for(int v=0; v<16; v++)
	{
		D3DXVECTOR3 bottomPlane(0, (v / 8.0) - 1.0, 1);
		D3DXVec3TransformCoord(&bottomPlane, &bottomPlane, &vpm);
		D3DXVec3Cross(&bottomPlane, &bottomPlane, &right);
		D3DXVec3Normalize(&bottomPlane, &bottomPlane);
		D3DXVECTOR3 topPlane(0, ((v+1) / 8.0) - 1.0, 1);
		D3DXVec3TransformCoord(&topPlane, &topPlane, &vpm);
		D3DXVec3Cross(&topPlane, &right, &topPlane);
		D3DXVec3Normalize(&topPlane, &topPlane);
		for(int u=0; u<16; u++)
		{
			D3DXVECTOR3 leftPlane((u / 8.0) - 1.0, 0, 1);
			D3DXVec3TransformCoord(&leftPlane, &leftPlane, &vpm);
			D3DXVec3Cross(&leftPlane, &up, &leftPlane);
			D3DXVec3Normalize(&leftPlane, &leftPlane);
			D3DXVECTOR3 rightPlane(((u+1) / 8.0) - 1.0, 0, 1);
			D3DXVec3TransformCoord(&rightPlane, &rightPlane, &vpm);
			D3DXVec3Cross(&rightPlane, &rightPlane, &up);
			D3DXVec3Normalize(&rightPlane, &rightPlane);

			unsigned int cTileParticles = 0;
			D3DXVECTOR4* tileParticleData = (D3DXVECTOR4*)( (char*)mappedTexture.pData + mappedTexture.RowPitch * (v * 16 + u));
			for(int i=0; i<displayedParticleCount && cTileParticles < 338; i++)
			{
				D3DXVECTOR3 cccp = particles[i].getPosition() - eye;
				float nradius = -particles[i].getRadius();
				nradius *= radiusScale;
				// if beam through tile intersects sphere
				if(	D3DXVec3Dot(&leftPlane, &cccp ) > nradius
					&&	D3DXVec3Dot(&rightPlane, &cccp ) > nradius
					&&	D3DXVec3Dot(&bottomPlane, &cccp ) > nradius
					&&	D3DXVec3Dot(&topPlane, &cccp ) > nradius
					&&	D3DXVec3Dot(&ahead, &cccp ) > nradius
					
					)
				{
					*(tileParticleData++) = particles[i].getPositionAndRadius();
					*(tileParticleData++) = particles[i].getGAndTau();
					*(tileParticleData++) = particles[i].getOrientation();
					cTileParticles++;
				}
			}
			*(tileParticleData++) = D3DXVECTOR4(100000, 100000, 100000, 0);
			*(tileParticleData++) = D3DXVECTOR4(0, 0, 0, 0);

			tileParticleCounts[(v*16+u)*4] = cTileParticles;
		}
	}
	context->Unmap(particleTexture, 0);
	context->Unmap(tileParticleCountBuffer, 0);
	context->Release();
}

bool Dxa::VolTransRaster::processMessage( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if(Dxa::Sas::processMessage(hWnd, uMsg, wParam, lParam))
		return true;
	return camera.HandleMessages(hWnd, uMsg, wParam, lParam);
}

void Dxa::VolTransRaster::drawAll(ID3D11DeviceContext* context, Mesh::Role role)
{
	for(int iEntity=0; iEntity < entities.size(); iEntity++)
	{
		D3DXMATRIX modelMatrix = entities.at(iEntity).modelMatrix;
		D3DXMATRIX modelMatrixInverse;
		D3DXMatrixInverse(&modelMatrixInverse, NULL, &modelMatrix);
		D3DXMATRIX viewMatrix = *camera.GetViewMatrix();
		D3DXMATRIX projMatrix = *camera.GetProjMatrix();
		D3DXMATRIX viewProjMatrix = viewMatrix * projMatrix;
		D3DXMATRIX modelViewProjMatrix = modelMatrix * viewProjMatrix;

		effect->GetVariableByName("modelMatrix")->AsMatrix()->SetMatrix((float*)&modelMatrix);
		effect->GetVariableByName("modelMatrixInverse")->AsMatrix()->SetMatrix((float*)&modelMatrixInverse);
		effect->GetVariableByName("viewMatrix")->AsMatrix()->SetMatrix((float*)&viewMatrix);
		effect->GetVariableByName("viewProjMatrix")->AsMatrix()->SetMatrix((float*)&viewProjMatrix);
		effect->GetVariableByName("modelViewProjMatrix")->AsMatrix()->SetMatrix((float*)&modelViewProjMatrix);

		effect->GetVariableByName("materialId")->AsScalar()->SetInt( entities.at(iEntity).transparentMaterialIndex);

		entities.at(iEntity).mesh->draw(context, role);
	}
}


void Dxa::VolTransRaster::render(ID3D11DeviceContext* context)
{

	#pragma region model, view, proj
	D3DXMATRIX modelMatrix, modelMatrixInverse;
	D3DXMatrixIdentity(&modelMatrix);
	D3DXMatrixInverse(&modelMatrixInverse, NULL, &modelMatrix);
	D3DXMATRIX viewMatrix = *camera.GetViewMatrix();
	D3DXMATRIX projMatrix = *camera.GetProjMatrix();
	D3DXMATRIX modelViewProjMatrix = modelMatrix * viewMatrix * projMatrix;
	#pragma endregion compute matrices

	#pragma region effect
	effect->GetVariableByName("modelMatrix")->AsMatrix()->SetMatrix((float*)&modelMatrix);
	effect->GetVariableByName("modelMatrixInverse")->AsMatrix()->SetMatrix((float*)&modelMatrixInverse);
	effect->GetVariableByName("modelViewProjMatrix")->AsMatrix()->SetMatrix((float*)&modelViewProjMatrix);
	#pragma endregion set global effect variables

	#pragma region viewDir
	D3DXMATRIX eyeMatrix;
	D3DXMatrixTranslation(&eyeMatrix, camera.GetEyePt()->x, camera.GetEyePt()->y, camera.GetEyePt()->z);
	D3DXMATRIX eyeViewProjMatrix = eyeMatrix * viewMatrix * projMatrix;
	D3DXMATRIX viewDirMatrix;
	D3DXMatrixInverse(&viewDirMatrix, NULL, &eyeViewProjMatrix);
	effect->GetVariableByName("viewDirMatrix")->AsMatrix()->SetMatrix((float*)&viewDirMatrix);
	effect->GetVariableByName("eyePos")->AsVector()->SetFloatVector((float*)camera.GetEyePt());
	#pragma endregion compute screen-to-viewdir matrix

	#pragma region defaults
	ID3D11RenderTargetView* defaultRtv = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* defaultDsv = DXUTGetD3D11DepthStencilView();
	unsigned int nViewports = 8;
	D3D11_VIEWPORT viewports[8];
	context->RSGetViewports(&nViewports, viewports);
	#pragma endregion save default state

	#pragma region defer
	context->ClearDepthStencilView( defaultDsv, D3D11_CLEAR_DEPTH, 1.0, 0 );    

	context->OMSetRenderTargets( 1, &opaqueRtv, defaultDsv );

	effect->GetVariableByName("envTexture")->AsShaderResource()->SetResource(envTextureSrv);

	drawAll(context, deferRole);
	quadMesh->draw(context, deferRole);
	
	context->ClearState();
	context->RSSetViewports(nViewports, viewports);
	#pragma endregion deferring pass

	if(!displayParticles)
	{
		#pragma region store
		float nearPlaneClearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f } ; 
		float nearPlaneClearIrradiance[4] = { 1.0f, 1.0f, 1.0f, 0.0f } ; 
		context->ClearRenderTargetView( nearPlaneRtv, nearPlaneClearColor );
		context->ClearRenderTargetView( nearPlaneIrradianceRtv, nearPlaneClearIrradiance );
		static const unsigned int clearValue[1] = { 0x01ffffff };
		context->ClearUnorderedAccessViewUint( startOffsetUav, clearValue );

		// Bind UAVs.
		effect->GetVariableByName("fragmentLinkRwBuffer")->AsUnorderedAccessView()->SetUnorderedAccessView(fragmentLinkUav);
		effect->GetVariableByName("startOffsetRwBuffer")->AsUnorderedAccessView()->SetUnorderedAccessView(startOffsetUav);
		effect->GetVariableByName("opaqueTexture")->AsShaderResource()->SetResource(opaqueSrv);

		ID3D11RenderTargetView* npsrvs[] = {nearPlaneRtv, nearPlaneIrradianceRtv};
		context->OMSetRenderTargets( 2, npsrvs, NULL);

		effect->GetTechniqueByName("volumetricTransparency")->GetPassByName("storeFragments")->Apply(0, context); // reset counter	
		drawAll(context, storeRole);

		context->ClearState();
		context->RSSetViewports(nViewports, viewports);
		#pragma endregion storing pass

		#pragma region sort
		effect->GetVariableByName("fragmentLinkBuffer")->AsShaderResource()->SetResource(fragmentLinkSrv);
		effect->GetVariableByName("startOffsetBuffer")->AsShaderResource()->SetResource(startOffsetSrv);
		effect->GetVariableByName("nearPlaneTexture")->AsShaderResource()->SetResource(nearPlaneSrv);
		effect->GetVariableByName("nearPlaneIrradianceTexture")->AsShaderResource()->SetResource(nearPlaneIrradianceSrv);

		float clearColor[4] = { 0.3f, 0.2f, 0.9f, 0.0f };
		context->ClearRenderTargetView( defaultRtv, clearColor );
		context->ClearDepthStencilView( defaultDsv, D3D11_CLEAR_DEPTH, 1.0, 0 );
		context->OMSetRenderTargets( 1, &defaultRtv, defaultDsv );

		quadMesh->draw(context, sortRole);
		#pragma endregion sorting & rendering pass
	}
	else
	{
		effect->GetVariableByName("puffTexture")->AsShaderResource()->SetResource(puffTextureSrv);

		#pragma region raytrace
		effect->GetVariableByName("opaqueTexture")->AsShaderResource()->SetResource(opaqueSrv);
		float clearColor[4] = { 0.3f, 0.2f, 0.9f, 0.0f };
		context->ClearRenderTargetView( defaultRtv, clearColor );
		context->ClearDepthStencilView( defaultDsv, D3D11_CLEAR_DEPTH, 1.0, 0 );
		context->OMSetRenderTargets( 1, &defaultRtv, defaultDsv );

		if(drawSpheresOnly)
			tileSet->draw(context, smokeRole);
		else
			tileSet->draw(context, sphereRole);

		#pragma endregion render raytraced particle system
	}

	Dxa::Sas::render(context);
}

