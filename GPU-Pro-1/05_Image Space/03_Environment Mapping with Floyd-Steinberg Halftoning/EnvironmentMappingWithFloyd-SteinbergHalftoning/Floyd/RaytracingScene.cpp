#include "DXUT.h"
#include "RaytracingScene.h"
#include "RaytracingMesh.h"
#include "RaytracingEntity.h"
#include "BIHBuilder.h"
#include "RenderContext.h"
#include "Theatre.h"
#include "Scene.h"
#include "xmlParser.h"
#include "Camera.h"
#include "ShadedEntity.h"
#include "ShadedMesh.h"
#include "PropsMaster.h"

RaytracingScene::RaytracingScene(Theatre* theatre, XMLNode& xMainNode)
:SceneManager(theatre)
{
}

RaytracingScene::~RaytracingScene()
{
	raytracingMeshList.deleteAll();

	if(raytracingBIHNodeTableArraySRV)
		raytracingBIHNodeTableArraySRV->Release();
	if(raytracingTriangleTableArraySRV)
		raytracingTriangleTableArraySRV->Release();

	if(raytracingBIHNodeTableArray)
		raytracingBIHNodeTableArray->Release();
	if(raytracingTriangleTableArray)
		raytracingTriangleTableArray->Release();

	if(raytracingEntityBuffer)
		raytracingEntityBuffer->Release();

}

void RaytracingScene::render(const RenderContext& context)
{
	updateRaytracingResources();
}

void RaytracingScene::animate(double dt, double t)
{
}

void RaytracingScene::control(const ControlContext& context)
{
}

void RaytracingScene::processMessage( const MessageContext& context)
{
}

void RaytracingScene::createRaytracingResources()
{
	if(raytracingMeshList.size() == 0)
	{
		raytracingBIHNodeTableArraySRV = NULL;
		raytracingTriangleTableArraySRV = NULL;
		raytracingBIHNodeTableArray = NULL;
		raytracingTriangleTableArray = NULL;
		raytracingEntityBuffer = NULL;
		return;
	}
	unsigned int sysMemNodeTextureByteSize = 8192 * 8; //DXGI_FORMAT_R16G16B16A16_SNORM
	unsigned int sysMemTriangleTextureByteSize = 8192 * 4 * 12; //DXGI_FORMAT_R32G32B32_FLOAT
//	unsigned int sysMemTriangleTextureByteSize = 8192 * 64 * 12; //DXGI_FORMAT_R32G32B32_FLOAT
	char* sysMemNodeData = new char[raytracingMeshList.size() * sysMemNodeTextureByteSize];
	char* sysMemTriangleData = new char[raytracingMeshList.size() * sysMemTriangleTextureByteSize];

	unsigned int iRaytracingMesh = 0;
	RaytracingMeshList::iterator i = raytracingMeshList.begin();
	while(i != raytracingMeshList.end())
	{
		const D3D10_INPUT_ELEMENT_DESC* elements;
		unsigned int nElements;
		(*i)->getMesh()->GetVertexDescription(&elements, &nElements);
		unsigned int positionByteOffset = 0;
		unsigned int normalByteOffset = 12;
		for(unsigned int u=0; u<nElements; u++)
		{
			if(strcmp(elements[u].SemanticName, "POSITION") == 0)
				positionByteOffset = elements[u].AlignedByteOffset;
			if(strcmp(elements[u].SemanticName, "NORMAL") == 0)
				normalByteOffset = elements[u].AlignedByteOffset;
		}

		ID3DX10MeshBuffer* vertexBuffer;
		ID3DX10MeshBuffer* indexBuffer;
		(*i)->getMesh()->GetVertexBuffer(0, &vertexBuffer);
		(*i)->getMesh()->GetIndexBuffer(&indexBuffer);
		unsigned int vertexBufferByteSize;
		BYTE* vertexArray;
		vertexBuffer->Map((void**)&vertexArray, (SIZE_T*)&vertexBufferByteSize);
		unsigned int vertexByteSize = vertexBufferByteSize / (*i)->getMesh()->GetVertexCount();
		unsigned int indexBufferByteSize;
		unsigned short* indexArray;
		indexBuffer->Map((void**)&indexArray, (SIZE_T*)&indexBufferByteSize);

		BIHBuilder builder(
			indexArray, 0,(*i)->getMesh()->GetFaceCount(),
			vertexArray, vertexByteSize, positionByteOffset, normalByteOffset, 
			(void*)(sysMemNodeData + sysMemNodeTextureByteSize * iRaytracingMesh),
			(void*)(sysMemTriangleData + sysMemTriangleTextureByteSize * iRaytracingMesh));

		(*i)->setMatrices( builder.unitizer, builder.deunitizer);

		vertexBuffer->Unmap();
		indexBuffer->Unmap();
		vertexBuffer->Release();
		indexBuffer->Release();

		iRaytracingMesh++;
		i++;
	}

	D3D10_TEXTURE1D_DESC bihNodeTableArrayDesc;
	bihNodeTableArrayDesc.ArraySize = raytracingMeshList.size();
	bihNodeTableArrayDesc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	bihNodeTableArrayDesc.CPUAccessFlags = 0;
	bihNodeTableArrayDesc.Format = DXGI_FORMAT_R16G16B16A16_SNORM;
	bihNodeTableArrayDesc.MipLevels = 1;
	bihNodeTableArrayDesc.MiscFlags = 0;
	bihNodeTableArrayDesc.Usage = D3D10_USAGE_IMMUTABLE;
	bihNodeTableArrayDesc.Width = 8192;

	D3D10_SUBRESOURCE_DATA* initialData = new D3D10_SUBRESOURCE_DATA[raytracingMeshList.size()];
	for(unsigned int iFiller = 0; iFiller < raytracingMeshList.size(); iFiller++)
		initialData[iFiller].pSysMem = sysMemNodeData + iFiller * sysMemNodeTextureByteSize;

	getTheatre()->getDevice()->CreateTexture1D(&bihNodeTableArrayDesc, initialData, &raytracingBIHNodeTableArray);

	D3D10_TEXTURE2D_DESC triangleTableArrayDesc;
	triangleTableArrayDesc.ArraySize = raytracingMeshList.size();
	triangleTableArrayDesc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	triangleTableArrayDesc.CPUAccessFlags = 0;
	triangleTableArrayDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
	triangleTableArrayDesc.MipLevels = 1;
	triangleTableArrayDesc.MiscFlags = 0;
	triangleTableArrayDesc.Usage = D3D10_USAGE_IMMUTABLE;
	triangleTableArrayDesc.Width = 8192;
	triangleTableArrayDesc.Height = 4;
	triangleTableArrayDesc.SampleDesc.Count = 1;
	triangleTableArrayDesc.SampleDesc.Quality = 0;


	for(unsigned int iFiller = 0; iFiller < raytracingMeshList.size(); iFiller++)
	{
		initialData[iFiller].pSysMem = sysMemTriangleData + iFiller * sysMemTriangleTextureByteSize;
		initialData[iFiller].SysMemPitch = sysMemTriangleTextureByteSize / 4;
	}

	getTheatre()->getDevice()->CreateTexture2D(&triangleTableArrayDesc, initialData, &raytracingTriangleTableArray);

	delete [] initialData;

	D3D10_SHADER_RESOURCE_VIEW_DESC viewDesc;
	viewDesc.Format = DXGI_FORMAT_R16G16B16A16_SNORM;
	viewDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE1DARRAY;
	viewDesc.Texture1DArray.ArraySize = raytracingMeshList.size();
	viewDesc.Texture1DArray.FirstArraySlice = 0;
	viewDesc.Texture1DArray.MipLevels = 1;
	viewDesc.Texture1DArray.MostDetailedMip = 0;
	getTheatre()->getDevice()->CreateShaderResourceView(raytracingBIHNodeTableArray, &viewDesc, &raytracingBIHNodeTableArraySRV);

	D3D10_SHADER_RESOURCE_VIEW_DESC triViewDesc;
	triViewDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
	triViewDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
	triViewDesc.Texture2DArray.ArraySize = raytracingMeshList.size();
	triViewDesc.Texture2DArray.FirstArraySlice = 0;
	triViewDesc.Texture2DArray.MipLevels = 1;
	triViewDesc.Texture2DArray.MostDetailedMip = 0;

	getTheatre()->getDevice()->CreateShaderResourceView(raytracingTriangleTableArray, &triViewDesc, &raytracingTriangleTableArraySRV);

	/// Global effect variable to be bound to raytraceBIHNodeTableArray.
	ID3D10EffectShaderResourceVariable* bihNodeTableArrayEffectVariable;
	/// Global effect variable to be bound to raytraceTriangleTableArray.
	ID3D10EffectShaderResourceVariable* triangleTableArrayEffectVariable;

	bihNodeTableArrayEffectVariable = getTheatre()->getEffect(L"pool")->GetVariableByName("nodeTableArray")->AsShaderResource();
	triangleTableArrayEffectVariable = getTheatre()->getEffect(L"pool")->GetVariableByName("triangleTableArray")->AsShaderResource();

	bihNodeTableArrayEffectVariable->SetResource(raytracingBIHNodeTableArraySRV);
	triangleTableArrayEffectVariable->SetResource(raytracingTriangleTableArraySRV);

	delete [] sysMemNodeData;
	delete [] sysMemTriangleData;

	D3D10_BUFFER_DESC bufferDesc;
	bufferDesc.BindFlags = D3D10_BIND_CONSTANT_BUFFER;
	bufferDesc.CPUAccessFlags = D3D10_CPU_ACCESS_WRITE;
	bufferDesc.MiscFlags = 0;
	bufferDesc.Usage = D3D10_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = sizeof(unsigned int) * sizeof(RaytracingEntityData) * raytracingEntityList.size();
	getTheatre()->getDevice()->CreateBuffer(&bufferDesc, NULL, &raytracingEntityBuffer);

	ID3D10EffectConstantBuffer* entityBufferEffectVariable;

	entityBufferEffectVariable = getTheatre()->getEffect(L"pool")->GetConstantBufferByName("entityBuffer");
	entityBufferEffectVariable->SetConstantBuffer(raytracingEntityBuffer);
	updateRaytracingResources();
}


void RaytracingScene::updateRaytracingResources()
{
	char* entityData;
	raytracingEntityBuffer->Map(D3D10_MAP_WRITE_DISCARD, 0, (void**)&entityData);

	*(unsigned int*)entityData = raytracingEntityList.size();
	RaytracingEntityData* entities = (RaytracingEntityData*)(entityData + 4 * sizeof(unsigned int));
	RaytracingEntityList::iterator iEntity = raytracingEntityList.begin();
	unsigned int cRayTraceEntity=0;
	while(iEntity != raytracingEntityList.end())
	{
		RaytracingMesh* raytracingMesh = (*iEntity)->getRaytracingMesh();
		if(raytracingMesh)
		{
			unsigned int raytracingMeshIndex = raytracingMesh->getIndex();

			entities[cRayTraceEntity].meshIndex = raytracingMeshIndex;
			entities[cRayTraceEntity].diffuse = raytracingMesh->getRaytracingDiffuseBrdfParameter();
			entities[cRayTraceEntity].specular = raytracingMesh->getRaytracingSpecularBrdfParameter();
			D3DXMATRIX modelMatrix;
			(*iEntity)->getModelMatrix(modelMatrix);
			entities[cRayTraceEntity].modelMatrix = modelMatrix;
			D3DXMatrixTranspose(&entities[cRayTraceEntity].modelMatrix, &entities[cRayTraceEntity].modelMatrix);
			D3DXMATRIX modelMatrixInverse;
			(*iEntity)->getModelMatrixInverse(modelMatrixInverse);
			entities[cRayTraceEntity].modelMatrixInverse = modelMatrixInverse * raytracingMesh->getUnitizerMatrix();
			entities[cRayTraceEntity].modelMatrix = modelMatrixInverse;
			D3DXMatrixTranspose(&entities[cRayTraceEntity].modelMatrixInverse, &entities[cRayTraceEntity].modelMatrixInverse);
		}
		cRayTraceEntity++;
		iEntity++;
	}
	raytracingEntityBuffer->Unmap();	
}


Entity* RaytracingScene::decorateEntity(Entity* entity, XMLNode& entityNode, bool& processed)
{
	if(wcscmp(entityNode.getName(), L"RaytracingEntity")==0)
	{
		ShadedEntity* shadedEntity = EntityClass::asShadedEntity(entity);
		if(shadedEntity == NULL)
		{
			EggXMLERR(entityNode, L"Entity decorated by RaytracingEntity could not be cast to ShadedEntity.");
			return entity;
		}
		const wchar_t* meshName = entityNode|L"mesh";
		ID3DX10Mesh* mesh = NULL;
		if(meshName)
			mesh = getTheatre()->getPropsMaster()->getMesh(meshName);
		if(mesh == NULL)
			mesh = shadedEntity->getShadedMesh()->getMesh();
		RaytracingMesh* raytracingMesh = new RaytracingMesh(mesh, raytracingMeshList.size());
		raytracingMeshList.push_back(raytracingMesh);

		RaytracingEntity* raytracingEntity = new RaytracingEntity(entity);
		raytracingEntity->setRaytracingMesh(raytracingMesh);
		raytracingEntityList.push_back(raytracingEntity);
		processed = true;
		return raytracingEntity;
	}
	return entity;
}

void RaytracingScene::finish()
{
	createRaytracingResources();
}

