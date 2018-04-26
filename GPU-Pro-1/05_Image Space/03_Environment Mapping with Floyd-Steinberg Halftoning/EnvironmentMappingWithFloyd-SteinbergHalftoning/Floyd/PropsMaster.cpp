#include "DXUT.h"
#include "PropsMaster.h"
#include "FileFinder.h"
#include "ShadedMesh.h"
#include "Rendition.h"
#include "XMLparser.h"
#include "Theatre.h"
#include "GeometryLoader.h"
#include "ShadedMesh.h"
#include "Repertoire.h"
#include "ShadingMaterial.h"

PropsMaster::PropsMaster(Theatre* theatre)
{
	this->theatre = theatre;
	createHardProps();
}

void PropsMaster::createHardProps()
{
	ID3D10Device* device = theatre->getDevice();

	// create full viewport quad mesh with name L"\4D71quad"
	ID3DX10Mesh* mesh;

	const D3D10_INPUT_ELEMENT_DESC quadElements[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 16, D3D10_INPUT_PER_VERTEX_DATA, 0 }
	};

	// Slide #15.1
	D3DX10CreateMesh(device, quadElements, 2, "POSITION", 4, 2, 0, &mesh);

	struct QuadVertex
	{
		D3DXVECTOR4 pos;
		D3DXVECTOR2 tex;
	} svQuad[4];
	static const float fSize = 1.0f;
	svQuad[0].pos = D3DXVECTOR4(-fSize, fSize, 0.0f, 1.0f);
	svQuad[0].tex = D3DXVECTOR2(0.0f, 0.0f);
	svQuad[1].pos = D3DXVECTOR4(fSize, fSize, 0.0f, 1.0f);
	svQuad[1].tex = D3DXVECTOR2(1.0f, 0.0f);
	svQuad[2].pos = D3DXVECTOR4(-fSize, -fSize, 0.0f, 1.0f);
	svQuad[2].tex = D3DXVECTOR2(0.0f, 1.0f);
	svQuad[3].pos = D3DXVECTOR4(fSize, -fSize, 0.0f, 1.0f);
	svQuad[3].tex = D3DXVECTOR2(1.0f, 1.0f);

	mesh->SetVertexData(0, (void*)svQuad);

	unsigned short siQuad[6];
	siQuad[0] = 0;
	siQuad[1] = 1;
	siQuad[2] = 2;
	siQuad[3] = 2;
	siQuad[4] = 1;
	siQuad[5] = 3;

	mesh->SetIndexData(siQuad, 6);
	mesh->CommitToDevice();
	meshDirectory[L"\4D71quad"] = mesh;

	GeometryLoader::LoadMeshFromFile(L"plays\\Standard\\media\\4D71ball.dgb", device, &mesh);
	meshDirectory[L"\4D71ball"] = mesh;
}

void PropsMaster::addProps(XMLNode& propsNode)
{
	loadMeshes(propsNode);
	loadRepertoires(propsNode);
	loadShadedMeshes(propsNode);
}

PropsMaster::~PropsMaster(void)
{
	repertoireDirectory.deleteAll();
	shaderResourceViewDirectory.releaseAll();
	meshDirectory.releaseAll();
	shadedMeshDirectory.deleteAll();
}

ID3D10ShaderResourceView* PropsMaster::loadTexture(XMLNode& textureNode)
{
	const wchar_t* textureFileName = textureNode|L"file";
	if(textureFileName == NULL)
	{
		MessageBox( NULL, L"Error.", L"No texture file name.", MB_OK);
		return NULL;
	}
	if(wcscmp(L"null", textureFileName) == 0)
		return NULL;
	ID3D10ShaderResourceView* texture;
	ShaderResourceViewDirectory::iterator iTex = shaderResourceViewDirectory.find(textureFileName);
	if(iTex == shaderResourceViewDirectory.end())
	{
		D3DX10_IMAGE_LOAD_INFO loadInfo;
		loadInfo.Width = D3DX10_DEFAULT;
		loadInfo.Height = D3DX10_DEFAULT;
		loadInfo.Depth = D3DX10_DEFAULT;
		loadInfo.FirstMipLevel = D3DX10_DEFAULT;
		loadInfo.Usage = D3D10_USAGE_IMMUTABLE;
		loadInfo.BindFlags = D3D10_BIND_SHADER_RESOURCE;
		loadInfo.CpuAccessFlags = 0;
		loadInfo.MiscFlags = D3DX10_DEFAULT;
		loadInfo.Format = DXGI_FORMAT_FROM_FILE;
		loadInfo.Filter = D3DX10_DEFAULT;
		loadInfo.MipFilter = D3DX10_DEFAULT;

		loadInfo.Format = textureNode.readFormat(L"format", DXGI_FORMAT_FROM_FILE);
		loadInfo.MipLevels = textureNode.readLong(L"mipLevels", D3DX_DEFAULT);

		std::wstring textureFilePath = theatre->getFileFinder()->completeFileName(textureFileName);
		HRESULT hr = D3DX10CreateShaderResourceViewFromFile( theatre->getDevice(), textureFilePath.c_str(), &loadInfo, NULL, &texture, NULL );

		if(hr != S_OK)
		{
			texture = NULL;
			MessageBox( NULL, L"Failed to load texture.", textureFilePath.c_str(), MB_OK);
		}
		shaderResourceViewDirectory[textureFileName] = texture;
	}
	else
		texture = iTex->second;
	return texture;
}

void PropsMaster::loadMeshes(XMLNode& xMainNode)
{
	int iMesh = 0;
	XMLNode meshNode;
	while( !(meshNode = xMainNode.getChildNode(L"Mesh", iMesh)).isEmpty() )
	{
		ID3DX10Mesh* mesh;
		const wchar_t* fileName = meshNode|L"fileName";
		if(fileName != NULL)
		{
			std::wstring meshFilePath = theatre->getFileFinder()->completeFileName(fileName);
			HRESULT hr = GeometryLoader::LoadMeshFromFile(meshFilePath.c_str(), theatre->getDevice(), &mesh);
			if(hr == S_OK)
			{
				const wchar_t* meshName = meshNode|L"name";
				if(meshName != NULL)
				{
					if(meshDirectory[meshName] == NULL)
						meshDirectory[meshName] = mesh;
					else
						EggXMLERR(meshNode, L"Duplicate mesh name: " << meshName);
				}
				else
					EggXMLERR(meshNode, L"No name specified for a mesh. Filename: " << fileName);
			}
			else
				EggXMLERR(meshNode, L"Failed to load mesh: " << fileName);
		}
		else
			EggXMLERR(meshNode, L"No file name specified for a mesh.");
		iMesh++;			
	}
}

void PropsMaster::loadShadedMeshes(XMLNode& xMainNode)
{
	int iShadedMesh = 0;
	XMLNode shadedMeshNode;
	while( !(shadedMeshNode = xMainNode.getChildNode(L"ShadedMesh", iShadedMesh)).isEmpty() )
	{
		const wchar_t* meshName = shadedMeshNode|L"mesh";
		if(meshName != NULL)
		{
			MeshDirectory::iterator iMesh = meshDirectory.find(meshName);
			if(iMesh != meshDirectory.end())
			{
				const wchar_t* shadedMeshName = shadedMeshNode|L"name";
				if(shadedMeshName != NULL)
				{
					ShadedMesh* shadedMesh = new ShadedMesh(iMesh->second);
					createShadingMaterials(shadedMeshNode, shadedMesh);
					if(shadedMeshDirectory[shadedMeshName] == NULL)
						shadedMeshDirectory[shadedMeshName] = shadedMesh;
					else
						EggXMLERR(shadedMeshNode, L"Duplicate ShadedMesh name: " << shadedMeshName);
				}
				else
					EggXMLERR(shadedMeshNode, L"No name specified for a ShadedMesh. ShadedMesh skipped.");
			}
			else
				EggXMLERR(shadedMeshNode, L"Unknown mesh specified for a ShadedMesh. Mesh name: " << meshName );
		}
		else
			EggXMLERR(shadedMeshNode, L"No mesh specified for a shadedMesh.");
		iShadedMesh++;
	}
}


ID3DX10Mesh* PropsMaster::getMesh(const std::wstring& name)
{
	if(!name.empty())
	{
		MeshDirectory::iterator iMesh = meshDirectory.find(name);
		if(iMesh != meshDirectory.end())
			return iMesh->second;
	}
	// warning
	return NULL;
}

ShadedMesh* PropsMaster::getShadedMesh(const std::wstring& name)
{
	if(!name.empty())
	{
		ShadedMeshDirectory::iterator iShadedMesh = shadedMeshDirectory.find(name);
		if(iShadedMesh != shadedMeshDirectory.end())
			return iShadedMesh->second;
	}
	EggERR(L"ShadedMesh " << name << " not found");
	return NULL;
}

void PropsMaster::createShadingMaterials(XMLNode& shadedMeshNode, ShadedMesh* shadedMesh)
{
	int iUseRepertoire = 0;
	XMLNode useRepertoireNode;
	while( !(useRepertoireNode = shadedMeshNode.getChildNode(L"use", iUseRepertoire)).isEmpty() )
	{
		const wchar_t* repertoireName = useRepertoireNode|L"repertoire";
		if(repertoireName)
		{
			RepertoireDirectory::iterator iRepertoire = repertoireDirectory.find(repertoireName);
			if(iRepertoire != repertoireDirectory.end())
			{
				Repertoire::RenditionDirectory::iterator iRendition = iRepertoire->second->begin();
				Repertoire::RenditionDirectory::iterator eRendition = iRepertoire->second->end();
				while(iRendition != eRendition)
				{
					ID3D10InputLayout* inputLayout = inputLayoutManager.getCompatibleInputLayout(
						theatre->getDevice(),
						iRendition->second->getTechnique(),
						shadedMesh->getMesh());
					shadedMesh->addShadingMaterial(
						iRendition->first,
						new ShadingMaterial(iRendition->second, inputLayout));
					iRendition++;
				}
			}
			else
				EggXMLERR(useRepertoireNode, "Unknown repertoire name: " << repertoireName);
		}
		else
			EggXMLERR(useRepertoireNode, "Missing repertoire tag in ShadedMesh.use.");
		iUseRepertoire++;
	}
	XMLNode overridesNode = shadedMeshNode.getChildNode(L"overrides");
	if(!overridesNode.isEmpty())
	{
		loadEffectSettings(overridesNode, shadedMesh->getOverrides());
	}
}

void PropsMaster::loadRenditions(XMLNode& repertoireNode, Repertoire* repertoire, Repertoire* baseRepertoire)
{
	int iRendition = 0;
	XMLNode renditionNode;
	while( !(renditionNode = repertoireNode.getChildNode(L"Rendition", iRendition)).isEmpty() )
	{
		const wchar_t* roleName = renditionNode|L"role";
		if(roleName != NULL)
		{
			Role role = theatre->getPlay()->getRole(roleName);
			if(role.isValid())
			{
				Rendition* rendition = NULL;

				bool replaceBase = renditionNode.readBool(L"replaceBase");
				if(baseRepertoire)
				{
					Rendition* baseRendition = baseRepertoire->getRendition(role);
					if(baseRendition && !replaceBase)
					{
						rendition = new Rendition(baseRendition);
					}
				}
				else
					replaceBase = true;
				
				if(replaceBase)
				{
					std::wstring effectName = renditionNode.readWString(L"effect");
					std::string techniqueName = renditionNode.readString(L"technique");
					ID3D10EffectTechnique* technique = theatre->getTechnique(effectName, techniqueName);
					rendition = new Rendition(theatre->getDevice(), technique);
				}

				repertoire->addRendition(role, rendition);

				loadEffectSettings(renditionNode, rendition);
			}
//			else
//				EggERR(L"WARNING: Unknown role provided. This is OK. [" << roleName << L"]");
		}
		else
		{
			EggXMLERR(renditionNode, L"No role name specified for a Rendition. Rendition skipped.");
		}
		iRendition++;
	}
}

void PropsMaster::loadEffectSettings(XMLNode& parentNode, EffectSettings* effectSettings)
{
	int iTexture = 0;
	XMLNode textureNode;
	while( !(textureNode = parentNode.getChildNode(L"Texture", iTexture)).isEmpty() )
	{
		ID3D10ShaderResourceView* texture = loadTexture(textureNode);
		std::wstring effectName = textureNode.readWString(L"effect");
		ID3D10Effect* effect = theatre->getEffect(effectName);
		
		effectSettings->setShaderResource(effect->GetVariableByName(textureNode.readString(L"name").c_str())->AsShaderResource(), texture);
		iTexture++;
	}

	int iVector = 0;
	XMLNode vectorNode;
	while( !(vectorNode = parentNode.getChildNode(L"Vector", iVector)).isEmpty() )
	{
		D3DXVECTOR4 value = vectorNode.readVector4(L"value");
		std::wstring effectName = textureNode.readWString(L"effect");
		ID3D10Effect* effect = theatre->getEffect(effectName);

		effectSettings->setVector(effect->GetVariableByName(vectorNode.readString(L"name").c_str())->AsVector(), value);
		iVector++;
	}
}

void PropsMaster::loadRepertoires(XMLNode& xMainNode)
{
	int iRepertoire = 0;
	XMLNode repertoireNode;
	while( !(repertoireNode = xMainNode.getChildNode(L"Repertoire", iRepertoire)).isEmpty() )
	{
		const wchar_t* repertoireName = repertoireNode|L"name";
		if(repertoireName != NULL)
		{
			Repertoire* baseRepertoire = NULL;
			const wchar_t* baseRepertoireName = repertoireNode|L"extends";

			if(baseRepertoireName != NULL)
			{
				RepertoireDirectory::iterator iBaseRepertoire = repertoireDirectory.find(baseRepertoireName);
				if(iBaseRepertoire != repertoireDirectory.end())
				{
					baseRepertoire = iBaseRepertoire->second;
				}
				else
					EggXMLERR(repertoireNode, L"Unknown base repertoire. New Repertoire created anyway. Base repertoire name: " << baseRepertoireName);
			}

			Repertoire* repertoire = new Repertoire();
			loadRenditions(repertoireNode, repertoire, baseRepertoire);
			if(repertoireDirectory[repertoireName] == NULL)
				repertoireDirectory[repertoireName] = repertoire;
			else
				EggXMLERR(repertoireNode, L"Duplicate Repertoire name: " << repertoireName);
		}
		else
			EggXMLERR(repertoireNode, L"Repertoire has no name. Skipped.");
		iRepertoire++;
	}
}