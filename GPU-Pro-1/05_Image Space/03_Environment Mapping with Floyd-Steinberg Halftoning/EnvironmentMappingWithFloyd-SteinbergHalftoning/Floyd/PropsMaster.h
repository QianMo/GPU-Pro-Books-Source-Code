#pragma once

#include "Directory.h"
#include "InputLayoutManager.h"

class Theatre;
class EffectSettings;
class XMLNode;

class PropsMaster
{
	Theatre* theatre;

	/// D3D mesh references accessible by unicode name.
	MeshDirectory		meshDirectory;

	/// D3D resource view references accessible by ANSI name.
	ShaderResourceViewDirectory	shaderResourceViewDirectory;

	InputLayoutManager inputLayoutManager;
	RepertoireDirectory	repertoireDirectory;

	/// ShadedMesh references accessible by unicode name.
	ShadedMeshDirectory	shadedMeshDirectory;

	/// Processes all 'mesh' tags in the XML node, creating meshes and ShadedMesh instances based on .x file rendition data.
	void loadMeshes(XMLNode& xMainNode);

	void createHeightFieldMeshes(XMLNode& xMainNode);

	ID3DX10Mesh* createMeshFromHeightField(NxHeightField* heightField);

	void loadRepertoires(XMLNode& xMainNode);

	/// Processes all 'shadedMesh' tags in the XML node, creating ShadedMesh instances.
	void loadShadedMeshes(XMLNode& xMainNode);

public:

	PropsMaster(Theatre* theatre);
	~PropsMaster(void);

	void createHardProps();
	void addProps(XMLNode& propsNode);

	ID3DX10Mesh* getMesh(const std::wstring& name);
	ShadedMesh* getShadedMesh(const std::wstring& name);

	void loadRenditions(XMLNode& repertoireNode, Repertoire* repertoire, Repertoire* baseRepertoire);

	// createShadingMaterials is public to support creating additional ShadedMeshes outside of PropsMaster, in particular in Quad
	void createShadingMaterials(XMLNode& shadedMeshNode, ShadedMesh* shadedMesh);

	void loadEffectSettings(XMLNode& parentNode, EffectSettings* effectSettings);

	/// Returns a texture stored in the texture directory, after loading it if not already loaded.
	ID3D10ShaderResourceView* loadTexture(XMLNode& textureNode);

};
