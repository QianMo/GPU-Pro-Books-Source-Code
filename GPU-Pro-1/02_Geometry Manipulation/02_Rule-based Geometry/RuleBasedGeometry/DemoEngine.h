/*
**********************************************************************
 * Demo program for
 * Rule-based Geometry Synthesis in Real-time
 * ShaderX 8 article.
 *
 * @author: Milan Magdics
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted for any non-commercial programs.
 * 
 * Use it for your own risk. The author(s) do(es) not take
 * responsibility or liability for the damages or harms caused by
 * this software.
**********************************************************************
*/

#pragma once
#include "engineinterface.h"
#include "DXUTCamera.h"
#include "SDKmisc.h"
#include "GrammarBasicTypes.h"
#include "TextureContainer.h"


const unsigned int TEXTURE_NUMBER = 2;

// settings of the demo
struct Settings
{
	bool enableModuleCulling;
	bool enableInstancing;
	bool enableAnimation;
	bool drawInfos;
};

///////////////////////////////////////////////////////////
// DemoEngine class - rendering engine of the demo program
///////////////////////////////////////////////////////////

class AxiomDescriptor;
class GrammarDescriptor;
class HLSLCodeGenerator;
class DXCPPCodeGenerator;
class XMLGrammarLoader;

class DemoEngine :
	public EngineInterface
{
public:
	typedef std::map<RenderingTechniqueName, ID3D10Effect*> EffectMap;
public:
	DemoEngine(ID3D10Device* device);
	virtual ~DemoEngine();
	
	virtual HRESULT createResources();
	virtual HRESULT createSwapChainResources();
	virtual HRESULT releaseResources();
	virtual HRESULT releaseSwapChainResources();

	virtual void animate(double dt, double t);
	virtual void processMessage( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	virtual void handleKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
	virtual void render();

protected:
	// resource creation
	void createLayouts();
	void createCoreEffects();
	void createInstancingEffects();
	void createBuffers();
	void loadMeshes();
	void loadGrammar( String inputName );
	void loadTextures();

	HRESULT openFile(HWND hWnd);

	// resource release/free
	void releaseLayouts();
	void releaseCoreEffects();
	void releaseInstancingEffects();
	void releaseBuffers();
	void releaseMeshes();
	void releaseGrammar();

	// rendering
	void generateGeometry();
	void sortModules( unsigned int* instancedModuleNumbers, unsigned int* instancedModuleOffsets );
	void instanceModules( unsigned int* instancedModuleNumbers, unsigned int* instancedModuleOffsets );
	void setRenderingTechnique( unsigned int instancingTypeIndex );
	void renderMesh( unsigned int instancingTypeIndex, unsigned int instanceNumber );
	void setCameraParameters();
	void drawInfos( unsigned int *instancedModuleNumbers );
protected:
	// an upper limit for the number of generated modules
	// - set this to fit your graphics card's memory
	// - HINT: storing maxModuleNumber modules requires sizeof(Module)*maxModuleNumber bytes of memory,
	//   the program needs an axiomBuffer (having size: sizeof(Module)*grammarDesc->axiomLength()),
	//   2 buffers for generation and sorting (having size of 
	//   grammarDesc->axiomLength()*sizeof(Module)*grammarDesc->getMaxRuleLength()^maxGenerationDepth)
	//   1 buffer for instance data
	unsigned int maxModuleNumber;
	// maximum generation depth. can be modified in run-time (via keyboard presses)
	unsigned int maxDepth;

	// settings of the demo
	Settings settings;

	// default working directory of the program
	LPTSTR defaultWorkingDirectory;

	// camera and camera parameters
	CFirstPersonCamera camera;
	float fov;
	float aspect;
	int screenResolutionX;
	int screenResolutionY;

	// input assembler layouts
	ID3D10InputLayout* generationLayout;
	ID3D10InputLayout* instancingLayout;

	ID3D10RenderTargetView* defaultRenderTargetView;
	ID3D10DepthStencilView* defaultDepthStencilView;

	// Text objects for rendering texts.
	CDXUTTextHelper* textHelper;
	ID3DX10Font* textFont;
	ID3DX10Sprite* textSprite;

	// Grammar Management stuff
	AxiomDescriptor *axiomDesc;
	GrammarDescriptor *grammarDesc;
	HLSLCodeGenerator *shaderCodeGenerator;
	DXCPPCodeGenerator *cppCodeGenerator;
	XMLGrammarLoader *grammarLoader;

	// vertex buffers to store module data
	                                        // buffer for axiom data (same in every frame)
	                                        // - TIP: for more advanced behavior (e.g. higher level cullings on the CPU)
	ID3D10Buffer* axiomBuffer;              //   it can be refreshed time-to-time
	// buffers for generation (ping-pong)
	ID3D10Buffer* srcBuffer;                // source buffer, used as input during the generation phase
	ID3D10Buffer* dstBuffer;                // output buffer for the generation phase
	// instance buffer, it stores the sorted modules
	ID3D10Buffer* instancedModuleBuffer;

	// instance meshes for instancing types
	ID3DX10Mesh** instanceMeshes;

	EffectMap instancingEffects;
	ID3D10Effect* instancingPoolEffect;
	ID3D10EffectPool* instancingPool;

	ID3D10Effect* generationEffect;
	ID3D10Effect* sortingEffect;

	// textures
	TextureContainer textureContainer;
	ID3D10Texture2D *textures[TEXTURE_NUMBER];
	ID3D10ShaderResourceView *texSRVs[TEXTURE_NUMBER];
};
