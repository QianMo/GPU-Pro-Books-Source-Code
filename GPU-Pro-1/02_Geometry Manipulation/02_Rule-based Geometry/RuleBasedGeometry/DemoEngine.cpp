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

#include "DXUT.h"
#include "DemoEngine.h"

#include <sstream>
#include <fstream>

#include "CodeGenerator.h"
#include "HLSLCodeGenerator.h"
#include "AxiomDescriptor.h"
#include "GrammarDescriptor.h"
#include "DXCPPCodeGenerator.h"
#include "XMLGrammarLoader.h"
#include "Ansi2Wide.h"
#include "GeometryLoader.h"

// this has to be done somewhere... :)
const IDType CodeGenerator::SYMBOLID_INVALID = 0;
const IDType CodeGenerator::SYMBOLID_START = 1;
const IDType CodeGenerator::RULEID_START = 1;
const IDType CodeGenerator::RULEID_NORULE = 1;
const unsigned int HLSLCodeGenerator::MAX_ROW_LENGTH = 50;
AttributeSet CodeGenerator::predefinedAttributes;
CodeGenerator::OperatorMap CodeGenerator::predefinedOperators;

DemoEngine::DemoEngine(ID3D10Device* device)
  :EngineInterface(device), 
  maxModuleNumber(2000000), maxDepth(0),
  defaultWorkingDirectory(NULL),
  generationLayout(NULL), instancingLayout(NULL), 
  axiomBuffer(NULL),srcBuffer(NULL),dstBuffer(NULL),instancedModuleBuffer(NULL),
  instanceMeshes(NULL),
  instancingPool(NULL), generationEffect(NULL), sortingEffect(NULL)
{
	ShaderCodeGenerator::initPredefined();

	axiomDesc = new AxiomDescriptor;
	grammarDesc = new GrammarDescriptor;
	grammarLoader = new XMLGrammarLoader(grammarDesc, axiomDesc);
	shaderCodeGenerator = new HLSLCodeGenerator(grammarDesc);
	cppCodeGenerator = new DXCPPCodeGenerator(grammarDesc);

	settings.enableAnimation = settings.enableInstancing = settings.enableModuleCulling = settings.drawInfos = true;

	DWORD length;
	length = GetCurrentDirectory( 0, NULL );
	defaultWorkingDirectory = new TCHAR[length];
	GetCurrentDirectory( length, defaultWorkingDirectory );
}

DemoEngine::~DemoEngine()
{
	SAFE_DELETE(axiomDesc);
	SAFE_DELETE(grammarDesc);
	SAFE_DELETE(grammarLoader);
	SAFE_DELETE(shaderCodeGenerator);
	SAFE_DELETE(cppCodeGenerator);
	SAFE_DELETE_ARRAY(defaultWorkingDirectory);
}

HRESULT DemoEngine::createResources()
{
	// camera
	D3DXVECTOR3 eye(0,0,-90);
	D3DXVECTOR3 lookAt(0,0,1);
	camera.SetViewParams(&eye, &lookAt);
	camera.SetScalers(0.01F,300.0F);

	// creating text drawing stuff
	D3DX10CreateFont( device, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
	OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
	L"Arial", &textFont );
	D3DX10CreateSprite( device, 8, &textSprite );
	textHelper = new CDXUTTextHelper(textFont, textSprite, 15);

	std::ifstream f("INPUT_NAME.cfg");
	if ( f.fail() )
	{
		MessageBoxA( NULL, "Failed to open INPUT_NAME.cfg!", "Failed to open INPUT_NAME.cfg!", MB_OK);
		exit(-1);
	}
	String inputName;
	f >> inputName;
	loadGrammar(inputName.c_str());

	loadTextures();

	return S_OK;
}

HRESULT DemoEngine::createSwapChainResources()
{
	DXGI_SWAP_CHAIN_DESC desc;
	swapChain->GetDesc(&desc);
	aspect = (float)desc.BufferDesc.Width / desc.BufferDesc.Height;
	fov = D3DX_PI / 2.0f;
	screenResolutionX = desc.BufferDesc.Width;
	screenResolutionY = desc.BufferDesc.Height;
	camera.SetProjParams( fov, aspect, 0.1f, 10000.0f);

	device->OMGetRenderTargets( 1, &defaultRenderTargetView, &defaultDepthStencilView );
	return S_OK;
}

HRESULT DemoEngine::releaseResources()
{
	releaseLayouts();
	releaseCoreEffects();
	releaseInstancingEffects();
	releaseBuffers();
	releaseMeshes();

	// release textures
	textureContainer.releaseAll();

	// release text drawing stuff
	delete textHelper;
	textFont->Release();
	textSprite->Release();

	return S_OK;
}

HRESULT DemoEngine::releaseSwapChainResources()
{
	defaultRenderTargetView->Release();
	defaultDepthStencilView->Release();
	return S_OK;
}

void DemoEngine::animate(double dt, double t)
{
	camera.FrameMove((float)dt);

	if ( settings.enableAnimation )
		generationEffect->GetVariableByName("currentTime")->AsScalar()->SetFloat((float)t);
}

void DemoEngine::processMessage( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	camera.HandleMessages(hWnd, uMsg, wParam, lParam);

	if(uMsg == WM_RBUTTONDOWN)
	{
		HMENU menu = CreatePopupMenu();
		AppendMenu(menu, 0, 1, L"Open");
		RECT rc;
		GetWindowRect(hWnd, &rc);
		int choice = (int)TrackPopupMenu(menu, TPM_RETURNCMD, rc.left + (lParam & 0xffff), rc.top + (lParam >> 16), 0, hWnd, NULL);
		if( 1 == choice )
			openFile(hWnd);
	}
}

void DemoEngine::handleKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
	if( bKeyDown )
	{
		switch( nChar )
		{
		// increasing generarion depth
		case VK_ADD:
			if ( axiomDesc->getSize()*pow((float)grammarDesc->getMaxRuleLength(),(float)maxDepth+1) < maxModuleNumber )
			{
				++maxDepth;
			}
			break;
		// decreasing generarion depth
		case VK_SUBTRACT:
			if ( maxDepth > 0 )
			{
				--maxDepth;
			}
			break;
		// toggle token cutting
		case VK_F1:
			settings.enableModuleCulling = ! settings.enableModuleCulling;
			generationEffect->GetVariableByName("enableModuleCulling")->AsScalar()->SetInt(settings.enableModuleCulling);
			break;
		// toggle drawing
		case VK_F2:
			settings.enableInstancing = ! settings.enableInstancing;
			break;
		// toggle animation
		case VK_F3:
			settings.enableAnimation = ! settings.enableAnimation;
			break;
		// toggle info drawing
		case VK_SPACE:
			settings.drawInfos = ! settings.drawInfos;

		default: break;
		}
	}
}

void DemoEngine::render()
{
	// Clear render target and the depth stencil 
	// float ClearColor[4] = { 0.176f, 0.196f, 0.667f, 0.0f };
	float ClearColor[4] = { 0.4f, 0.4f, 0.4f, 0.0f };
	device->ClearRenderTargetView( DXUTGetD3D10RenderTargetView(), ClearColor );
	device->ClearDepthStencilView( DXUTGetD3D10DepthStencilView(), D3D10_CLEAR_DEPTH, 1.0, 0 );

	setCameraParameters();

	// generating and rendering the procedural geometry
	unsigned int *instancedModuleNumbers = new unsigned int[grammarDesc->getInstancingTypeNumber()];
	unsigned int *instancedModuleOffsets = new unsigned int[grammarDesc->getInstancingTypeNumber()];
	generateGeometry();
	sortModules( instancedModuleNumbers, instancedModuleOffsets );
	instanceModules( instancedModuleNumbers, instancedModuleOffsets );

	drawInfos( instancedModuleNumbers );

	delete [] instancedModuleNumbers;
	delete [] instancedModuleOffsets;
}

void DemoEngine::createLayouts()
{
	DXCPPCodeGenerator::LayoutDescriptor generationDesc, instancingDesc;
	cppCodeGenerator->createInputLayouts(generationDesc, instancingDesc);

	// layout for generation and sorting
	{
		unsigned int nElements = generationDesc.size();
		D3D10_INPUT_ELEMENT_DESC* elements = new D3D10_INPUT_ELEMENT_DESC[nElements];
		for ( unsigned int i = 0; i < nElements; ++i )
		{
			elements[i] = generationDesc[i].first;
			elements[i].SemanticName = generationDesc[i].second.c_str();
		}

		D3D10_PASS_DESC passDesc;
		generationEffect->GetTechniqueByName("generateGeometry")->GetPassByIndex(0)->GetDesc(&passDesc);
		device->CreateInputLayout(elements, nElements, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &generationLayout);

		delete [] elements;
	}

	// layout for instancing
	if ( grammarDesc->getInstancingTypeNumber() != 0 )
	{
		const D3D10_INPUT_ELEMENT_DESC* meshElements;
		unsigned int nMeshElements;
		// for simplicity, we assume that every mesh has the same description
		instanceMeshes[0]->GetVertexDescription(&meshElements, &nMeshElements);

		unsigned int nInstanceElements = instancingDesc.size();
		D3D10_INPUT_ELEMENT_DESC* instanceElements = new D3D10_INPUT_ELEMENT_DESC[nInstanceElements];
		for ( unsigned int i = 0; i < nInstanceElements; ++i )
		{
			instanceElements[i] = instancingDesc[i].first;
			instanceElements[i].SemanticName = instancingDesc[i].second.c_str();
		}

		unsigned int nElements = nMeshElements + nInstanceElements;
		D3D10_INPUT_ELEMENT_DESC *elements = new D3D10_INPUT_ELEMENT_DESC[nElements];

		for ( int i = 0; i < (int)nMeshElements; ++i )
		{
			elements[i] = meshElements[i];
		}
		for ( int i = 0; i < (int)nInstanceElements; ++i )
		{
			elements[nMeshElements + i] = instanceElements[i];
		}

		D3D10_PASS_DESC passDesc;
		// for simplicity, we assume that every instancing shader has the same pass description
		ID3D10Effect* anInstancingEffect = instancingEffects.begin()->second;
		anInstancingEffect->GetTechniqueByName("instance")->GetPassByIndex(0)->GetDesc(&passDesc);
		device->CreateInputLayout(elements, nElements, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &instancingLayout);

		delete [] instanceElements;
		delete [] elements;
	}
}

void DemoEngine::createCoreEffects()
{
	ID3D10Blob* compilationErrors;
	if(FAILED(
		D3DX10CreateEffectFromFileW(L"generation.fx", NULL, NULL, "fx_4_0", 0, 0, device, NULL, NULL, &generationEffect, &compilationErrors, NULL)))
	{
		MessageBoxA( NULL, (LPSTR)compilationErrors->GetBufferPointer(), "Failed to load generation effect file!", MB_OK);
		exit(-1);
	}

	if(FAILED(
		D3DX10CreateEffectFromFileW(L"sorting.fx", NULL, NULL, "fx_4_0", 0, 0, device, NULL, NULL, &sortingEffect, &compilationErrors, NULL)))
	{
		MessageBoxA( NULL, (LPSTR)compilationErrors->GetBufferPointer(), "Failed to load sorting effect file!", MB_OK);
		exit(-1);
	}

	generationEffect->GetVariableByName("enableModuleCulling")->AsScalar()->SetInt(settings.enableModuleCulling);
}

void DemoEngine::createInstancingEffects()
{
	ID3D10Blob* compilationErrors;
	if(FAILED(
		D3DX10CreateEffectPoolFromFileW( L"instancing.fxh", 
                                              NULL,
                                              NULL, 
                                              "fx_4_0", 
                                              0,
                                              0,
                                              device,
                                              NULL,
                                              &instancingPool,
                                              &compilationErrors,
                                              NULL ) ) )
	{
		MessageBoxA( NULL, (LPSTR)compilationErrors->GetBufferPointer(), "Failed to load instancing effect pool file!", MB_OK);
		exit(-1);
	}

	instancingPool->AsEffect()->GetVariableByName("txFloor")->AsShaderResource()->SetResource( textureContainer.getTexture(L"media\\floor1.jpg") );
	instancingPool->AsEffect()->GetVariableByName("txWall")->AsShaderResource()->SetResource( textureContainer.getTexture(L"media\\wall1.jpg") );

	for ( unsigned int i = 0; i < grammarDesc->getInstancingTypeNumber(); ++i )
	{
		InstancingTypeCounter &type = grammarDesc->getInstancingType(i);
		// converting the name to wide character
		CAnsi2Wide stringConverter(type.instancingType.technique.c_str());
		WString techniqueName = stringConverter.operator LPCWSTR();
		WString effectName = techniqueName + L".fx";

		// avoid loading an effect twice
		if ( instancingEffects.count( type.instancingType.technique ) > 0 )
		{		
			continue;
		}

		if(FAILED(
			D3DX10CreateEffectFromFileW( effectName.c_str(),
			NULL,
			NULL,
			"fx_4_0",
			0,
			D3D10_EFFECT_COMPILE_CHILD_EFFECT,
			device,
			instancingPool,
			NULL,
			&instancingEffects[type.instancingType.technique],
			&compilationErrors,
			NULL)))
		{
			MessageBoxA( NULL, (LPSTR)compilationErrors->GetBufferPointer(), "Failed to load an instancing effect file!", MB_OK);
			exit(-1);
		}
	}
}

void DemoEngine::createBuffers()
{
	// creation of axiom buffer
	D3D10_BUFFER_DESC axiomBufferDesc =
	{
		axiomDesc->getSize()*sizeof(Module),
		D3D10_USAGE_DEFAULT,
		D3D10_BIND_VERTEX_BUFFER,
		0,
		0
	};

	D3D10_SUBRESOURCE_DATA srd;
	srd.pSysMem = (void*)axiomDesc->getDataPtr();
	device->CreateBuffer(&axiomBufferDesc, &srd, &axiomBuffer);

	// creation of generation buffers
	unsigned int maxGenerationDepth = (unsigned int)floor(log((float)(maxModuleNumber/axiomDesc->getSize())) /
		log((float)grammarDesc->getMaxRuleLength()));
	unsigned int currentMaxModules = 
		axiomDesc->getSize() * (unsigned int)pow((float)grammarDesc->getMaxRuleLength(),(float)maxGenerationDepth);

	D3D10_BUFFER_DESC instanceBufferDesc =
	{
		sizeof(Module)*currentMaxModules,
		D3D10_USAGE_DEFAULT,
		D3D10_BIND_STREAM_OUTPUT | D3D10_BIND_VERTEX_BUFFER,
		0,
		0
	};
	device->CreateBuffer(&instanceBufferDesc, NULL, &srcBuffer);
	device->CreateBuffer(&instanceBufferDesc, NULL, &dstBuffer);

	// creation of the buffer for storing sorted modules
	D3D10_BUFFER_DESC sortedBufferDesc =
	{
		sizeof(SortedModule)*currentMaxModules,
		D3D10_USAGE_DEFAULT,
		D3D10_BIND_STREAM_OUTPUT | D3D10_BIND_VERTEX_BUFFER,
		0,
		0
	};
	device->CreateBuffer(&sortedBufferDesc, NULL, &instancedModuleBuffer);
}

void DemoEngine::loadMeshes()
{
	instanceMeshes = new ID3DX10Mesh*[grammarDesc->getInstancingTypeNumber()];

	unsigned int index = 0;
	CAnsi2Wide stringConverter(grammarDesc->getMeshLibrary().c_str());
	WString meshLibrary = stringConverter.operator LPCWSTR();
	meshLibrary += L"\\";
	for ( unsigned int typeIndex = 0; typeIndex < grammarDesc->getInstancingTypeNumber(); ++typeIndex )
	{
		InstancingTypeCounter &type = grammarDesc->getInstancingType(typeIndex);
		
		CAnsi2Wide stringConverter(type.instancingType.meshName.c_str());
		WString meshName = stringConverter.operator LPCWSTR();
		meshName = meshLibrary + meshName;
		// TODO: a map should be used to avoid loading a mesh more than one times
		//       (different instancing types are allowed to have the same mesh)
		GeometryLoader::LoadMeshFromFile(meshName.c_str(), device, &instanceMeshes[index]);
		++index;
	}
}

void DemoEngine::loadGrammar( String inputName )
{
	releaseGrammar();

	grammarDesc->clear();
	axiomDesc->clear();
	grammarLoader->loadGrammar(inputName);
	SetCurrentDirectory( defaultWorkingDirectory );

	if ( cppCodeGenerator->moduleTypeFingerprint() != Module::moduleTypeFingerprint() )
	{
		MessageBoxA( NULL, "The module attribute type has been changed. Since the demo uses auto-generated CPU Module type, the CPU code needs to be recompiled for this grammar to work.", 
			"Recompile needed!", MB_OK);
		exit(1);
	}

	shaderCodeGenerator->generateCode();

	// NOTE: there are some depencies between these calls, so the following order should be kept
	createCoreEffects();
	createInstancingEffects();
	createBuffers();
	loadMeshes();
	createLayouts();

	maxDepth = grammarDesc->getGenerationDepth();
}

void DemoEngine::loadTextures()
{
	textureContainer.addTexture( L"media\\floor1.jpg", device );
	textureContainer.addTexture( L"media\\wall1.jpg", device );

	instancingPool->AsEffect()->GetVariableByName("txFloor")->AsShaderResource()->SetResource( textureContainer.getTexture(L"media\\floor1.jpg") );
	instancingPool->AsEffect()->GetVariableByName("txWall")->AsShaderResource()->SetResource( textureContainer.getTexture(L"media\\wall1.jpg") );
}

HRESULT DemoEngine::openFile(HWND hWnd)
{
	wchar_t fileName[512];
	fileName[0] = L'\0';
	OPENFILENAME of;
	of.lStructSize = sizeof(OPENFILENAME);
	of.hwndOwner = hWnd;
	of.hInstance = NULL;
	of.lpstrFilter = L"XML Grammar Descriptor\0*.xml\0\0";
	of.lpstrCustomFilter = NULL;
	of.nMaxCustFilter = 0;
	of.nFilterIndex = 0;
	of.lpstrFile = fileName;
	of.nMaxFile = 512;
	of.lpstrFileTitle = NULL;
	of.nMaxFileTitle = 0;
	of.lpstrInitialDir = L"grammars";
	of.lpstrTitle = NULL;
	of.Flags = 0;
	of.nFileOffset = 0;
	of.nFileExtension = 0;
	of.lpstrDefExt = NULL;
	of.lCustData = NULL;
	of.lpfnHook = NULL;
	of.lpTemplateName = NULL;
	of.pvReserved = NULL;
	of.dwReserved = NULL;
	of.FlagsEx = 0;

	if ( GetOpenFileName(&of) == TRUE )
	{
		if ( 1 == of.nFilterIndex )
		{
			unsigned int nChars = 0;
			nChars = WideCharToMultiByte(CP_ACP, 0, fileName, -1, NULL, 0, false, false);
			char *f = new char[nChars];
			WideCharToMultiByte(CP_ACP, 0, fileName, -1, f, nChars, false, false);
			loadGrammar( f );
			delete [] f;
			return S_OK;
		}
	}

	return E_FAIL;
}

void DemoEngine::releaseLayouts()
{
	SAFE_RELEASE( generationLayout );
	SAFE_RELEASE( instancingLayout );
}

void DemoEngine::releaseCoreEffects()
{
	SAFE_RELEASE( generationEffect );
	SAFE_RELEASE( sortingEffect );
}

void DemoEngine::releaseInstancingEffects()
{
	EffectMap::iterator it = instancingEffects.begin(), endit = instancingEffects.end();
	for ( ; it != endit; ++it)
		SAFE_RELEASE( it->second );
	instancingEffects.clear();
	SAFE_RELEASE( instancingPool );
}

void DemoEngine::releaseBuffers()
{
	SAFE_RELEASE( axiomBuffer );
	SAFE_RELEASE( srcBuffer );
	SAFE_RELEASE( dstBuffer );
	SAFE_RELEASE( instancedModuleBuffer );
}

void DemoEngine::releaseMeshes()
{
	if ( NULL == instanceMeshes ) return;

	for ( unsigned int i = 0; i < grammarDesc->getInstancingTypeNumber(); ++i )
	{
		SAFE_RELEASE(instanceMeshes[i]);
	}
	delete [] instanceMeshes;
}

void DemoEngine::releaseGrammar()
{
	releaseLayouts();
	releaseCoreEffects();
	releaseInstancingEffects();
	releaseBuffers();
	releaseMeshes();
}

void DemoEngine::generateGeometry()
{
	const unsigned int zeroOffset = 0;
	const unsigned int modulStrides = sizeof(Module);
	ID3D10EffectTechnique *generationTechnique = generationEffect->GetTechniqueByName("generateGeometry");
	const unsigned int axiomLength = axiomDesc->getSize();
	device->IASetInputLayout(generationLayout);
	device->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_POINTLIST);

	// the input of the first iteration is the axiom
	device->IASetVertexBuffers(0, 1, &axiomBuffer,
		&modulStrides, &zeroOffset);
	// apply generation shaders
	generationTechnique->GetPassByIndex(0)->Apply(0);
	// execute an iteration maxDepth times
	for ( unsigned int depth = 0; depth < maxDepth; ++depth )
	{
		// set dstBuffer as stream output
		device->SOSetTargets(1, &dstBuffer, &zeroOffset);
		// first draw call has to process the axiom
		if (0 == depth)
			device->Draw(axiomLength, 0);
		else
			device->DrawAuto();
		// ping pong
		ID3D10Buffer* nullBuffer = NULL;
		device->SOSetTargets(1, &nullBuffer, &zeroOffset);
		std::swap(dstBuffer, srcBuffer);

		// set srcBuffer as input
		device->IASetVertexBuffers(0, 1, &srcBuffer,
			&modulStrides, &zeroOffset);

		// NOTE: this may result in a small performance loss. however, generationLevel is a useful built-in
		//   variable for modeling (we can make level-dependent rule selections and attribute assignements), 
		//   so we added it.
		generationEffect->GetVariableByName("generationLevel")->AsScalar()->SetInt((int)depth);
		generationTechnique->GetPassByIndex(0)->Apply(0);
	}
}

void DemoEngine::sortModules( unsigned int* instancedModuleNumbers, unsigned int* instancedModuleOffsets )
{
	unsigned int offset = 0;
	ID3D10EffectVariable* minIDEffectVar = sortingEffect->GetVariableByName("minID");
	ID3D10EffectVariable* maxIDEffectVar = sortingEffect->GetVariableByName("maxID");
	ID3D10EffectTechnique* sortingTechnique = sortingEffect->GetTechniqueByName("sortModules");

	// query the number of emitted modules (for counting the number of generated modules per instancing type)
	D3D10_QUERY_DESC queryDesc;
	queryDesc.Query = D3D10_QUERY_SO_STATISTICS;
	queryDesc.MiscFlags = 0;
	ID3D10Query * pQuery;
	device->CreateQuery(&queryDesc, &pQuery);

	int minID;
	int maxID = grammarDesc->getSymbolIDStart() - 1;
	for( unsigned int i = 0; i < grammarDesc->getInstancingTypeNumber(); ++i)
	{
		pQuery->Begin();

		// set interval borders
		minID = maxID+1;
		maxID = minID - 1 + grammarDesc->getInstancingType(i).number;
		minIDEffectVar->AsScalar()->SetInt(minID);
		maxIDEffectVar->AsScalar()->SetInt(maxID);
		sortingTechnique->GetPassByIndex(0)->Apply(0);

		device->SOSetTargets(1, &instancedModuleBuffer, &offset);
		device->DrawAuto();

		// get the result of the query
		pQuery->End();
		D3D10_QUERY_DATA_SO_STATISTICS queryData;
		while(S_OK != pQuery->GetData(&queryData,
			sizeof(D3D10_QUERY_DATA_SO_STATISTICS), 0))
		{}

		instancedModuleNumbers[i] = (unsigned int)queryData.NumPrimitivesWritten;
		instancedModuleOffsets[i] = offset;
		offset += instancedModuleNumbers[i]*sizeof(SortedModule);
	}
	pQuery->Release();

	const unsigned int zeroOffset = 0;
	ID3D10Buffer* nullBuffer = NULL;
	device->SOSetTargets(1, &nullBuffer, &zeroOffset);
}

void DemoEngine::instanceModules( unsigned int* instancedModuleNumbers, unsigned int* instancedModuleOffsets )
{
	// if instancing is disabled, we do not instance the modules (but still generate them)
	if ( ! settings.enableInstancing ) return;
	// if there is no instance module, simply return
	if ( grammarDesc->getInstancingTypeNumber() == 0 )
		return;

	device->IASetInputLayout(instancingLayout);
	device->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	const unsigned int moduleStride = sizeof(SortedModule);
	for( unsigned int i = 0; i < grammarDesc->getInstancingTypeNumber(); ++i )
	{
		unsigned int offset = instancedModuleOffsets[i];

		// set instance buffers
		device->IASetVertexBuffers(1, 1, &instancedModuleBuffer,
			&moduleStride,
			&offset);
		// set the proper rendering technique
		setRenderingTechnique(i);
		// rendering the mesh, this sets the vertex buffer as well
		renderMesh(i,                        // instancing type ID
			instancedModuleNumbers[i]);      // number of instances
	}

  const unsigned int zeroOffset = 0;
	ID3D10Buffer* nullBuffer = NULL;
  device->IASetVertexBuffers(1,1,&nullBuffer,&moduleStride,&zeroOffset);
}

void DemoEngine::setRenderingTechnique( unsigned int instancingTypeIndex )
{
	RenderingTechniqueName techniqueName = 
		grammarDesc->getInstancingType( instancingTypeIndex ).instancingType.technique;
	ID3D10Effect* effect = instancingEffects[techniqueName];
	effect->GetTechniqueByName("instance")->GetPassByIndex(0)->Apply(0);
}

void DemoEngine::renderMesh( unsigned int instancingTypeIndex, unsigned int instanceNumber )
{
	instanceMeshes[instancingTypeIndex]->DrawSubsetInstanced( 0, instanceNumber, 0 );
}

void DemoEngine::setCameraParameters()
{
	D3DXMATRIX viewProjMatrix;
	viewProjMatrix = *camera.GetViewMatrix() * *camera.GetProjMatrix();
	instancingPool->AsEffect()->GetVariableByName("viewProjMatrix")->AsMatrix()->SetMatrix((float*)&viewProjMatrix);

	D3DXMATRIX modelMatrix;
	modelMatrix = *camera.GetViewMatrix() * *camera.GetWorldMatrix();
	D3DXMATRIX modelMatrixInverse;
	D3DXMatrixInverse( &modelMatrixInverse, NULL, &modelMatrix );
	instancingPool->AsEffect()->GetVariableByName("modelMatrixInverse")->AsMatrix()->SetMatrix((float*)&modelMatrixInverse);

	generationEffect->GetVariableByName("fov")->AsScalar()->SetFloat(fov);
	generationEffect->GetVariableByName("aspect")->AsScalar()->SetFloat(aspect);
	int screenResolution[2] = { screenResolutionX, screenResolutionY };
	generationEffect->GetVariableByName("screenResolution")->AsVector()->SetIntVector(screenResolution);
	generationEffect->GetVariableByName("cameraPos")->AsVector()->SetFloatVector((float*)camera.GetEyePt());
	D3DXVECTOR3 unitViewVector;
	D3DXVECTOR3 lookAtVector = *camera.GetLookAtPt() - *camera.GetEyePt();
	D3DXVec3Normalize( &unitViewVector, &lookAtVector );
	generationEffect->GetVariableByName("unitViewVector")->AsVector()->SetFloatVector((float*)(&unitViewVector));

	// Left clipping plane
	D3DXVECTOR4 p_planes[6];
	p_planes[0].x = viewProjMatrix._14 + viewProjMatrix._11;
	p_planes[0].y = viewProjMatrix._24 + viewProjMatrix._21;
	p_planes[0].z = viewProjMatrix._34 + viewProjMatrix._31;
	p_planes[0].w = viewProjMatrix._44 + viewProjMatrix._41;
	// Right clipping plane
	p_planes[1].x = viewProjMatrix._14 - viewProjMatrix._11;
	p_planes[1].y = viewProjMatrix._24 - viewProjMatrix._21;
	p_planes[1].z = viewProjMatrix._34 - viewProjMatrix._31;
	p_planes[1].w = viewProjMatrix._44 - viewProjMatrix._41;
	// Top clipping plane
	p_planes[2].x = viewProjMatrix._14 - viewProjMatrix._12;
	p_planes[2].y = viewProjMatrix._24 - viewProjMatrix._22;
	p_planes[2].z = viewProjMatrix._34 - viewProjMatrix._32;
	p_planes[2].w = viewProjMatrix._44 - viewProjMatrix._42;
	// Bottom clipping plane
	p_planes[3].x = viewProjMatrix._14 + viewProjMatrix._12;
	p_planes[3].y = viewProjMatrix._24 + viewProjMatrix._22;
	p_planes[3].z = viewProjMatrix._34 + viewProjMatrix._32;
	p_planes[3].w = viewProjMatrix._44 + viewProjMatrix._42;
	// Near clipping plane
	p_planes[4].x = viewProjMatrix._13;
	p_planes[4].y = viewProjMatrix._23;
	p_planes[4].z = viewProjMatrix._33;
	p_planes[4].w = viewProjMatrix._43;
	// Far clipping plane
	p_planes[5].x = viewProjMatrix._14 - viewProjMatrix._13;
	p_planes[5].y = viewProjMatrix._24 - viewProjMatrix._23;
	p_planes[5].z = viewProjMatrix._34 - viewProjMatrix._33;
	p_planes[5].w = viewProjMatrix._44 - viewProjMatrix._43;

	D3DXVECTOR3 tmp;
	float length;
	// normalize plane normals
	for ( int i = 0; i < 6; ++i )
	{
		tmp.x = p_planes[i].x;
		tmp.y = p_planes[i].y;
		tmp.z = p_planes[i].z;
		length = D3DXVec3Length( &tmp );
		p_planes[i].x /= length;
		p_planes[i].y /= length;
		p_planes[i].z /= length;
		p_planes[i].w /= length;
	}

	generationEffect->GetVariableByName("p_left")->AsVector()->SetFloatVector((float*)p_planes[0]);
	generationEffect->GetVariableByName("p_right")->AsVector()->SetFloatVector((float*)p_planes[1]);
	generationEffect->GetVariableByName("p_top")->AsVector()->SetFloatVector((float*)p_planes[2]);
	generationEffect->GetVariableByName("p_bottom")->AsVector()->SetFloatVector((float*)p_planes[3]);
	generationEffect->GetVariableByName("p_near")->AsVector()->SetFloatVector((float*)p_planes[4]);
	generationEffect->GetVariableByName("p_far")->AsVector()->SetFloatVector((float*)p_planes[5]);
}

std::wstring bool2OnOffStr(bool b)
{
	if ( b )
		return L"on";
	return L"off";
}

void DemoEngine::drawInfos( unsigned int *instancedModuleNumbers )
{
	if ( ! settings.drawInfos ) return;

	// drawing text
	textHelper->Begin();
	textHelper->SetInsertionPos( 2, 0 );
	textHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 0.6f, 0.4f, 1.0f ) );

	// stats, FPS
	textHelper->DrawTextLine( DXUTGetFrameStats(true) );
	textHelper->DrawTextLine( DXUTGetDeviceStats() );

	// generation depth
	{
		unsigned int maximumAllowedDepth = (unsigned int)floor(log((float)(maxModuleNumber/axiomDesc->getSize())) /
			log((float)grammarDesc->getMaxRuleLength()));
		std::wstringstream strStream;
		strStream << "Generation depth (current/maximum allowed): " << maxDepth << "/" << maximumAllowedDepth
			<< "  (NUMPAD +/- : increase / decrease generation depth )";
		textHelper->DrawTextLine( strStream.str().c_str() );
	}
	long unsigned int vertexCnt = 0;
	// instance number
	{
		unsigned int instanceCnt = 0;
		for ( unsigned int i = 0; i < grammarDesc->getInstancingTypeNumber(); ++i )
		{
			instanceCnt += instancedModuleNumbers[i];
			vertexCnt += instancedModuleNumbers[i] * instanceMeshes[i]->GetFaceCount();
		}
		std::wstringstream strStream;
		strStream << "Number of generated instances: " << instanceCnt;
		textHelper->DrawTextLine( strStream.str().c_str() );
	}
	// total triangle number at the instancing phase
	{
		if ( ! settings.enableInstancing ) vertexCnt = 0;
		std::wstringstream strStream;
		strStream << "Number of drawn triangles: " << vertexCnt << " (" << vertexCnt / 1000000 << "M)";
		textHelper->DrawTextLine( strStream.str().c_str() );
	}
	// settings
	{
		std::wstringstream strStream;
		strStream << "Module culling: " << bool2OnOffStr( settings.enableModuleCulling )
			<< "  (F1: toggle culling)";
		textHelper->DrawTextLine( strStream.str().c_str() );
	}
	{
		std::wstringstream strStream;
		strStream << "Module instancing: " << bool2OnOffStr( settings.enableInstancing )
			<< "  (F2: toggle instancing)";
		textHelper->DrawTextLine( strStream.str().c_str() );
	}
	{
		std::wstringstream strStream;
		strStream << "Animation (upload elapsed time to the GPU to use it in time-dependent rules): " << bool2OnOffStr( settings.enableAnimation )
			<< "  (F3: toggle animation)";
		textHelper->DrawTextLine( strStream.str().c_str() );
	}
	{
		std::wstringstream strStream;
		strStream << L"(W,S,A,D,E,Q: move camera; RCLICK: load grammar file; SPACE: show/hide infos)";
		textHelper->DrawTextLine( strStream.str().c_str() );
	}

	textHelper->End();
}
