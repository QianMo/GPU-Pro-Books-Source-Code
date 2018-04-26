#include "DXUT.h"
#include "Theatre.h"
#include "Cueable.h"
#include "Scene.h"
#include "Act.h"
#include "XMLparser.h"
#include "ControlContext.h"
#include "MessageContext.h"
#include "ControlStatus.h"
#include "ResourceLocator.h"

Theatre::Theatre(ID3D10Device* device)
{
	this->device = device;
	play = NULL;
	loaded = false;
}

void Theatre::loadEffectPool(const std::wstring& fileName)
{
	ID3D10Blob* compilationErrors = NULL;
	if(FAILED(
		D3DX10CreateEffectPoolFromFileW(fileName.c_str(), NULL, NULL, "fx_4_0", 0, 0, device, NULL, &effectPool, &compilationErrors, NULL)))
	{
		if(!compilationErrors)
			EggERR(L"Effect pool file not found. [" << fileName << "]")
		else
			MessageBoxA( NULL, (LPSTR)compilationErrors->GetBufferPointer(), "Failed to load effect file!", MB_OK);
		exit(-1); // TODO: CLEAN EXIT
	}
	effects[L"pool"] = effectPool->AsEffect();
}

void Theatre::processMessage(HWND hWnd,  UINT uMsg, WPARAM wParam, LPARAM lParam, bool* trapped)
{
	if(loaded)
		play->processMessage(MessageContext(this, NULL, controlStatus, NULL, hWnd, uMsg, wParam, lParam, trapped));
	controlStatus.handleInput(hWnd, uMsg, wParam, lParam);
}

void Theatre::animate(double dt, double t)
{
	if(loaded)
	{
		play->control(controlStatus, dt);
		play->animate(dt, t);
	}
}

HRESULT Theatre::releaseResources()
{
	delete resourceLocator;
	delete play;
	EffectDirectory::iterator i = effects.begin();
	while(i != effects.end())
	{
		i->second->Release();
		i++;
	}
	return S_OK;
}

HRESULT Theatre::createResources()
{
	XMLNode playNode=XMLNode::openFileHelper(L"plays\\Floyd\\floyd^Play.xml", L"Play");

	loadPlay(playNode);
	resourceLocator = new ResourceLocator(this);
	return S_OK;
}


void Theatre::loadPlay(XMLNode& playNode)
{
	const wchar_t* effectPoolFile = playNode|L"effectPoolFile";
	if(!effectPoolFile) {EggXMLERR(playNode, L"Effect pool file not specified in play."); exit(-1); } // TODO: CLEAN EXIT
	loadEffectPool(effectPoolFile);
	int iChildEffect = 0;
	XMLNode childEffectNode;
	while( !(childEffectNode = playNode.getChildNode(L"use", iChildEffect)).isEmpty() )
	{
		const wchar_t* childEffectFile = childEffectNode|L"effectFile";
		const wchar_t* childEffectName = childEffectNode|L"effectName";

		if(childEffectFile && childEffectName)
		{
			loadChildEffect(childEffectFile, childEffectName);
		}
		iChildEffect++;
	}
	play = new Play(this);
	play->loadPlay(playNode);
	loaded = true;
}


void Theatre::render()
{
	if(!loaded)
		return;

	play->render();
}

HRESULT Theatre::createSwapChainResources()
{
	ID3D10RenderTargetView* swapChainRenderTargetView;
	device->OMGetRenderTargets(1, &swapChainRenderTargetView, NULL);
	ID3D10Texture2D* defaultRenderTargetResource;
	swapChainRenderTargetView->GetResource((ID3D10Resource**)&defaultRenderTargetResource);
	
	D3D10_TEXTURE2D_DESC defaultTexture2DDesc;
	defaultRenderTargetResource->GetDesc(&defaultTexture2DDesc);
	defaultRenderTargetResource->Release();
	swapChainRenderTargetView->Release();
	controlStatus.viewportWidth = defaultTexture2DDesc.Width;
	controlStatus.viewportHeight =  defaultTexture2DDesc.Height;

	play->assumeAspect();
	return S_OK;
}

HRESULT Theatre::releaseSwapChainResources()
{
	play->dropAspect();
	return S_OK;
}

const unsigned int Theatre::actActivate = 1;

void Theatre::loadChildEffect(const std::wstring& fileName, const std::wstring& name)
{
	ID3D10Blob* compilationErrors;
	ID3D10Effect* childEffect;
	if(FAILED(
		D3DX10CreateEffectFromFileW(fileName.c_str(), NULL, NULL, "fx_4_0", 0, D3D10_EFFECT_COMPILE_CHILD_EFFECT,
		device, effectPool, NULL, &childEffect, &compilationErrors, NULL)))
	{
		if(!compilationErrors)
			EggERR(L"Child effect file not found. [" << fileName << "]")
		else
			MessageBoxA( NULL, (LPSTR)compilationErrors->GetBufferPointer(), "Failed to load effect file!", MB_OK);
		exit(-1); // TODO: CLEAN EXIT
	}
	if(childEffect)
	{
		if(effects[name] == NULL)
			effects[name] = childEffect;
		else
			EggERR(L"Duplicate child effect name: " << name);
	}
}

ID3D10EffectTechnique* Theatre::getTechnique(const std::wstring& effectName, const std::string& techniqueName)
{
	ID3D10Effect* effect = getEffect(effectName);
	if(!effect)
		EggERR(L"Effect <" << effectName << "> unknown.");
	ID3D10EffectTechnique* technique = effect->GetTechniqueByName(techniqueName.c_str());
	if(!technique)
		EggERR(L"Technique <" << techniqueName.c_str() << "> in effect <" << effectName << "> not known.");
	return technique;
}