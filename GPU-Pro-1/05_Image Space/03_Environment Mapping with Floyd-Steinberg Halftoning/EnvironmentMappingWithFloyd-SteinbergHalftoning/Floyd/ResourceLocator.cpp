#include "DXUT.h"
#include "ResourceLocator.h"
#include "ResourceOwner.h"
#include "ResourceSet.h"

ResourceLocator::ResourceLocator(Theatre* theatre)
{
	this->theatre = theatre;
}

ResourceLocator::~ResourceLocator(void)
{
}

ScriptResourceVariable* ResourceLocator::getResourceVariable(const std::wstring& name, ResourceOwner* localResourceOwner)
{
	return localResourceOwner->getResourceSet()->getResourceVariable(name);
}

ScriptRenderTargetViewVariable* ResourceLocator::getRenderTargetViewVariable(const std::wstring& name, ResourceOwner* localResourceOwner)
{
	return localResourceOwner->getResourceSet()->getRenderTargetViewVariable(name);

}

ScriptDepthStencilViewVariable* ResourceLocator::getDepthStencilViewVariable(const std::wstring& name, ResourceOwner* localResourceOwner)
{
	return localResourceOwner->getResourceSet()->getDepthStencilViewVariable(name);

}

ScriptShaderResourceViewVariable* ResourceLocator::getShaderResourceViewVariable(const std::wstring& name, ResourceOwner* localResourceOwner)
{
	return localResourceOwner->getResourceSet()->getShaderResourceViewVariable(name);
}
