#include "DXUT.h"
#include "ResourceSet.h"
#include "ScriptResourceVariable.h"
#include "ScriptRenderTargetViewVariable.h"
#include "ScriptDepthStencilViewVariable.h"
#include "ScriptShaderResourceViewVariable.h"
#include "ScriptCustomVariable.h"
#include "ScriptVariableClass.h"


ResourceSet::ResourceSet()
{
}

ResourceSet::~ResourceSet(void)
{
//	renderTargetViewVariableDirectory.deleteAll();
//	depthStencilViewVariableDirectory.deleteAll();
//	shaderResourceViewVariableDirectory.deleteAll();
//	resourceVariableDirectory.deleteAll();
//	customVariableDirectory.deleteAll();
	
	variableDirectories.deleteAll();

}

void ResourceSet::addResource(const std::wstring& name, ID3D10Resource* resource, bool swapChainBound)
{
	ScriptResourceVariable* var = (ScriptResourceVariable*)getVariable(
		ScriptVariableClass::Resource, 
		name);
	var->setResource(resource);
	if(swapChainBound)
		swapChainHardlinks.push_back(var);
	else
		hardlinks.push_back(var);
}

void ResourceSet::addRenderTargetView(const std::wstring& name, ID3D10RenderTargetView* resource, bool swapChainBound)
{
	ScriptRenderTargetViewVariable* var = (ScriptRenderTargetViewVariable*)getVariable(
		ScriptVariableClass::RenderTargetView, 
		name);
	var->setRenderTargetView(resource);
	if(swapChainBound)
		swapChainHardlinks.push_back(var);
	else
		hardlinks.push_back(var);
}

void ResourceSet::addDepthStencilView(const std::wstring& name, ID3D10DepthStencilView* resource, bool swapChainBound)
{
	ScriptDepthStencilViewVariable* var = (ScriptDepthStencilViewVariable*)getVariable(
		ScriptVariableClass::DepthStencilView,
		name);
	var->setDepthStencilView(resource);
	if(swapChainBound)
		swapChainHardlinks.push_back(var);
	else
		hardlinks.push_back(var);
}

void ResourceSet::addShaderResourceView(const std::wstring& name, ID3D10ShaderResourceView* resource, bool swapChainBound)
{
	ScriptShaderResourceViewVariable* var = (ScriptShaderResourceViewVariable*)
		getVariable(
			ScriptVariableClass::ShaderResourceView, 
			name);
	var->setShaderResourceView(resource);
	if(swapChainBound)
		swapChainHardlinks.push_back(var);
	else
		hardlinks.push_back(var);
}


ScriptResourceVariable* ResourceSet::getResourceVariable(const std::wstring& name)
{
	return (ScriptResourceVariable*)
		getVariable(ScriptVariableClass::Resource, name);
}

ScriptRenderTargetViewVariable* ResourceSet::getRenderTargetViewVariable(const std::wstring& name)
{
	return (ScriptRenderTargetViewVariable*)
		getVariable(ScriptVariableClass::RenderTargetView, name);
}

ScriptDepthStencilViewVariable* ResourceSet::getDepthStencilViewVariable(const std::wstring& name)
{
	return (ScriptDepthStencilViewVariable*)
		getVariable(ScriptVariableClass::DepthStencilView, name);
}

ScriptShaderResourceViewVariable* ResourceSet::getShaderResourceViewVariable(const std::wstring& name)
{
	return (ScriptShaderResourceViewVariable*)
		getVariable(ScriptVariableClass::ShaderResourceView, name);
}
/*
ScriptRenderTargetViewVariable*	ResourceSet::createRenderTargetViewVariable(const std::wstring& name)
{
	ScriptRenderTargetViewVariable* var = new ScriptRenderTargetViewVariable(NULL);
	renderTargetViewVariableDirectory[name] = var;
	return var;
}


ScriptDepthStencilViewVariable* ResourceSet::createDepthStencilViewVariable(const std::wstring& name)
{
	ScriptDepthStencilViewVariable* var = new ScriptDepthStencilViewVariable(NULL);
	depthStencilViewVariableDirectory[name] = var;
	return var;
}

ScriptShaderResourceViewVariable* ResourceSet::createShaderResourceViewVariable(const std::wstring& name)
{
	ScriptShaderResourceViewVariable* var = new ScriptShaderResourceViewVariable(NULL);
	shaderResourceViewVariableDirectory[name] = var;
	return var;
}

ScriptResourceVariable* ResourceSet::createResourceVariable(const std::wstring& name)
{
	ScriptResourceVariable* var = new ScriptResourceVariable(NULL);
	resourceVariableDirectory[name] = var;
	return var;
}
*/

void ResourceSet::releaseResources()
{
	std::vector<ScriptVariable*>::iterator i = hardlinks.begin();
	std::vector<ScriptVariable*>::iterator e = hardlinks.end();
	while(i != e)
	{
		(*i)->releaseResource();
		i++;
	}
}

void ResourceSet::releaseSwapChainResources()
{
	std::vector<ScriptVariable*>::iterator i = swapChainHardlinks.begin();
	std::vector<ScriptVariable*>::iterator e = swapChainHardlinks.end();
	while(i != e)
	{
		(*i)->releaseResource();
		i++;
	}
}


ScriptVariable*	ResourceSet::createVariable(const ScriptVariableClass& type, const std::wstring& name)
{
	ScriptVariable* var = type.instantiate();
	ScriptVariable*& pvar = variableDirectories[&type][name];
	if(pvar)
		delete pvar;
	pvar = var;
	return var;
}

ScriptVariable*	ResourceSet::getVariable(const ScriptVariableClass& type, const std::wstring& name)
{
	VariableDirectory::iterator i = variableDirectories[&type].find(name);
	if(i != variableDirectories[&type].end())
		return i->second;
	return NULL;
}
