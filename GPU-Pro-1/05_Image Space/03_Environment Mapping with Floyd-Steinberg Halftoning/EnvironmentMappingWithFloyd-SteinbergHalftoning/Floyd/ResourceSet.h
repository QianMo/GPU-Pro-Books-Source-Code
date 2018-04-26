#pragma once
#include "Directory.h"

class ScriptVariableClass;
class ScriptVariable;
class ScriptResourceVariable;
class ScriptRenderTargetViewVariable;
class ScriptDepthStencilViewVariable;
class ScriptShaderResourceViewVariable;
class ScriptCustomVariable;

class ResourceSet
{
//	typedef CompositMap<const std::wstring, ScriptResourceVariable*> ResourceVariableDirectory;
//	typedef CompositMap<const std::wstring, ScriptRenderTargetViewVariable*> RenderTargetViewVariableDirectory;
//	typedef CompositMap<const std::wstring, ScriptDepthStencilViewVariable*> DepthStencilViewVariableDirectory;
//	typedef CompositMap<const std::wstring, ScriptShaderResourceViewVariable*> ShaderResourceViewVariableDirectory;
//	typedef CompositMap<const std::wstring, ScriptCustomVariable*> CustomVariableDirectory;

	typedef CompositMap<const std::wstring, ScriptVariable*> VariableDirectory;

	std::vector<ScriptVariable*> hardlinks;
	std::vector<ScriptVariable*> swapChainHardlinks;

//	RenderTargetViewVariableDirectory renderTargetViewVariableDirectory;
//	DepthStencilViewVariableDirectory depthStencilViewVariableDirectory;
//	ShaderResourceViewVariableDirectory shaderResourceViewVariableDirectory;
//	ResourceVariableDirectory resourceVariableDirectory;
//	CustomVariableDirectory customVariableDirectory;

	CompositMapList<const ScriptVariableClass*, VariableDirectory> variableDirectories;
public:
	ResourceSet();
	~ResourceSet(void);

	void addRenderTargetView(const std::wstring& name, ID3D10RenderTargetView* resource, bool swapChainBound);
	void addDepthStencilView(const std::wstring& name, ID3D10DepthStencilView* resource, bool swapChainBound);
	void addShaderResourceView(const std::wstring& name, ID3D10ShaderResourceView* resource, bool swapChainBound);
	void addResource(const std::wstring& name, ID3D10Resource* resource, bool swapChainBound);

	ScriptVariable*	createVariable(const ScriptVariableClass& type, const std::wstring& name);
	ScriptVariable*	getVariable(const ScriptVariableClass& type, const std::wstring& name);

//	ScriptRenderTargetViewVariable*	createRenderTargetViewVariable(const std::wstring& name);
//	ScriptDepthStencilViewVariable* createDepthStencilViewVariable(const std::wstring& name);
//	ScriptShaderResourceViewVariable* createShaderResourceViewVariable(const std::wstring& name);
//	ScriptResourceVariable* createResourceVariable(const std::wstring& name);

	ScriptResourceVariable* getResourceVariable(const std::wstring& name);
	ScriptRenderTargetViewVariable* getRenderTargetViewVariable(const std::wstring& name);
	ScriptDepthStencilViewVariable* getDepthStencilViewVariable(const std::wstring& name);
	ScriptShaderResourceViewVariable* getShaderResourceViewVariable(const std::wstring& name);

	void releaseResources();
	void releaseSwapChainResources();
};
