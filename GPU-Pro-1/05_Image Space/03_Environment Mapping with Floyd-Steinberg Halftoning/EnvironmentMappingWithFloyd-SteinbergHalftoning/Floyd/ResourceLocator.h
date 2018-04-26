#pragma once

class Theatre;
class ResourceOwner;
class ScriptVariable;
class ScriptResourceVariable;
class ScriptRenderTargetViewVariable;
class ScriptDepthStencilViewVariable;
class ScriptShaderResourceViewVariable;
class ScriptCustomVariable;

class ResourceLocator
{
	Theatre* theatre;
public:
	ResourceLocator(Theatre* theatre);
	~ResourceLocator(void);

	ScriptResourceVariable* getResourceVariable(const std::wstring& name, ResourceOwner* localResourceOwner);
	ScriptRenderTargetViewVariable* getRenderTargetViewVariable(const std::wstring& name, ResourceOwner* localResourceOwner);
	ScriptDepthStencilViewVariable* getDepthStencilViewVariable(const std::wstring& name, ResourceOwner* localResourceOwner);
	ScriptShaderResourceViewVariable* getShaderResourceViewVariable(const std::wstring& name, ResourceOwner* localResourceOwner);
};
