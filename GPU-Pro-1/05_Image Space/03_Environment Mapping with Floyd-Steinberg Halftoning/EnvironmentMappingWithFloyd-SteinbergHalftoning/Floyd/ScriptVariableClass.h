#pragma once

class ScriptVariable;
class ScriptResourceVariable;
class ScriptShaderResourceViewVariable;
class ScriptRenderTargetViewVariable;
class ScriptDepthStencilViewVariable;
class ScriptBlobVariable;
class ScriptCustomVariable;
class ScriptCameraVariable;

class ScriptVariableClass
{
	typedef std::map<const std::wstring, const ScriptVariableClass*> ScriptVariableClassInstanceMap;

	static ScriptVariableClassInstanceMap names;

	const unsigned int id;
	ScriptVariableClass(unsigned int id, const wchar_t* typeName);

public:
	bool operator==(const ScriptVariableClass& o) const;

	static const ScriptVariableClass Resource;
	static const ScriptVariableClass ShaderResourceView;
	static const ScriptVariableClass RenderTargetView;
	static const ScriptVariableClass DepthStencilView;
	static const ScriptVariableClass Blob;
	static const ScriptVariableClass Custom;
	static const ScriptVariableClass Camera;
	static const ScriptVariableClass Unknown;

	ScriptVariable* instantiate() const;

	static const ScriptVariableClass& fromString(const std::wstring& typeName);

};
