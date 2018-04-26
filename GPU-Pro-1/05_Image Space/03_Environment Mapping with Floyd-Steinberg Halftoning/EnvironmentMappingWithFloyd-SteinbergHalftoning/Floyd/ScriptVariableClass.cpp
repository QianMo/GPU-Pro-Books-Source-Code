#include "DXUT.h"
#include "ScriptVariableClass.h"
#include "ScriptVariable.h"
#include "ScriptResourceVariable.h"
#include "ScriptShaderResourceViewVariable.h"
#include "ScriptRenderTargetViewVariable.h"
#include "ScriptDepthStencilViewVariable.h"
#include "ScriptBlobVariable.h"
#include "ScriptCustomVariable.h"
//#include "ScriptCameraVariable.h"

ScriptVariableClass::ScriptVariableClass(unsigned int id, const wchar_t* typeName):
id(id)
{
	names[typeName] = this;
}

bool ScriptVariableClass::operator==(const ScriptVariableClass& o) const
{
	return id == o.id;
}

ScriptVariableClass::ScriptVariableClassInstanceMap ScriptVariableClass::names;

const ScriptVariableClass ScriptVariableClass::Resource(0, L"Resource");
const ScriptVariableClass ScriptVariableClass::ShaderResourceView(1, L"ShaderResourceView");
const ScriptVariableClass ScriptVariableClass::RenderTargetView(2, L"RenderTargetView");
const ScriptVariableClass ScriptVariableClass::DepthStencilView(3, L"DepthStencilView");
const ScriptVariableClass ScriptVariableClass::Blob(4, L"Blob");
const ScriptVariableClass ScriptVariableClass::Custom(5, L"Custom");
const ScriptVariableClass ScriptVariableClass::Camera(6, L"Camera");

const ScriptVariableClass ScriptVariableClass::Unknown(0xffff, L"Unknown");

ScriptVariable* ScriptVariableClass::instantiate() const
{
	if(*this == Resource) return new ::ScriptResourceVariable(NULL);
	if(*this == ShaderResourceView) return new ::ScriptShaderResourceViewVariable(NULL);
	if(*this == RenderTargetView) return new ::ScriptRenderTargetViewVariable(NULL);
	if(*this == DepthStencilView) return new ::ScriptDepthStencilViewVariable(NULL);
	if(*this == Blob) return new ::ScriptBlobVariable(NULL, 0);
//	if(*this == ScriptCameraVariable) return new ::ScriptCameraVariable();
}

const ScriptVariableClass& ScriptVariableClass::fromString(const std::wstring& typeName)
{
	ScriptVariableClassInstanceMap::iterator i = names.find(typeName);
	if(i != names.end())
	{
		return *(i->second);
	}
	else
		EggERR("Unknown type.");
	return Unknown;
}