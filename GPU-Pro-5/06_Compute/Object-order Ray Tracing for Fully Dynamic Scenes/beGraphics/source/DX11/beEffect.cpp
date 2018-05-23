/*****************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beEffect.h"
#include "beGraphics/DX11/beEffectConfig.h"
#include "beGraphics/DX11/beTextureCache.h"

#include "beGraphics/DX/beEffect.h"
#include "beGraphics/DX/beIncludeManager.h"
#include "beGraphics/DX/beError.h"

#include "beGraphics/DX11/beEffectConfig.h"

#include <D3DCompiler.h>

#include <lean/logging/log.h>

namespace beGraphics
{

namespace DX11
{

// Compiles the given effect file's source to object code.
lean::com_ptr<ID3DBlob, true> CompileEffect(const lean::utf8_ntri &fileName,
	const D3D_SHADER_MACRO *pMacros,
	ID3DInclude *pIncludeManager)
{
	LEAN_ASSERT(pIncludeManager != nullptr);

	const char *source;
	UINT sourceLength;

	BE_THROW_DX_ERROR_CTX(
		pIncludeManager->Open(
			D3D_INCLUDE_LOCAL, fileName.c_str(),
			nullptr,
			reinterpret_cast<const void**>(&source), &sourceLength),
		"ID3DInclude::Open()",
		fileName.c_str() );

	struct FileData
	{
		const lean::utf8_ntri &fileName;
		ID3DInclude *pIncludeManager;
		const char *source;

		FileData(const lean::utf8_ntri &fileName, ID3DInclude *pIncludeManager, const char *source)
			: fileName(fileName),
			pIncludeManager(pIncludeManager),
			source(source) { }
		~FileData()
		{
			BE_LOG_DX_ERROR_CTX(
				pIncludeManager->Close(source),
				"ID3DInclude::Close()",
				fileName.c_str() );
		}
	} fileData(fileName, pIncludeManager, source);

	return CompileEffect(source, sourceLength, fileName, pMacros, pIncludeManager);
}

// Compiles the given effect source to object code.
lean::com_ptr<ID3DBlob, true> CompileEffect(const lean::utf8_t *source, uint4 sourceLength,
	const lean::utf8_ntri &debugName,
	const D3D_SHADER_MACRO *pMacros,
	ID3DInclude *pIncludeManager)
{
	lean::com_ptr<ID3DBlob> pBytecode;
	lean::com_ptr<ID3DBlob> pErrors;

	HRESULT result = ::D3DCompile(
		source, sourceLength,
		debugName.c_str(),
		pMacros,
		pIncludeManager,
		nullptr,
		"fx_5_0",
		D3D10_SHADER_PACK_MATRIX_ROW_MAJOR | D3D10_SHADER_PARTIAL_PRECISION | D3D10_SHADER_ENABLE_STRICTNESS | 
#ifdef LEAN_DEBUG_BUILD
		D3D10_SHADER_DEBUG | D3D10_SHADER_OPTIMIZATION_LEVEL2,
#else
		D3D10_SHADER_OPTIMIZATION_LEVEL3,
#endif
		0,
		pBytecode.rebind(),
		pErrors.rebind() );

	if (pErrors != nullptr)
		LEAN_LOG_ERROR(
			"There were errors compiling " << debugName.c_str() << ": " << std::endl
			<< static_cast<utf8_t*>(pErrors->GetBufferPointer()) );

	BE_THROW_DX_ERROR_CTX(result, "D3DCompile()", debugName.c_str());

	return pBytecode.transfer();
}

// Reflects the given shader byte code.
lean::com_ptr<ID3D11ShaderReflection, true> ReflectShader(const char *data, uint4 dataLength)
{
	lean::com_ptr<ID3D11ShaderReflection> pReflection;

	BE_THROW_DX_ERROR_MSG(
		D3DReflect( data, dataLength, IID_ID3D11ShaderReflection, reinterpret_cast<void**>(pReflection.rebind()) ),
		"D3DReflect()");

	return pReflection.transfer();
}

// Gets a shader reflection interface from the given shader variable, if valid, returns nullptr otherwise.
lean::com_ptr<ID3D11ShaderReflection, true> MaybeReflectShader(ID3DX11EffectShaderVariable *pShaderVariable, UINT shaderIndex)
{
	lean::com_ptr<ID3D11ShaderReflection> pReflection;

	if (pShaderVariable && pShaderVariable->IsValid())
	{
		D3DX11_EFFECT_SHADER_DESC shaderDesc;
		BE_THROW_DX_ERROR_MSG(
			pShaderVariable->GetShaderDesc(shaderIndex, &shaderDesc),
			"ID3DX11EffectShaderVariable::GetShaderDesc()");

		// WARNING: Effects Framework translates Set*Shader(NULL) calls to nullptr byte code!
		if (shaderDesc.pBytecode)
			pReflection = ReflectShader( reinterpret_cast<const char*>(shaderDesc.pBytecode), shaderDesc.BytecodeLength );
	}
	
	return pReflection.transfer();
}

// Gets a shader reflection interface from the given pass variable & shader type, if valid, returns nullptr otherwise.
lean::com_ptr<ID3D11ShaderReflection, true> MaybeReflectShader(ID3DX11EffectPass *pPassVariable, ShaderType::T type)
{
	lean::com_ptr<ID3D11ShaderReflection> pReflection;

	typedef HRESULT (__stdcall ID3DX11EffectPass::*get_shader_desc_funptr)(D3DX11_PASS_SHADER_DESC*);
	static get_shader_desc_funptr getShaderDescTable[ShaderType::End] = {
			&ID3DX11EffectPass::GetVertexShaderDesc,
			&ID3DX11EffectPass::GetHullShaderDesc,
			&ID3DX11EffectPass::GetDomainShaderDesc,
			&ID3DX11EffectPass::GetGeometryShaderDesc,
			&ID3DX11EffectPass::GetPixelShaderDesc,
			&ID3DX11EffectPass::GetComputeShaderDesc
		};

	if (pPassVariable && pPassVariable->IsValid() && static_cast<uint4>(type) < ShaderType::End)
	{
		D3DX11_PASS_SHADER_DESC shaderDesc;
		BE_THROW_DX_ERROR_MSG(
			(pPassVariable->*getShaderDescTable[type])(&shaderDesc),
			"ID3DX11EffectPass::Get*ShaderDesc()");

		pReflection = MaybeReflectShader(shaderDesc.pShaderVariable, shaderDesc.ShaderIndex);
	}
	
	return pReflection.transfer();
}

// Gets the description of the given effect variable.
D3DX11_EFFECT_VARIABLE_DESC GetDesc(ID3DX11EffectVariable *variable, const char *name)
{
	D3DX11_EFFECT_VARIABLE_DESC desc;
	BE_THROW_DX_ERROR_CTX(
		variable->GetDesc(&desc),
		"ID3DX11EffectVariable::GetDesc()", name );
	return desc;
}

// Gets the description of the given effect variable.
D3DX11_EFFECT_TYPE_DESC GetDesc(ID3DX11EffectType *type, const char *name)
{
	D3DX11_EFFECT_TYPE_DESC desc;
	BE_THROW_DX_ERROR_CTX(
		type->GetDesc(&desc),
		"ID3DX11EffectType::GetDesc()", name );
	return desc;
}

// Gets the description of the given effect variable.
D3DX11_EFFECT_DESC GetDesc(ID3DX11Effect *effect, const char *name)
{
	D3DX11_EFFECT_DESC desc;
	BE_THROW_DX_ERROR_CTX(
		effect->GetDesc(&desc),
		"ID3DX11Effect::GetDesc()", name );
	return desc;
}

// Gets the description of the given effect variable.
D3DX11_TECHNIQUE_DESC GetDesc(ID3DX11EffectTechnique *technique, const char *name)
{
	D3DX11_TECHNIQUE_DESC desc;
	BE_THROW_DX_ERROR_CTX(
		technique->GetDesc(&desc),
		"ID3DX11EffectTechnique::GetDesc()", name );
	return desc;
}

// Gets the description of the given effect variable.
D3DX11_PASS_DESC GetDesc(ID3DX11EffectPass *pass, const char *name)
{
	D3DX11_PASS_DESC desc;
	BE_THROW_DX_ERROR_CTX(
		pass->GetDesc(&desc),
		"ID3DX11EffectPass::GetDesc()", name );
	return desc;
}

// Validates the given effect variable.
ID3DX11EffectVariable* Validate(ID3DX11EffectVariable *pVariable, const char *src, const char *name)
{
	if (!pVariable || !pVariable->IsValid())
		LEAN_THROW_ERROR_CTX_FROM(src, "ID3DX11EffectVariable::Validate()", name);
	return pVariable;
}
// Validates the given effect variable.
ID3DX11EffectType* Validate(ID3DX11EffectType *pVariable, const char *src, const char *name)
{
	if (!pVariable || !pVariable->IsValid())
		LEAN_THROW_ERROR_CTX_FROM(src, "ID3DX11EffectVariable::Validate()", name);
	return pVariable;
}
// Validates the given effect variable.
ID3DX11EffectTechnique* Validate(ID3DX11EffectTechnique *pVariable, const char *src, const char *name)
{
	if (!pVariable || !pVariable->IsValid())
		LEAN_THROW_ERROR_CTX_FROM(src, "ID3DX11EffectVariable::Validate()", name);
	return pVariable;
}
// Validates the given effect variable.
ID3DX11EffectPass* Validate(ID3DX11EffectPass *pVariable, const char *src, const char *name)
{
	if (!pVariable || !pVariable->IsValid())
		LEAN_THROW_ERROR_CTX_FROM(src, "ID3DX11EffectVariable::Validate()", name);
	return pVariable;
}

// Creates an effect from the given object code.
lean::com_ptr<ID3DX11Effect, true> CreateEffect(const char *data, uint4 dataLength, ID3D11Device *pDevice)
{
	lean::com_ptr<ID3DX11Effect> pEffect;

	BE_THROW_DX_ERROR_MSG(
		::D3DX11CreateEffectFromMemory(data, dataLength, 0, pDevice, pEffect.rebind()),
		"D3DX11CreateEffectFromMemory()");

	return pEffect.transfer();
}

// Constructor.
Effect::Effect(ID3DX11Effect *effect, beGraphics::TextureCache *pTextureCache)
	: m_pEffect( effect )
{
	LEAN_ASSERT(effect != nullptr);

	m_pConfig = new_resource EffectConfig(this, ToImpl(pTextureCache));
}

// Destructor.
Effect::~Effect()
{
}

// Sets the default configuration.
void Effect::SetConfig(EffectConfig *config)
{
	m_pConfig = config;
}

} // namespace

} // namespace
