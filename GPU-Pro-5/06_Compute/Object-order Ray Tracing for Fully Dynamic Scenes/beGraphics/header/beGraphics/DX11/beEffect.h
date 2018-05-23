/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_EFFECT_DX11
#define BE_GRAPHICS_EFFECT_DX11

#include "beGraphics.h"
#include "../beEffect.h"
#include <beCore/beWrapper.h>
#include <D3DX11Effect.h>
#include <lean/smart/com_ptr.h>
#include <lean/smart/resource_ptr.h>

namespace beGraphics
{

namespace DX11
{

class Effect;
class EffectConfig;

// Compiles the given effect file's source to object code.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3DBlob, true> CompileEffect(const lean::utf8_ntri &fileName,
	const D3D_SHADER_MACRO *pMacros,
	ID3DInclude *pIncludeManager);
/// Compiles the given effect source to object code.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3DBlob, true> CompileEffect(const lean::utf8_t *source, uint4 sourceLength,
	const lean::utf8_ntri &debugName,
	const D3D_SHADER_MACRO *pMacros,
	ID3DInclude *pIncludeManager);

/// Creates an effect from the given object code.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3DX11Effect, true> CreateEffect(const char *data, uint4 dataLength, ID3D11Device *pDevice);
/// Creates an effect from the given object code.
LEAN_INLINE lean::com_ptr<ID3DX11Effect, true> CreateEffect(ID3DBlob *pBlob, ID3D11Device *pDevice)
{
	return CreateEffect(static_cast<const char*>(pBlob->GetBufferPointer()), static_cast<uint4>(pBlob->GetBufferSize()), pDevice);
}

/// Shader type enumeration.
struct ShaderType
{
	/// Enumeration.
	enum T
	{
		Begin = 0,				///< First valid shader type.

		VertexShader = Begin,	///< Vertex shader.
		HullShader,				///< Hull shader.
		DomainShader,			///< Domain shader.
		GeometryShader,			///< Geometry shader.
		PixelShader,			///< Pixel shader.
		ComputeShader,			///< Compute shader.

		End						/// One past the last valid shader type.
	};
	LEAN_MAKE_ENUM_STRUCT(ShaderType)
};

/// Reflects the given shader byte code.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11ShaderReflection, true> ReflectShader(const char *data, uint4 dataLength);
/// Gets a shader reflection interface from the given shader variable, if valid, returns nullptr otherwise.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11ShaderReflection, true> MaybeReflectShader(ID3DX11EffectShaderVariable *pShaderVariable, UINT shaderIndex = 0);
/// Gets a shader reflection interface from the given pass variable & shader type, if valid, returns nullptr otherwise.
BE_GRAPHICS_DX11_API lean::com_ptr<ID3D11ShaderReflection, true> MaybeReflectShader(ID3DX11EffectPass *pShaderVariable, ShaderType::T type);

/// Gets the description of the given effect variable.
BE_GRAPHICS_DX11_API D3DX11_EFFECT_VARIABLE_DESC GetDesc(ID3DX11EffectVariable *variable, const char *name = nullptr);
/// Gets the description of the given effect variable.
BE_GRAPHICS_DX11_API D3DX11_EFFECT_TYPE_DESC GetDesc(ID3DX11EffectType *type, const char *name = nullptr);
/// Gets the description of the given effect variable.
BE_GRAPHICS_DX11_API D3DX11_EFFECT_DESC GetDesc(ID3DX11Effect *effect, const char *name = nullptr);
/// Gets the description of the given effect variable.
BE_GRAPHICS_DX11_API D3DX11_TECHNIQUE_DESC GetDesc(ID3DX11EffectTechnique *technique, const char *name = nullptr);
/// Gets the description of the given effect variable.
BE_GRAPHICS_DX11_API D3DX11_PASS_DESC GetDesc(ID3DX11EffectPass *pass, const char *name = nullptr);

/// Validates the given effect variable.
BE_GRAPHICS_DX11_API ID3DX11EffectVariable* Validate(ID3DX11EffectVariable *pVariable, const char *src = nullptr, const char *name = nullptr);
/// Validates the given effect variable.
BE_GRAPHICS_DX11_API ID3DX11EffectType* Validate(ID3DX11EffectType *pVariable, const char *src = nullptr, const char *name = nullptr);
/// Validates the given effect variable.
BE_GRAPHICS_DX11_API ID3DX11EffectTechnique* Validate(ID3DX11EffectTechnique *pVariable, const char *src = nullptr, const char *name = nullptr);
/// Validates the given effect variable.
BE_GRAPHICS_DX11_API ID3DX11EffectPass* Validate(ID3DX11EffectPass *pVariable, const char *src = nullptr, const char *name = nullptr);

/// Validates the given effect variable.
template <class Variable>
LEAN_INLINE Variable* ValidateEffectVariable(Variable *pVariable, const char *src = nullptr, const char *name = nullptr)
{
	Validate(pVariable, src, name);
	return pVariable;
}

/// Effect implementation.
class Effect : public beCore::IntransitiveWrapper<ID3DX11Effect, Effect>, public beGraphics::Effect
{
private:
	lean::com_ptr<ID3DX11Effect> m_pEffect;

	lean::resource_ptr<EffectConfig> m_pConfig;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API Effect(ID3DX11Effect *effect, beGraphics::TextureCache *pTextureCache);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~Effect();

	/// Gets the D3DX effect.
	LEAN_INLINE ID3DX11Effect*const& GetInterface() const { return m_pEffect.get(); }
	/// Gets the D3DX effect.
	LEAN_INLINE ID3DX11Effect*const& GetEffect() const { return m_pEffect.get(); }

	/// Sets the default configuration.
	BE_GRAPHICS_DX11_API void SetConfig(EffectConfig *config);
	/// Gets the default configuration.
	LEAN_INLINE EffectConfig* GetConfig() const { return m_pConfig; }

	/// Gets the implementation identifier.
	LEAN_INLINE ImplementationID GetImplementationID() const { return DX11Implementation; }
};

template <> struct ToImplementationDX11<beGraphics::Effect> { typedef Effect Type; };

/// Technique implementation.
class Technique : public beCore::IntransitiveWrapper<ID3DX11EffectTechnique, Technique>, public beGraphics::Technique
{
private:
	lean::resource_ptr<const Effect> m_pEffect;
	ID3DX11EffectTechnique *m_pTechnique;

public:
	/// Constructor.
	LEAN_INLINE Technique(const Effect *pEffect, ID3DX11EffectTechnique *pTechnique)
		: m_pEffect( LEAN_ASSERT_NOT_NULL(pEffect) ),
		m_pTechnique( LEAN_ASSERT_NOT_NULL(pTechnique) ) { }

	/// Gets the D3DX technique.
	LEAN_INLINE ID3DX11EffectTechnique*const& GetInterface() const { return m_pTechnique; }
	/// Gets the D3DX technique.
	LEAN_INLINE ID3DX11EffectTechnique*const& GetTechnique() const { return m_pTechnique; }

	/// Gets the effect.
	LEAN_INLINE const Effect* GetEffect() const { return m_pEffect; }

	/// Gets the implementation identifier.
	LEAN_INLINE ImplementationID GetImplementationID() const { return DX11Implementation; }
};

template <> struct ToImplementationDX11<beGraphics::Technique> { typedef Technique Type; };

} // namespace

using DX11::CompileEffect;
using DX11::CreateEffect;
using DX11::GetDesc;

} // namespace

#endif