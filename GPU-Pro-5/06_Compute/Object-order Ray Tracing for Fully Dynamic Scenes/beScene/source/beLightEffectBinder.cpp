/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beLightEffectBinder.h"
#include "beScene/beLight.h"
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beBuffer.h>
#include <beGraphics/Any/beTexture.h>
#include <beGraphics/DX/beError.h>
#include <beMath/beVector.h>
#include <lean/io/numeric.h>

namespace beScene
{

/// Pass.
struct LightEffectBinder::Pass
{
	uint4 lightTypeID;
	tristate shadowed;
	bool bAllowMultiple;
};

namespace
{

/// Gets a shader constant variable of the given name or nullptr, if unavailable.
beg::API::EffectConstantBuffer* MaybeGetConstantVariable(beg::API::Effect *pEffect, const lean::utf8_ntri &name)
{
	beGraphics::API::EffectConstantBuffer *pVariable = pEffect->GetConstantBufferByName(name.c_str());
	return (pVariable->IsValid()) ? pVariable : nullptr;
}

/// Gets a shader resource variable of the given name or nullptr, if unavailable.
beg::API::EffectShaderResource* MaybeGetResourceVariable(beg::API::Effect *pEffect, const lean::utf8_ntri &name)
{
	beg::API::EffectShaderResource *pVariable = pEffect->GetVariableBySemantic(name.c_str())->AsShaderResource();
	return (pVariable->IsValid()) ? pVariable : nullptr;
}

/// Gets the given pass from the given technique.
LightEffectBinder::Pass GetPass(ID3DX11EffectTechnique *pTechnique, UINT passID)
{
	LightEffectBinder::Pass pass;

	beGraphics::API::EffectPass *pPass = pTechnique->GetPassByIndex(passID);

	if (!pPass->IsValid())
		LEAN_THROW_ERROR_MSG("ID3DX11Technique::GetPassByIndex()");


	const char *lightTypeName = "";
	pPass->GetAnnotationByName("LightType")->AsString()->GetString(&lightTypeName);
	
	pass.lightTypeID = beCore::Identifiers::InvalidID;
	if (!lean::char_traits<char>::empty(lightTypeName))
		pass.lightTypeID = GetLightTypes().GetID(lightTypeName);


	beGraphics::Any::API::EffectScalar *pShadowed = pPass->GetAnnotationByName("Shadowed")->AsScalar();

	if (pShadowed->IsValid())
	{
		BOOL bShadowed = FALSE;
		pShadowed->GetBool(&bShadowed);
		pass.shadowed = (bShadowed != FALSE) ? caretrue : carefalse;
	}
	else
		pass.shadowed = dontcare;


	BOOL bAllowMultiple = FALSE;
	pPass->GetAnnotationByName("AllowMultiple")->AsScalar()->GetBool(&bAllowMultiple);
	pass.bAllowMultiple = (bAllowMultiple != FALSE);

	return pass;
}

/// Gets all passes in the given technique.
LightEffectBinder::pass_vector GetPasses(ID3DX11EffectTechnique *pTechnique, uint4 singlePassID = static_cast<uint4>(-1))
{
	LightEffectBinder::pass_vector passes;

	D3DX11_TECHNIQUE_DESC techniqueDesc;
	BE_THROW_DX_ERROR_MSG(
		pTechnique->GetDesc(&techniqueDesc),
		"ID3DX11Technique::GetDesc()");
	
	if (singlePassID < techniqueDesc.Passes)
		// Load single pass
		passes.push_back( GetPass(pTechnique, singlePassID) );
	else
	{
		passes.reserve(techniqueDesc.Passes);

		// Load all passes
		for (UINT passID = 0; passID < techniqueDesc.Passes; ++passID)
			passes.push_back( GetPass(pTechnique, passID) );
	}

	return passes;
}

} // namespace

// Constructor.
LightEffectBinder::LightEffectBinder(const beGraphics::Any::Technique &technique, uint4 passID)
	: m_technique( technique ),
	m_pLightData( MaybeGetConstantVariable(m_technique.GetEffect()->Get(), "LightData") ),
	m_pShadowData( MaybeGetConstantVariable(m_technique.GetEffect()->Get(), "ShadowData") ),
	m_pShadowMaps( MaybeGetResourceVariable(m_technique.GetEffect()->Get(), "ShadowMaps") ),
	m_passes( (m_pLightData) ? GetPasses(m_technique, passID) : pass_vector() )
{
	// ASSERT: No light groups => no (lit) passes
	LEAN_ASSERT(m_pLightData || m_passes.empty());
}

// Destructor.
LightEffectBinder::~LightEffectBinder()
{
}

// Applies the n-th step of the given pass.
bool LightEffectBinder::Apply(uint4 &nextPassID, const LightEffectData &light, beGraphics::Any::API::DeviceContext *pContext) const
{
	uint4 passID = nextPassID++;

	// Ignore end of pass range
	if (passID >= m_passes.size())
		return true;

	const Pass &pass = m_passes[passID];

	// Ignore un-lit passes
	if (pass.lightTypeID != light.TypeID)
		return true;

	beg::api::ShaderResourceView *pLightData = ToImpl(light.Lights);
	beg::api::ShaderResourceView *pShadowData = ToImpl(light.Shadows);

	// NOTE: dontcare checked implicitly! (shadowed of type tristate)
	if (pass.shadowed == caretrue && !pShadowData || pass.shadowed == carefalse && pShadowData)
		// Skip pass
		return false;

	LEAN_ASSERT_NOT_NULL(m_pLightData)->SetTextureBuffer(pLightData);
	if (m_pShadowData)
		m_pShadowData->SetTextureBuffer(pShadowData);
	if (m_pShadowMaps)
		m_pShadowMaps->SetResourceArray(const_cast<beg::api::ShaderResourceView**>(&ToImpl(light.ShadowMaps[0]).Get()), 0, light.ShadowMapCount);

	return true;
}

} // namespace