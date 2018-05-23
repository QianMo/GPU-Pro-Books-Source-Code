/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beLightingEffectBinder.h"
#include "beScene/beLight.h"
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beBuffer.h>
#include <beGraphics/Any/beTexture.h>
#include <beGraphics/DX/beError.h>
#include <beMath/beVector.h>
#include <lean/io/numeric.h>

namespace beScene
{

/// Light group.
struct LightingEffectBinder::LightGroup
{
	beGraphics::Any::API::EffectConstantBuffer *pConstants;
	beGraphics::Any::API::EffectShaderResource *pShadowMaps;
	beGraphics::Any::API::EffectShaderResource *pLightMaps;
};

/// Pass.
struct LightingEffectBinder::Pass
{
	uint4 lightTypeID;
	tristate shadowed;
};

namespace
{

/// Gets a scalar variable of the given name or nullptr, if unavailable.
ID3DX11EffectScalarVariable* MaybeGetScalarVariable(ID3DX11Effect *pEffect, const lean::utf8_ntri &name)
{
	ID3DX11EffectScalarVariable *pVariable = pEffect->GetVariableBySemantic(name.c_str())->AsScalar();
	return (pVariable->IsValid()) ? pVariable : nullptr;
}

/// Gets a shader resource variable of the given name or nullptr, if unavailable.
ID3DX11EffectShaderResourceVariable* MaybeGetResourceVariable(ID3DX11Effect *pEffect, const lean::utf8_ntri &name)
{
	ID3DX11EffectShaderResourceVariable *pVariable = pEffect->GetVariableBySemantic(name.c_str())->AsShaderResource();
	return (pVariable->IsValid()) ? pVariable : nullptr;
}

/// Gets all light groups in the given effect.
LightingEffectBinder::light_group_vector GetLightGroups(ID3DX11Effect *pEffect)
{
	LightingEffectBinder::light_group_vector lightGroups;

	for (uint4 index = 0; ; ++index)
	{
		const utf8_string indexPostfix =  lean::int_to_string(index);

		LightingEffectBinder::LightGroup lightGroup;
		lightGroup.pConstants = pEffect->GetConstantBufferByName( ("Light" + indexPostfix).c_str() );

		// Stop on first missing index
		if (!lightGroup.pConstants->IsValid())
			break;

		lightGroup.pShadowMaps = MaybeGetResourceVariable( pEffect, ("ShadowMaps" + indexPostfix).c_str() );
		lightGroup.pLightMaps = MaybeGetResourceVariable( pEffect, ("LightMaps" + indexPostfix).c_str() );

		lightGroups.push_back(lightGroup);
	}

	return lightGroups;
}

/// Gets the given pass from the given technique.
LightingEffectBinder::Pass GetPass(ID3DX11EffectTechnique *pTechnique, UINT passID)
{
	LightingEffectBinder::Pass pass;

	beGraphics::Any::API::EffectPass *pPass = pTechnique->GetPassByIndex(passID);

	if (!pPass->IsValid())
		LEAN_THROW_ERROR_MSG("ID3DX11Technique::GetPassByIndex()");

	pass.lightTypeID = beCore::Identifiers::InvalidID;
	
	const char *lightTypeName = "";
	pPass->GetAnnotationByName("LightType")->AsString()->GetString(&lightTypeName);
	
	if (!lean::char_traits<char>::empty(lightTypeName))
		pass.lightTypeID = GetLightTypes().GetID(lightTypeName);

	pass.shadowed = dontcare;
	beGraphics::Any::API::EffectScalar *pShadowed = pPass->GetAnnotationByName("Shadowed")->AsScalar();

	if (pShadowed->IsValid())
	{
		BOOL bShadowed = FALSE;
		pShadowed->GetBool(&bShadowed);
		pass.shadowed = (bShadowed != FALSE) ? caretrue : carefalse;
	}

	return pass;
}

/// Gets all passes in the given technique.
LightingEffectBinder::pass_vector GetPasses(ID3DX11EffectTechnique *pTechnique, uint4 singlePassID = static_cast<uint4>(-1))
{
	LightingEffectBinder::pass_vector passes;

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
LightingEffectBinder::LightingEffectBinder(const beGraphics::Any::Technique &technique, uint4 passID)
	: m_technique( technique ),
	m_lightGroups( GetLightGroups(*m_technique.GetEffect()) ),
	m_passes( (!m_lightGroups.empty()) ? GetPasses(m_technique, passID) : pass_vector() ),

	m_pLightCount( MaybeGetScalarVariable(*m_technique.GetEffect(), "LightCount") )
{
	// ASSERT: No light groups => no (lit) passes
	LEAN_ASSERT(!m_lightGroups.empty() || m_passes.empty());
}

// Destructor.
LightingEffectBinder::~LightingEffectBinder()
{
}

// Applies the n-th step of the given pass.
bool LightingEffectBinder::Apply(uint4 &nextPassID, const LightJob *lights, const LightJob *lightsEnd,
		LightingBinderState &lightState, beGraphics::Any::API::DeviceContext *pContext) const
{
	uint4 passID = nextPassID++;

	// Ignore end of pass range
	if (passID >= m_passes.size())
		return true;

	const Pass &pass = m_passes[passID];

	// Ignore un-lit passes
	if (pass.lightTypeID == LightTypes::InvalidID)
		return true;

	// Mask out all lights failing shadowing requirements
	uint4 lightFlagMask = (pass.shadowed != dontcare) ? LightFlags::Shadowed : 0;
	uint4 lightFlagResult = (pass.shadowed == caretrue) ? LightFlags::Shadowed : 0;

	const size_t maxLightCount = m_lightGroups.size();
	uint4 lightCount = 0;

	// ASSERT: No light groups => no (lit) passes (see constructor)
	LEAN_ASSERT(maxLightCount > 0);

	for (const LightJob *it = lights + lightState.LightOffset; it < lightsEnd; ++it)
	{
		const LightJob &lightJob = *it;

		if (lightJob.Light->GetLightTypeID() == pass.lightTypeID &&
			(lightJob.Light->GetLightFlags() & lightFlagMask) == lightFlagResult)
		{
			if (lightCount < maxLightCount)
			{
				const LightGroup &lightGroup = m_lightGroups[lightCount];

				lightGroup.pConstants->SetConstantBuffer( ToImpl(lightJob.Light->GetConstants()) );
				
				if (lightGroup.pShadowMaps && pass.shadowed != carefalse)
				{
					uint4 shadowMapCount = 0;
					const beGraphics::TextureViewHandle *pShadowMaps = lightJob.Light->GetShadowMaps(lightJob.Data, shadowMapCount);

					ID3D11ShaderResourceView* shadowMaps[16];

					// Clamp to array size
					shadowMapCount = min<uint4>(shadowMapCount, static_cast<uint4>(lean::arraylen(shadowMaps)));

					for (uint4 i = 0; i < shadowMapCount; ++i)
						shadowMaps[i] = ToImpl(pShadowMaps[i]);
					
					lightGroup.pShadowMaps->SetResourceArray(shadowMaps, 0, shadowMapCount);
				}

				++lightCount;
			}
			else
			{
				// More lights of the current type, repeat this pass.
				--nextPassID;
				lightState.LightOffset = static_cast<uint4>(it - lights);
				break;
			}
		}
	}

	// Reset light offset for next pass
	if (nextPassID != passID)
		lightState.LightOffset = 0;

	// Update light count
	if (m_pLightCount)
		m_pLightCount->SetInt(lightCount);

	// Allow for skipping of zero-light-passes
	return (lightCount != 0);
}

} // namespace