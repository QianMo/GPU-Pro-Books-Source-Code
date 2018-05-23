/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/bePerspectiveEffectBinder.h"
#include "beScene/bePerspective.h"
#include "beScene/bePerspectiveEffectBinderPool.h"
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/DX/beError.h>

namespace beScene
{

namespace
{

/// Gets the perspective constant buffer in the given effect.
ID3DX11EffectConstantBuffer* MaybeGetPerspectiveConstants(ID3DX11Effect *pEffect)
{
	ID3DX11EffectConstantBuffer* pBuffer = pEffect->GetConstantBufferByName("PerspectiveConstants");
	return (pBuffer->IsValid()) ? pBuffer : nullptr;
}

/// Gets the flipped version of the given rasterizer state.
lean::com_ptr<ID3D11RasterizerState, true> GetFlippedState(ID3D11RasterizerState *pState)
{
	lean::com_ptr<ID3D11RasterizerState> pFlippedState;

	D3D11_RASTERIZER_DESC flippedDesc;
	pState->GetDesc(&flippedDesc);
	flippedDesc.FrontCounterClockwise = !flippedDesc.FrontCounterClockwise;

	lean::com_ptr<ID3D11Device> pDevice;
	pState->GetDevice(pDevice.rebind());
	BE_THROW_DX_ERROR_MSG(
		pDevice->CreateRasterizerState(&flippedDesc, pFlippedState.rebind()),
		"ID3D11Device::CreateRasterizerState()" );

	return pFlippedState.transfer();
}

/// Gets the original and flipped versions of all rasterizer state variables.
PerspectiveEffectBinder::rasterizer_state_vector GetRasterizerStates(ID3DX11Effect *pEffect, ID3DX11EffectTechnique *pTechnique)
{
	PerspectiveEffectBinder::rasterizer_state_vector rasterizerStates;

	// NOTE: ComputeStateBlockMask does not initialize to 0
	D3DX11_STATE_BLOCK_MASK stateMask = { 0 };
	BE_THROW_DX_ERROR_MSG(
		pTechnique->ComputeStateBlockMask(&stateMask),
		"ID3DX11EffectTechnique::ComputeStateBlockMask()" );

	BOOL dontFlip = false;
	pTechnique->GetAnnotationByName("DontFlip")->AsScalar()->GetBool(&dontFlip);

	if (stateMask.RSRasterizerState && !dontFlip)
	{
		D3DX11_EFFECT_DESC effectDesc;
		BE_THROW_DX_ERROR_MSG(
			pEffect->GetDesc(&effectDesc),
			"ID3DX11Effect::GetDesc()");

		for (UINT variableID = 0; variableID < effectDesc.GlobalVariables; ++variableID)
		{
			ID3DX11EffectRasterizerVariable *pVariable = pEffect->GetVariableByIndex(variableID)->AsRasterizer();

			if (!pVariable->IsValid())
				continue;

			ID3DX11EffectType *pType = pVariable->GetType();

			if (!pType->IsValid())
				LEAN_THROW_ERROR_MSG("Invalid rasterizer variable type");

			D3DX11_EFFECT_TYPE_DESC typeDesc;
			BE_THROW_DX_ERROR_MSG(
				pType->GetDesc(&typeDesc),
				"ID3DX11EffectType::GetDesc()" );

			// Treat everything as array
			if (typeDesc.Elements == 0)
				typeDesc.Elements = 1;

			for (UINT element = 0; element < typeDesc.Elements; ++element)
			{
				lean::com_ptr<ID3D11RasterizerState> pState;
				BE_THROW_DX_ERROR_MSG(
					pVariable->GetRasterizerState(element, pState.rebind()),
					"ID3DX11EffectRasterizerVariable::GetRasterizerState()" );

				rasterizerStates.push_back(
						PerspectiveEffectBinder::Rasterizer(pVariable, element, GetFlippedState(pState).get())
					);
			}
		}
	}

	return rasterizerStates;
}

} // namespace

// Constructor.
PerspectiveEffectBinder::PerspectiveEffectBinder(const beGraphics::Any::Technique &technique, PerspectiveEffectBinderPool *pPool)
	: m_effect( technique.GetEffect() ),
	m_pPool( LEAN_ASSERT_NOT_NULL(pPool) ),
	m_pPerspectiveConstants( MaybeGetPerspectiveConstants(*m_effect) ),
	m_rasterizerStates( GetRasterizerStates(*m_effect, technique) )
{
}

// Destructor.
PerspectiveEffectBinder::~PerspectiveEffectBinder()
{
}

// Applies the given perspective data to the effect bound by this effect driver.
void PerspectiveEffectBinder::Apply(const Perspective &perspective, beGraphics::Any::StateManager& stateManager, ID3D11DeviceContext *pContext) const
{
	if (m_pPerspectiveConstants)
		m_pPerspectiveConstants->SetConstantBuffer(
			m_pPool->GetPerspectiveConstants(&perspective, pContext) );

	const PerspectiveDesc &desc = perspective.GetDesc();

	if (desc.Flipped)
	{
		for (rasterizer_state_vector::const_iterator it = m_rasterizerStates.begin();
			it != m_rasterizerStates.end(); ++it)
			it->pState->SetRasterizerState(it->stateIndex, it->pFlippedState);
	}
	else
	{
		for (rasterizer_state_vector::const_iterator it = m_rasterizerStates.begin();
			it != m_rasterizerStates.end(); ++it)
			it->pState->UndoSetRasterizerState(it->stateIndex);
	}
}

} // namespace