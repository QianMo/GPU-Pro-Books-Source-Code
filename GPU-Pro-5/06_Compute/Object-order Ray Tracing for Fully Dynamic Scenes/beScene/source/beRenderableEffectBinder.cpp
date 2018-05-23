/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beRenderableEffectBinder.h"
#include "beScene/beRenderable.h"
#include "beScene/bePerspective.h"
#include <memory>

#include <beMath/beVector.h>
#include <beMath/beMatrix.h>

namespace beScene
{

namespace
{

/// Gets a scalar variable of the given name or nullptr, if unavailable.
ID3DX11EffectScalarVariable* MaybeGetScalarVariable(ID3DX11Effect *pEffect, const lean::utf8_ntri &name)
{
	ID3DX11EffectScalarVariable *pVariable = pEffect->GetVariableBySemantic(name.c_str())->AsScalar();
	return (pVariable->IsValid()) ? pVariable : nullptr;
}

/// Gets a vector variable of the given name or nullptr, if unavailable.
ID3DX11EffectVectorVariable* MaybeGetVectorVariable(ID3DX11Effect *pEffect, const lean::utf8_ntri &name)
{
	ID3DX11EffectVectorVariable *pVariable = pEffect->GetVariableBySemantic(name.c_str())->AsVector();
	return (pVariable->IsValid()) ? pVariable : nullptr;
}

/// Gets a matrix variable of the given name or nullptr, if unavailable.
ID3DX11EffectMatrixVariable* MaybeGetMatrixVariable(ID3DX11Effect *pEffect, const lean::utf8_ntri &name)
{
	ID3DX11EffectMatrixVariable *pVariable = pEffect->GetVariableBySemantic(name.c_str())->AsMatrix();
	return (pVariable->IsValid()) ? pVariable : nullptr;
}

} // namespace

// Constructor.
RenderableEffectBinder::RenderableEffectBinder(const beGraphics::Any::Technique &technique, PerspectiveEffectBinderPool *pPool)
	: m_perspectiveBinder(technique, pPool),

	m_pWorld( MaybeGetMatrixVariable(*technique.GetEffect(), "World") ),
	m_pWorldInverse( MaybeGetMatrixVariable(*technique.GetEffect(), "WorldInverse") ),
	
	m_pWorldViewProj( MaybeGetMatrixVariable(*technique.GetEffect(), "WorldViewProj") ),
	m_pWorldView( MaybeGetMatrixVariable(*technique.GetEffect(), "WorldView") ),
	
	m_pObjectCamPos( MaybeGetVectorVariable(*technique.GetEffect(), "ObjectCamPos") ),
	m_pObjectCamDir( MaybeGetVectorVariable(*technique.GetEffect(), "ObjectCamDir") ),

	m_pID( MaybeGetScalarVariable(*technique.GetEffect(), "ID") )
{
}

// Destructor.
RenderableEffectBinder::~RenderableEffectBinder()
{
}

// Applies the given renderable & perspective data to the effect bound by this effect driver.
void RenderableEffectBinder::Apply(const RenderableEffectData *pRenderableData, const Perspective &perspective,
		beGraphics::Any::StateManager& stateManager, ID3D11DeviceContext *pContext) const
{
	// Make renderable optional
	if (pRenderableData)
	{
		if (m_pWorld)
			m_pWorld->SetRawValue( pRenderableData->Transform.data(), 0, sizeof(float4) * 16 );
		if (m_pWorldInverse)
			m_pWorldInverse->SetRawValue( pRenderableData->TransformInv.data(), 0, sizeof(float4) * 16 );

		const PerspectiveDesc &perspectiveDesc = perspective.GetDesc();

		if (m_pWorldViewProj)
			m_pWorldViewProj->SetRawValue( mul(pRenderableData->Transform, perspectiveDesc.ViewProjMat).data(), 0, sizeof(float4) * 16 );
		if (m_pWorldView)
			m_pWorldView->SetRawValue( mul(pRenderableData->Transform, perspectiveDesc.ViewMat).data(), 0, sizeof(float4) * 16 );

		if (m_pObjectCamPos)
			m_pObjectCamPos->SetRawValue( mulh(perspectiveDesc.CamPos, pRenderableData->TransformInv).data(), 0, sizeof(float4) * 3 );
		if (m_pObjectCamDir)
			m_pObjectCamDir->SetRawValue( mulh(perspectiveDesc.CamLook, pRenderableData->TransformInv).data(), 0, sizeof(float4) * 3 );

		if (m_pID)
			m_pID->SetRawValue( &pRenderableData->ID, 0, sizeof(uint4) );
	}

	return m_perspectiveBinder.Apply(perspective, stateManager, pContext);
}

} // namespace