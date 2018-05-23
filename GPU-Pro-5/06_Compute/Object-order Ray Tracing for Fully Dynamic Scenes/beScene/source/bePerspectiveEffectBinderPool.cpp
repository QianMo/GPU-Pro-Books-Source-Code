/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/bePerspectiveEffectBinderPool.h"
#include "beScene/bePerspective.h"
#include <beGraphics/Any/beBuffer.h>
#include <beGraphics/DX/beError.h>
#include <memory>

#include <beMath/beMatrix.h>

namespace beScene
{

namespace
{

/// Creates a perspective constant buffer.
lean::com_ptr<ID3D11Buffer, true> CreatePerspectiveConstantBuffer(ID3D11Device *pDevice)
{
	D3D11_BUFFER_DESC desc;
	desc.ByteWidth = sizeof(PerspectiveConstantBuffer);
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = 0;
	desc.StructureByteStride = desc.ByteWidth;

	return beGraphics::Any::CreateBuffer(desc, nullptr, pDevice);
}

/// Creates the given number of perspective constant buffes.
PerspectiveEffectBinderPool::buffer_vector CreatePerspectiveConstantBuffers(ID3D11Device *pDevice, size_t count)
{
	PerspectiveEffectBinderPool::buffer_vector buffers(count);

	for (PerspectiveEffectBinderPool::buffer_vector::iterator it = buffers.begin(); it != buffers.end(); ++it)
		*it = CreatePerspectiveConstantBuffer(pDevice);

	return buffers;
}

/// Updates the given constant buffer with information on the given perspective.
void UpdatePerspectiveConstants(ID3D11Buffer *pBuffer, const Perspective *pPerspective, ID3D11DeviceContext *pContext)
{
	PerspectiveConstantBuffer buffer;

	const PerspectiveDesc &perspective = pPerspective->GetDesc();

	memcpy(buffer.ViewProj, perspective.ViewProjMat.data(), sizeof(float4) * 16);
	memcpy(buffer.ViewProjInv, inverse(perspective.ViewProjMat).data(), sizeof(float4) * 16);

	memcpy(buffer.View, perspective.ViewMat.data(), sizeof(float4) * 16);
	memcpy(buffer.ViewInv, inverse(perspective.ViewMat).data(), sizeof(float4) * 16);

	memcpy(buffer.Proj, perspective.ProjMat.data(), sizeof(float4) * 16);
	memcpy(buffer.ProjInv, inverse(perspective.ProjMat).data(), sizeof(float4) * 16);

	memcpy(buffer.CamRight, perspective.CamRight.data(), sizeof(float4) * 3);
	buffer.CamRight[3] = 0.0f;
	memcpy(buffer.CamUp, perspective.CamUp.data(), sizeof(float4) * 3);
	buffer.CamUp[3] = 0.0f;
	memcpy(buffer.CamDir, perspective.CamLook.data(), sizeof(float4) * 3);
	buffer.CamDir[3] = 0.0f;
	memcpy(buffer.CamPos, perspective.CamPos.data(), sizeof(float4) * 3);
	buffer.CamPos[3] = 1.0f;

	buffer.NearPlane = perspective.NearPlane;
	buffer.FarPlane = perspective.FarPlane;

	buffer.Time = perspective.Time;
	buffer.TimeStep = perspective.TimeStep;

	pContext->UpdateSubresource(pBuffer, 0, nullptr, &buffer, 0, 0);
}

} // namespace

// Constructor.
PerspectiveEffectBinderPool::PerspectiveEffectBinderPool(ID3D11Device *pDevice, uint4 bufferCount)
	: m_pDevice(pDevice),
	m_buffers( CreatePerspectiveConstantBuffers(pDevice, bufferCount) ),
	m_currentIndex(0),
	m_pCurrentBuffer(nullptr),
	m_pCurrentPerspective(nullptr)
{
	LEAN_ASSERT(!m_buffers.empty());
}

// Destructor.
PerspectiveEffectBinderPool::~PerspectiveEffectBinderPool()
{
}

// Gets a constant buffer filled with information on the given perspective.
ID3D11Buffer* PerspectiveEffectBinderPool::GetPerspectiveConstants(const Perspective *pPerspective, ID3D11DeviceContext *pContext)
{
	if (m_pCurrentPerspective != pPerspective)
	{
		if (++m_currentIndex == m_buffers.size())
			m_currentIndex = 0;

		m_pCurrentBuffer = m_buffers[m_currentIndex];
		m_pCurrentPerspective = pPerspective;

		UpdatePerspectiveConstants(m_pCurrentBuffer, m_pCurrentPerspective, pContext);
	}

	return m_pCurrentBuffer;
}

// Invalidates all cached perspective data.
void PerspectiveEffectBinderPool::Invalidate()
{
	m_pCurrentPerspective = nullptr;
}

} // namespace