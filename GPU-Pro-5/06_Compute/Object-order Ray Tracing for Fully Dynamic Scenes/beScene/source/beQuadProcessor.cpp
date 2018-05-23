/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beQuadProcessor.h"
#include "beScene/DX11/beMesh.h"
#include "beScene/beMeshGeneration.h"
#include "beScene/beEffectBinderCache.h"
#include "beScene/beAbstractProcessingEffectDriver.h"
#include "beScene/beRenderContext.h"
#include "beScene/beRenderingLimits.h"
#include <beGraphics/Any/beDevice.h>
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beStateManager.h>
#include <beGraphics/DX/beError.h>

namespace beScene
{

/// Effect binding.
struct QuadProcessor::Layer
{
	lean::com_ptr<ID3D11InputLayout> pInputLayout;
	lean::resource_ptr<const AbstractProcessingEffectDriver> pEffectDriver;
	const beg::MaterialTechnique *technique;

	/// Constructor.
	Layer(ID3D11InputLayout *pInputLayout,
		const AbstractProcessingEffectDriver *pEffectDriver,
		const beg::MaterialTechnique *technique)
			: pInputLayout(pInputLayout),
			pEffectDriver(pEffectDriver),
			technique(technique) { }
};

// Constructor.
QuadProcessor::QuadProcessor(const beGraphics::Device *pDevice, EffectDriverCache<AbstractProcessingEffectDriver> *pEffectBinderCache)
	: MaterialDriven(nullptr),
	m_pEffectBinderCache( LEAN_ASSERT_NOT_NULL(pEffectBinderCache) ),
	m_pDevice( LEAN_ASSERT_NOT_NULL(pDevice) ),
	m_pQuad(
		GenerateGridMesh(
			beMath::vec(-1.0f, -1.0f, 0.0f),
			beMath::vec(4.0f, 0.0f, 0.0f),
			beMath::vec(0.0f, 4.0f, 0.0f),
			0.0f, 0.0f, 0.0f, 0.0f,
			1, 1,
			0, *pDevice) )
{
}

// Destructor.
QuadProcessor::~QuadProcessor()
{
}

// Applies this processor.
void QuadProcessor::Render(uint4 layerIdx, uint4 stageID, uint4 queueID, const Perspective *pPerspective, const RenderContext &context, bool &bProcessorReady) const
{
	ID3D11DeviceContext *pContextDX = ToImpl(context.Context());
	const DX11::Mesh &mesh = beGraphics::ToImpl(*m_pQuad);

	const Layer &layer = m_layers[layerIdx];

	AbstractProcessingEffectDriver::PassRange passes = layer.pEffectDriver->GetPasses();
	bool bLayerReady = false;

	for (uint4 passID = 0; passID < Size(passes); ++passID)
	{
		const QueuedPass &pass = passes.Begin[passID];

		if (pass.GetStageID() == stageID && pass.GetQueueID() == queueID)
		{
			if (!bProcessorReady)
			{
				ToImpl(context.StateManager()).Revert();
				ToImpl(context.StateManager()).Reset();

				pContextDX->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

				UINT vertexStride = mesh.GetVertexSize();
				UINT vertexOffset = 0;
				pContextDX->IASetVertexBuffers(0, 1, &mesh.GetVertexBuffer().GetBuffer(), &vertexStride, &vertexOffset);

				pContextDX->IASetIndexBuffer(mesh.GetIndexBuffer(), mesh.GetIndexFormat(), 0);

				bProcessorReady = true;
			}

			if (!bLayerReady)
			{
				pContextDX->IASetInputLayout(layer.pInputLayout);

				layer.technique->Apply(context.Context());
				// NOTE: Obsolete
//				layer.pEffectDriver->Apply(pPerspective, context.StateManager(), context.Context());

				bLayerReady = true;
			}

			struct DrawJob : lean::vcallable_base<AbstractProcessingEffectDriver::DrawJobSignature, DrawJob>
			{
				uint4 indexCount;

				void operator ()(uint4 passIdx, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context)
				{
					ToImpl(stateManager).Reset();
					ToImpl(context)->DrawIndexed(indexCount, 0, 0);
				}
			};

			DrawJob drawJob;
			drawJob.indexCount = mesh.GetIndexCount();

			layer.pEffectDriver->Render(&pass, this, pPerspective, drawJob, context.StateManager(), context.Context());
		}
	}
}

// Applies this processor (unclassified passes only).
void QuadProcessor::Render(const Perspective *pPerspective, const RenderContext &context) const
{
	QuadProcessor::Render(InvalidPipelineStage, InvalidRenderQueue, pPerspective, context);
}

/// Applies this processor (classified passes only).
void QuadProcessor::Render(uint4 stageID, uint4 queueID, const Perspective *pPerspective, const RenderContext &context) const
{
	const uint4 layerCount = static_cast<uint4>( m_layers.size() );
	bool bProcessorReady = false;

	for (uint4 i = 0; i < layerCount; ++i)
		Render(i, stageID, queueID, pPerspective, context, bProcessorReady);
}

// Applies this processor (unclassified passes only).
void QuadProcessor::Render(uint4 layerIdx, const Perspective *pPerspective, const RenderContext &context) const
{
	if (layerIdx < m_layers.size())
	{
		bool bProcessorReady = false;
		Render(layerIdx, InvalidPipelineStage, InvalidRenderQueue, pPerspective, context, bProcessorReady);
	}
}

// Applies this processor (classified passes only).
void QuadProcessor::Render(uint4 layerIdx, uint4 stageID, uint4 queueID, const Perspective *pPerspective, const RenderContext &context) const
{
	if (layerIdx < m_layers.size())
	{
		bool bProcessorReady = false;
		Render(layerIdx, stageID, queueID, pPerspective, context, bProcessorReady);
	}
}

// Sets the processing effect.
void QuadProcessor::SetMaterial(beg::Material *pMaterial)
{
	if (pMaterial != m_pMaterial)
	{
		m_pMaterial = nullptr;
		m_layers.clear();

		if (pMaterial)
		{
			ID3D11Device *pDevice = ToImpl(*m_pDevice);
			const DX11::Mesh &mesh = beGraphics::Any::ToImpl(*m_pQuad);

			const uint4 layerCount = pMaterial->GetTechniqueCount();

			for (uint4 layerID = 0; layerID < layerCount; ++layerID)
			{
				const beg::MaterialTechnique *technique = pMaterial->GetTechnique(layerID);
				const AbstractProcessingEffectDriver *pEffectBinder = m_pEffectBinderCache->GetEffectBinder(*technique->Technique);

				if (Size(pEffectBinder->GetPasses()))
				{
					uint4 passSignatureSize = 0;
					const char *pPassSignature = pEffectBinder->GetPasses()[0].GetInputSignature(passSignatureSize);

					// Ignore compute-only techniques
					if (pPassSignature)
					{
						lean::com_ptr<ID3D11InputLayout> pInputLayout;
						BE_THROW_DX_ERROR_MSG(
							pDevice->CreateInputLayout(
								mesh.GetVertexElementDescs(),
								mesh.GetVertexElementDescCount(),
								pPassSignature, passSignatureSize,
								pInputLayout.rebind()),
							"ID3D11Device::CreateInputLayout()");

						m_layers.push_back( Layer(pInputLayout, pEffectBinder, technique) );
					}
				}
			}
		}

		m_pMaterial = pMaterial;
	}
}

} // namespace
