

#ifndef __DX11_RENDERER_STATES_HPP__
#define __DX11_RENDERER_STATES_HPP__

#include <d3dx11.h>
#include <d3d11.h>

#include <Common/Assert.hpp>
#include <Common/Common.hpp>
#include <Math/Vector/Vector.hpp>
#include <Common/Incopiable.hpp>


///< Rasterizer states and others.
namespace List
{
	///<
	class States : public Incopiable
	{
		friend class Dx11Renderer;

		static States m_inst;

		void Create		(ID3D11Device* _pDevice);
		void Release	();

	public:

		States(){ memset(this,0,sizeof(States)); }
		~States(){ ASSERT(m_pCullBackRasterizer==NULL, "Pointers not released!"); }

		///< Acces singleton!
		static States& Get(){ return m_inst; }

		void SetSamplers(ID3D11DeviceContext* _pImmediateContext);

		///< List of all states.
		ID3D11RasterizerState*		m_pCullBackRasterizer;
		ID3D11RasterizerState*		m_pCullFrontRasterizer;

		ID3D11RasterizerState*		m_pNoCullBackRasterizer;
		ID3D11RasterizerState*		m_pBlendingRasterizer;

		ID3D11RasterizerState*		m_pNoCullBackWireframeRasterizer;

		///< Samplers
		ID3D11SamplerState*			m_pLinearSampler; 
		ID3D11SamplerState*			m_pPointSampler;

		///< Depths
		ID3D11DepthStencilState*	m_pDepthLess;
		ID3D11DepthStencilState*	m_pNoDepth;

		///< Blends
		ID3D11BlendState*			m_pDefaultNoBlendState;

		ID3D11BlendState*			m_pAddBlendState;

		ID3D11BlendState*			m_pTransparentBlendState;
		

	};
}

#endif