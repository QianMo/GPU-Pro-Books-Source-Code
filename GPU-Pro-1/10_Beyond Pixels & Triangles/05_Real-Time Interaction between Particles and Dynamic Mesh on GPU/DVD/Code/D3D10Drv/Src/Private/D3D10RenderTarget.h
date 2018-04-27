#ifndef D3D10DRV_D3D10RENDERTARGET_H_INCLUDED
#define D3D10DRV_D3D10RENDERTARGET_H_INCLUDED

#include "Wrap3D/Src/RenderTarget.h"

namespace Mod
{

	class D3D10RenderTarget : public RenderTarget
	{
		// types
	public:
		typedef ComPtr<ID3D10Resource>	ResourcePtr;
		typedef ID3D10RenderTargetView* BindType;

		// construction/ destruction
	public:
		D3D10RenderTarget( const RenderTargetConfig& cfg, ID3D10Device* dev );
		~D3D10RenderTarget();

		// manipulation/ access
	public:
		void Clear( ID3D10Device* dev, const Math::float4& colr );
		void BindTo( BindType& target ) const;

		static void SetBindToZero( BindType& target );

		// data
	private:
		ComPtr< ID3D10RenderTargetView >	mRTView;	
	};
	
}

#endif