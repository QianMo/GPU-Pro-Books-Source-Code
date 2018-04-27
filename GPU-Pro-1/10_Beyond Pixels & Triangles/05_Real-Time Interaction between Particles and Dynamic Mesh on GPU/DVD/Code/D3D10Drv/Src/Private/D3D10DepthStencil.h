#include "Precompiled.h"
#include "Wrap3D/Src/DepthStencil.h"

namespace Mod
{
	class D3D10DepthStencil : public DepthStencil
	{
		// types
	public:
		typedef ComPtr<ID3D10Resource>		ResourcePtr;

		// construction/ destruction
	public:
		D3D10DepthStencil( const DepthStencilConfig& cfg, ID3D10Device* dev );
		~D3D10DepthStencil();

	public:
		void Clear( ID3D10Device* dev, float depthVal, UINT32 stencilVal );
		void BindTo( ID3D10DepthStencilView*& ds ) const;

		// data
	private:
		ComPtr<ID3D10DepthStencilView>	mDSView;
		
	};
}