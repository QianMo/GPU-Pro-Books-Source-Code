#include "Precompiled.h"
#include "Wrap3D/Src/DepthStencil.h"

namespace Mod
{
	class D3D9DepthStencil : public DepthStencil
	{
		// types
	public:
		typedef ComPtr<IDirect3DSurface9>		ResourcePtr;


		// construction/ destruction
	public:
		D3D9DepthStencil( const DepthStencilConfig& cfg, IDirect3DDevice9* dev );
		explicit D3D9DepthStencil( IDirect3DSurface9* surf );
		~D3D9DepthStencil();

	public:
		void Clear( IDirect3DDevice9* dev, float depthVal, UINT32 stencilVal );
		void BindTo( IDirect3DDevice9* dev ) const;

		static void SetBindToZero( IDirect3DDevice9* dev );

		// data
	private:
		ComPtr<IDirect3DSurface9>	mDSResource;
		
	};
}