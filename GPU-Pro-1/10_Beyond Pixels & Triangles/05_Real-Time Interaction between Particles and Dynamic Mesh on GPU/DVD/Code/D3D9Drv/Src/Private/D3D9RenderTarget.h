#ifndef D3D9DRV_D3D9RENDERTARGET_H_INCLUDED
#define D3D9DRV_D3D9RENDERTARGET_H_INCLUDED

#include "Wrap3D/Src/RenderTarget.h"

namespace Mod
{

	class D3D9RenderTarget : public RenderTarget
	{
		// types
	public:
		typedef RenderTarget				Base;
		typedef ComPtr<IDirect3DSurface9>	ResourcePtr;
		typedef IDirect3DSurface9*			BindType;

		// construction/ destruction
	public:
		D3D9RenderTarget( const RenderTargetConfig& cfg, IDirect3DDevice9* dev );
		explicit D3D9RenderTarget( IDirect3DSurface9* surf );
		~D3D9RenderTarget();

		// manipulation/ access
	public:
		void Clear( IDirect3DDevice9* dev, const Math::float4& colr );
		void BindTo( IDirect3DDevice9* dev, UINT32 slot ) const;

		static void SetBindToZero( IDirect3DDevice9* dev, UINT32 slot );

		// data
	private:
		ComPtr< IDirect3DSurface9 >	mRTSurface;
	};
	
}

#endif