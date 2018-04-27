#ifndef D3D9DRV_D3D9TEXTURE2D_H_INCLUDED
#define D3D9DRV_D3D9TEXTURE2D_H_INCLUDED

#include "D3D9Texture.h"

namespace Mod
{
	class D3D9Texture2D : public D3D9Texture
	{
		// types
	public:
		typedef ComPtr< IDirect3DSurface9 > SurfacePtr;

		// construction/ destruction
	public:
		D3D9Texture2D( const Texture2DConfig& cfg, IDirect3DDevice9* dev );
		D3D9Texture2D( const Texture2DConfig& cfg, ResourcePtr res );
		~D3D9Texture2D();

		// manipulation access
	public:
		// does stuff in case the texture is in multisampled mode
		void				ResolveTo( IDirect3DDevice9* dev, D3D9Texture2D& tex );
		const SurfacePtr&	GetRenderTargetSurface() const;

		// polymorphism
	private:
		virtual void		DirtyImpl() OVERRIDE;
		virtual void		CopyToImpl( IDirect3DDevice9* dev, D3D9Texture& dest ) OVERRIDE;
		virtual void		UpdateFromRTImpl( IDirect3DDevice9* dev ) OVERRIDE;

		// data
	private:
		SurfacePtr	mRenderTargetSurface;
		SurfacePtr	mResolveSurface;
		bool		mDirty;
	};
}

#endif