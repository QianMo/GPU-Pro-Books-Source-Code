#ifndef D3D9DRV_D3D9TEXTURE1D_H_INCLUDED
#define D3D9DRV_D3D9TEXTURE1D_H_INCLUDED

#include "D3D9Texture.h"

namespace Mod
{
	class D3D9Texture1D : public D3D9Texture
	{
		// construction/ destruction
	public:
		D3D9Texture1D( const Texture1DConfig& cfg, IDirect3DDevice9* dev );
		D3D9Texture1D( const Texture1DConfig& cfg, ResourcePtr res );
		~D3D9Texture1D();

		// manipulation access
	public:

		// polymorphism
	private:
		virtual void CopyToImpl( IDirect3DDevice9* dev, D3D9Texture& dest ) OVERRIDE;
	};
}

#endif