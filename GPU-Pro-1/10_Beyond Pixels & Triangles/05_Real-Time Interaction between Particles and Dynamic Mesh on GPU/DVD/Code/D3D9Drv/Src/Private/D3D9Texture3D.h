#ifndef D3D9DRV_D3D9TEXTURE3D_H_INCLUDED
#define D3D9DRV_D3D9TEXTURE3D_H_INCLUDED

#include "D3D9Texture.h"

namespace Mod
{
	class D3D9Texture3D : public D3D9Texture
	{
		// construction/ destruction
	public:
		D3D9Texture3D( const Texture3DConfig& cfg, IDirect3DDevice9* dev );
		D3D9Texture3D( const Texture3DConfig& cfg, ResourcePtr res );
		~D3D9Texture3D();

		// manipulation access
	public:
	};
}

#endif