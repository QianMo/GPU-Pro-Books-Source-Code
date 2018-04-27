#ifndef D3D9DRV_D3D9TEXTURECUBE_H_INCLUDED
#define D3D9DRV_D3D9TEXTURECUBE_H_INCLUDED

#include "D3D9Texture.h"

namespace Mod
{
	class D3D9TextureCube : public D3D9Texture
	{
		// construction/ destruction
	public:
		D3D9TextureCube( const TextureCUBEConfig& cfg, IDirect3DDevice9* dev );
		D3D9TextureCube( const TextureCUBEConfig& cfg, ResourcePtr res );
		~D3D9TextureCube();

		// manipulation access
	public:
	};
}

#endif