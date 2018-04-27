#ifndef D3D9DRV_D3D9TEXTURECOORDINATOR_H_INCLUDED
#define D3D9DRV_D3D9TEXTURECOORDINATOR_H_INCLUDED

#include "Forw.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE D3D9TextureCoordinatorNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class D3D9TextureCoordinator : public D3D9TextureCoordinatorNS::ConfigurableImpl<D3D9TextureCoordinatorConfig>
	{
		// types
	public:
		struct Texture
		{
			ComPtr<IDirect3DBaseTexture9> tex;
			bool dirty;

			Texture();
		};

		typedef Types< Texture > :: Vec Textures;

		// constructors / destructors
	public:
		explicit D3D9TextureCoordinator( const D3D9TextureCoordinatorConfig& cfg );
		~D3D9TextureCoordinator();
	
		// manipulation/ access
	public:
		void SetTexture( UINT32 slot, IDirect3DBaseTexture9* texture );
		void Update();
		void Sync();

		// data
	private:
		Textures mTextures;
	};
}

#endif