#ifndef D3D10DRV_D3D10TEXTUREIMPL_H_INCLUDED
#define D3D10DRV_D3D10TEXTUREIMPL_H_INCLUDED

#include "D3D10Texture.h"

namespace Mod
{

	//------------------------------------------------------------------------

	template <typename Config>
	class D3D10TextureImpl : public D3D10Texture
	{
		// types
	public:
		typedef D3D10Texture					Base;
		typedef D3D10TextureImpl<Config>		Parent; // we're this parent
		typedef typename Config::Child			Child;
		typedef typename Config::DescType		DescType;
		typedef typename Config::ResType		ResType;
		typedef typename Config::TexConfigType	TexConfigType;

		typedef Types< D3D10_SUBRESOURCE_DATA > :: Vec D3D10SubresData;

		// construction / destruction
	protected:
		D3D10TextureImpl( const TextureConfig& cfg, ID3D10Device* dev );
		D3D10TextureImpl( const TextureConfig& cfg, ResourcePtr res );

		virtual ~D3D10TextureImpl() = 0;

		const TexConfigType& GetConfig() const;

		static UINT32 GetSubresCount( const TextureConfig& cfg );

	};
}


#endif