#ifndef D3D10DRV_D3D10TEXTURE_H_INCLUDED
#define D3D10DRV_D3D10TEXTURE_H_INCLUDED

#include "Forw.h"

#include "Wrap3D\Src\Texture.h"

namespace Mod
{

	class D3D10Texture : public Texture
	{
		// types
	public:
		typedef ComPtr<ID3D10Resource> ResourcePtr;
		typedef Texture Base;

		// construction/ destruction
	public:
		explicit D3D10Texture( const TextureConfig& cfg );		
		virtual ~D3D10Texture() = 0;
	protected:
		D3D10Texture( const TextureConfig& cfg, ResourcePtr res );

		// manipulation/ access
	public:
		const ResourcePtr& GetResource() const;

		// child manipulation&access
	protected:
		void			SetResource( ResourcePtr::PtrType res );
		ID3D10Device*	GetD3D10Device() const;

		// data
	private:
		ResourcePtr						mResource;
	};

	//------------------------------------------------------------------------	

#define MD_D3D10_CREATE_TEXTURE_FROM_RESOURCE_FUNCTION	\
	TexturePtr CreateTextureFromResource( D3D10Texture::ResourcePtr res, const D3D10FormatMap& fm, const D3D10UsageMap& um )

	MD_D3D10_CREATE_TEXTURE_FROM_RESOURCE_FUNCTION;
}

#endif