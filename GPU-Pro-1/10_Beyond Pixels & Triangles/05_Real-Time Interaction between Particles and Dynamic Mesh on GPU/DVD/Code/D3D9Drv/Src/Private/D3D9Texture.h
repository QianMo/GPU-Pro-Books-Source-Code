#ifndef D3D9DRV_D3D9TEXTURE_H_INCLUDED
#define D3D9DRV_D3D9TEXTURE_H_INCLUDED

#include "Forw.h"

#include "Wrap3D\Src\Texture.h"

namespace Mod
{

	class D3D9Texture : public Texture
	{
		// types
	public:
		typedef ComPtr<IDirect3DBaseTexture9> ResourcePtr;
		typedef Texture			Base;
		typedef D3D9Texture		Parent;

		// construction/ destruction
	public:
		explicit D3D9Texture( const TextureConfig& cfg );		
		virtual ~D3D9Texture() = 0;
	protected:
		D3D9Texture( const TextureConfig& cfg, ResourcePtr res );

		// manipulation/ access
	public:
		const ResourcePtr&	GetResource() const;
		void				Dirty();
		void				CopyTo( IDirect3DDevice9* dev, D3D9Texture& dest );

		void				BindTo( IDirect3DDevice9* dev, UINT32 slot );
		void				BindTo( ID3DXEffect* eff, D3DXHANDLE handle );

		// child manipulation&access
	protected:
		void				SetResource( ResourcePtr::PtrType res );
		IDirect3DDevice9*	GetD3D9Device() const;

		// polymorphism
	private:
		virtual void		DirtyImpl();
		virtual void		CopyToImpl( IDirect3DDevice9* dev, D3D9Texture& dest );
		virtual void		UpdateFromRTImpl( IDirect3DDevice9* dev );

		// data
	private:
		ResourcePtr						mResource;
	};

	//------------------------------------------------------------------------	

#define MD_D3D9_CREATE_TEXTURE_FROM_RESOURCE_FUNCTION	\
	TexturePtr CreateTextureFromResource( D3D9Texture::ResourcePtr res, D3D9Device& dev )

	MD_D3D9_CREATE_TEXTURE_FROM_RESOURCE_FUNCTION;
}

#endif