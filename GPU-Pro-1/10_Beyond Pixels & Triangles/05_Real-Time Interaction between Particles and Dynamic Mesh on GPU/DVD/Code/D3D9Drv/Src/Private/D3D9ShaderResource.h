#ifndef D3D9DRV_D3D9SHADERRESOURCE_H_INCLUDED
#define D3D9DRV_D3D9SHADERRESOURCE_H_INCLUDED

#include "Wrap3D/Src/ShaderResource.h"

namespace Mod
{
	class D3D9ShaderResource : public ShaderResource
	{
		// types
	public:
		typedef ComPtr<IDirect3DBaseTexture9>		ResourcePtr;

		// construction/ destruction
	public:
		D3D9ShaderResource( const ShaderResourceConfig& cfg );
		~D3D9ShaderResource();

		// manipulation/ access
	public:
		void			BindTo( ID3DXEffect* eff, D3DXHANDLE handle ) const;
		void			BindTo( IDirect3DDevice9* dev, UINT32 slot );

		static void		SetBindToZero( ID3DXEffect* eff, D3DXHANDLE handle );
		static void		SetBindToZero( IDirect3DDevice9* dev, UINT32 slot );

		// data
	private:
		ResourcePtr	mSR;
	};
}

#endif