#include "Precompiled.h"

#include "Wrap3D/Src/ShaderResourceConfig.h"

#include "D3D9Texture.h"

#include "D3D9ShaderResource.h"

namespace Mod
{
	D3D9ShaderResource::D3D9ShaderResource( const ShaderResourceConfig& cfg ) :
	Parent( cfg )
	{

	}

	//------------------------------------------------------------------------

	D3D9ShaderResource::~D3D9ShaderResource()
	{

	}

	//------------------------------------------------------------------------

	void
	D3D9ShaderResource::BindTo( ID3DXEffect* eff, D3DXHANDLE handle ) const
	{
		static_cast<D3D9Texture&>(*GetConfig().tex).BindTo( eff, handle );
	}

	//------------------------------------------------------------------------

	void
	D3D9ShaderResource::BindTo( IDirect3DDevice9* dev, UINT32 slot )
	{
		static_cast<D3D9Texture&>(*GetConfig().tex).BindTo( dev, slot );
	}

	//------------------------------------------------------------------------
	/*static*/
	void
	D3D9ShaderResource::SetBindToZero( ID3DXEffect* eff, D3DXHANDLE handle )
	{
		MD_D3DV( eff->SetTexture( handle, NULL ) );
	}

	//------------------------------------------------------------------------
	/*static*/
	
	void
	D3D9ShaderResource::SetBindToZero( IDirect3DDevice9* dev, UINT32 slot )
	{
		MD_D3DV( dev->SetTexture( slot, NULL ) );
	}

}