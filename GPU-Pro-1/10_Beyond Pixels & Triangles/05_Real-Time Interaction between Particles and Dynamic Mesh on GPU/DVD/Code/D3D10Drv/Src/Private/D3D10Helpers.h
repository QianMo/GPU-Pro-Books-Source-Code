#ifndef D3D10DRV_D3D10HELPERS_H_INCLUDED
#define D3D10DRV_D3D10HELPERS_H_INCLUDED

#include "Forw.h"
#include "D3D10Exception.h"


namespace Mod
{
	namespace D3D10
	{
#define MD_DEFINE_CREATE_RES_FUNC(type,desc_type,func)	\
		inline type* CreateResource( ID3D10Device* dev, const desc_type& desc )	{		type* res;		D3D10_THROW_IF( dev->func( &desc, NULL, &res ) ); return res;	};

		MD_DEFINE_CREATE_RES_FUNC(ID3D10Buffer,		D3D10_BUFFER_DESC,		CreateBuffer	)
		MD_DEFINE_CREATE_RES_FUNC(ID3D10Texture1D,	D3D10_TEXTURE1D_DESC,	CreateTexture1D )
		MD_DEFINE_CREATE_RES_FUNC(ID3D10Texture2D,	D3D10_TEXTURE2D_DESC,	CreateTexture2D )
		MD_DEFINE_CREATE_RES_FUNC(ID3D10Texture3D,	D3D10_TEXTURE3D_DESC,	CreateTexture3D )

#undef MD_DEFINE_CREATE_RES_FUNC

		// NOTE : Nothing is copied, just compatible resource is created
		template <typename T>		
		void CreateStagedResource( ComPtr< typename T::ResType >& oRes, const T& res, ID3D10Device* dev )
		{
			const T::ResourcePtr&	resPtr = res.GetResource();
			T::ResType*				rawRes = static_cast<T::ResType*>( &*resPtr );

			T::DescType desc;
			rawRes->GetDesc( &desc );

			desc.CPUAccessFlags = D3D10_CPU_ACCESS_READ;
			desc.Usage			= D3D10_USAGE_STAGING;
			desc.BindFlags		= 0;

			ComPtr< T::ResType > stagingRes;

			{
				T::ResType* res = D3D10::CreateResource( dev, desc );
				MD_FERROR_ON_FALSE( res );
				oRes.set( res );
			}
		}
	}

}

#endif