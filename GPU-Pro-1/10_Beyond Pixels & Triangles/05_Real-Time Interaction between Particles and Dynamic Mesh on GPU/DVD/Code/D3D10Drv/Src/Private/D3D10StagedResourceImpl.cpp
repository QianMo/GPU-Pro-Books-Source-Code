#include "Precompiled.h"

#include "Wrap3D/Src/StagedResourceConfig.h"
#include "Wrap3D/Src/TextureConfig.h"

#include "D3D10Buffer.h"
#include "D3D10Texture1D.h"
#include "D3D10Texture2D.h"
#include "D3D10Texture3D.h"

#include "D3D10StagedResourceImpl.h"

#include "D3D10Helpers.h"

namespace Mod
{

	namespace
	{
		template <typename R>
		struct ResourceToWrapper;

		template <>
		struct ResourceToWrapper< ID3D10Buffer >
		{
			typedef D3D10Buffer Result;
		};

		template <>
		struct ResourceToWrapper< ID3D10Texture1D >
		{
			typedef D3D10Texture1D Result;
		};

		template <>
		struct ResourceToWrapper< ID3D10Texture2D >
		{
			typedef D3D10Texture2D Result;
		};

		template <>
		struct ResourceToWrapper< ID3D10Texture3D >
		{
			typedef D3D10Texture3D Result;
		};

		template < typename R >
		struct ResourceToWrapperBasePtr
		{
			typedef TexturePtr Result;
		};

		template <>
		struct ResourceToWrapperBasePtr< ID3D10Buffer >
		{
			typedef BufferPtr Result;
		};

		template < typename R >
		typename ResourceToWrapperBasePtr<R> :: Result GetWrapper( const StagedResourceConfig& cfg )
		{
			return cfg.tex;
		};

		template <>
		BufferPtr GetWrapper< ID3D10Buffer > ( const StagedResourceConfig& cfg )
		{
			return cfg.buf;
		}

		template <typename R>
		ComPtr<R> CreateResource( const StagedResourceConfig& cfg, ID3D10Device* dev )
		{
			typedef ResourceToWrapper< R > :: Result Wrapper;
			Wrapper* wrapper = static_cast<  Wrapper* >( &*GetWrapper< R >( cfg ) );

			ComPtr<R> res;
			D3D10::CreateStagedResource( res, *wrapper, dev );

			return res;
		}

		template <typename R>
		R* GetResource( const StagedResourceConfig& cfg )
		{
			return static_cast<R*>( &*static_cast< ResourceToWrapper<R>::Result& > ( *cfg.tex ).GetResource() );
		}

		template <>
		ID3D10Buffer* GetResource<ID3D10Buffer>( const StagedResourceConfig& cfg )
		{
			return static_cast<ID3D10Buffer*>( &*static_cast< D3D10Buffer& > ( *cfg.buf ).GetResource() );
		}

	}

	//------------------------------------------------------------------------

	template <typename R>
	D3D10StagedResourceImpl<R>::D3D10StagedResourceImpl( const StagedResourceConfig& cfg, ID3D10Device *dev, UINT64 resourceSize  ) :
	Base( cfg ),
	mResourceSize( resourceSize )
	{
		mResource = CreateResource<R>( cfg, dev );
	}

	//------------------------------------------------------------------------

	template <typename R>
	D3D10StagedResourceImpl<R>::~D3D10StagedResourceImpl()
	{

	}

	//------------------------------------------------------------------------

	template <typename R>
	void
	D3D10StagedResourceImpl<R>::Sync( ID3D10Device * dev )
	{
		dev->CopyResource( &*mResource, GetResource<R>( GetConfig() ) );
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	template <typename R>
	UINT64
	D3D10StagedResourceImpl<R>::GetSizeImpl() const /*OVERRIDE*/
	{
		return mResourceSize;		
	}

	//------------------------------------------------------------------------

	namespace
	{		
		HRESULT Map( const ComPtr<ID3D10Buffer>& res, void **ptr )
		{
			return res->Map( D3D10_MAP_READ, D3D10_MAP_FLAG_DO_NOT_WAIT, ptr );
		}

		HRESULT Map( const ComPtr<ID3D10Texture1D>& res, void **ptr )
		{
			return res->Map( 0, D3D10_MAP_READ, D3D10_MAP_FLAG_DO_NOT_WAIT, ptr );
		}

		HRESULT Map( const ComPtr<ID3D10Texture2D>& res, void **ptr )
		{
			// TODO : Correct pitch handling!

			D3D10_MAPPED_TEXTURE2D mapped;
			HRESULT hr = res->Map( 0, D3D10_MAP_READ, D3D10_MAP_FLAG_DO_NOT_WAIT, &mapped );

			*ptr = mapped.pData;

			return hr;
		}

		HRESULT Map( const ComPtr<ID3D10Texture3D>& res, void **ptr )
		{
			// TODO : Correct pitch handling!

			D3D10_MAPPED_TEXTURE3D mapped;
			HRESULT hr = res->Map( 0, D3D10_MAP_READ, D3D10_MAP_FLAG_DO_NOT_WAIT, &mapped );

			*ptr = mapped.pData;

			return hr;
		}

		void Unmap( const ComPtr<ID3D10Buffer>& res )
		{
			res->Unmap();
		}

		template <typename T>
		void Unmap( const T& res )
		{
			res->Unmap( 0 );
		}

	}

	/*virtual*/

	template <typename R>
	bool
	D3D10StagedResourceImpl<R>::GetDataImpl( Bytes& oBytes ) /*OVERRIDE*/
	{		
		void* ptr;
		HRESULT hr = Map( mResource, &ptr );

		bool result = false;

		// this is because ATI has their own way ;]
		if( ptr )
		{
			memcpy( &oBytes[0], ptr, (size_t)mResourceSize );
			result = true;
		}

		if( hr ==  S_OK )
		{
			Unmap( mResource );
		}

		return result;
	}

	//------------------------------------------------------------------------
	
	template class D3D10StagedResourceImpl< ID3D10Buffer	>;
	template class D3D10StagedResourceImpl< ID3D10Texture1D	>;
	template class D3D10StagedResourceImpl< ID3D10Texture2D	>;
	template class D3D10StagedResourceImpl< ID3D10Texture3D	>;

}