#ifndef D3D10DRV_D3D10RESOURCEMAP_H_INCLUDED
#define D3D10DRV_D3D10RESOURCEMAP_H_INCLUDED

#include "Wrap3D/Src/Format.h"

namespace Mod
{
	template <typename T, typename R>
	class D3D10ResourceMap
	{
		// types
	public:
		typedef R ResourceType;

		// construction/ destruction
	public:
		D3D10ResourceMap( const D3D10FormatMap& fmtMap, const ResourceType& res, D3D10_MAP mapType, UINT32 subResIdx = 0 );
		~D3D10ResourceMap();

		// manipulation/ access
	public:
		T&			operator [] ( size_t idx );
		const T&	operator [] ( size_t idx ) const;

		UINT32		GetMappedSize() const;
		void*		GetRawMappedPtr() const;

	private:
		ResourceType	mResource;
		T*				mLockedData;
		UINT32			mNumItems;
		UINT32			mLockedSubresIdx;
	};

	//------------------------------------------------------------------------

	template <typename R>
	void MapD3D10Resource( R res, D3D10_MAP mapType, UINT32 subResIdx, void** data );

	template <typename R>
	void UnmapD3D10Resource( R res, UINT32 subResIdx );

	template <typename R>
	UINT32 GetD3D10ResourceSize( const D3D10FormatMap& fmtMap, R res );

	//------------------------------------------------------------------------

	template <typename T, typename R>
	D3D10ResourceMap<T, R>::D3D10ResourceMap( const D3D10FormatMap& fmtMap, const ResourceType& res, D3D10_MAP mapType, UINT32 subResIdx /*= 0*/ ) : 
	mResource( res ),
	mNumItems( 0 ),
	mLockedSubresIdx( subResIdx )
	{
		UINT32 resSize = GetD3D10ResourceSize( fmtMap, &*res );
		MD_FERROR_ON_TRUE( resSize % sizeof( T ) );

		mNumItems = resSize / sizeof( T );

		MapD3D10Resource( &*mResource, mapType, subResIdx, reinterpret_cast<void**>( &mLockedData ) );
	}

	//------------------------------------------------------------------------

	template <typename T, typename R>
	D3D10ResourceMap<T, R>::~D3D10ResourceMap()
	{
		UnmapD3D10Resource( &*mResource, mLockedSubresIdx );
	}

	//------------------------------------------------------------------------

	template <typename T, typename R>
	T&
	D3D10ResourceMap<T, R>::operator [] ( size_t idx )
	{
		MD_ASSERT( idx < mNumItems );
		return mLockedData[ idx ];
	}

	//------------------------------------------------------------------------

	template <typename T, typename R>
	const T&
	D3D10ResourceMap<T, R>::operator [] ( size_t idx ) const
	{
		MD_ASSERT( idx < mNumItems );
		return mLockedData[ idx ];
	}

	//------------------------------------------------------------------------

	template <typename T, typename R>
	UINT32
	D3D10ResourceMap<T, R>::GetMappedSize() const
	{
		return mNumItems * sizeof( T );
	}

	//------------------------------------------------------------------------

	template <typename T, typename R>
	void*
	D3D10ResourceMap<T, R>::GetRawMappedPtr() const
	{
		return mLockedData;
	}

	//------------------------------------------------------------------------
	// use common template for textures and specialized one for buffer

	template <typename T>
	struct D3D10MappingInfo	{	};

	template <> struct D3D10MappingInfo< ID3D10Texture1D* > { typedef void* MappedType; };
	template <> struct D3D10MappingInfo< ID3D10Texture2D* > { typedef D3D10_MAPPED_TEXTURE2D MappedType; };
	template <> struct D3D10MappingInfo< ID3D10Texture3D* > { typedef D3D10_MAPPED_TEXTURE3D MappedType; };

	void D3D10PlaceMappedPtr( void** ptr, void *source )							{ *ptr = source;		}
	void D3D10PlaceMappedPtr( void** ptr, const D3D10_MAPPED_TEXTURE2D& source )	{ *ptr = source.pData;	}
	void D3D10PlaceMappedPtr( void** ptr, const D3D10_MAPPED_TEXTURE3D& source )	{ *ptr = source.pData;	}

	template <typename R>
	void MapD3D10Resource( R res, D3D10_MAP mapType, UINT32 subResIdx, void** data )
	{
		D3D10MappingInfo< R > :: MappedType mapped;
		res->Map( subResIdx, mapType, 0, &mapped );
		D3D10PlaceMappedPtr( data, mapped );
	}

	template <>
	void MapD3D10Resource( ID3D10Buffer* res, D3D10_MAP mapType, UINT32 /*subResIdx*/, void** data )
	{
		res->Map( mapType, 0, data );
		MD_FERROR_ON_FALSE( *data );
	}

	//------------------------------------------------------------------------

	template <typename R>
	void UnmapD3D10Resource( R res, UINT32 subResIdx )
	{
		res->Unmap( subResIdx );
	}

	template <>
	void UnmapD3D10Resource( ID3D10Buffer* res, UINT32 /*subResIdx*/ )
	{
		res->Unmap();
	}

	//------------------------------------------------------------------------

	template <>
	UINT32 GetD3D10ResourceSize( const D3D10FormatMap& /*fmtMap*/, ID3D10Buffer* res )
	{
		D3D10_BUFFER_DESC desc;

		res->GetDesc( &desc );

		return desc.ByteWidth;
	}

	template <>
	UINT32 GetD3D10ResourceSize( const D3D10FormatMap& fmtMap, ID3D10Texture1D* tex )
	{
		D3D10_TEXTURE1D_DESC desc;
		tex->GetDesc( &desc );

		return GetElementsByteSizeChecked( fmtMap.GetFormat( desc.Format ), desc.Width );
	}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4554)
#endif

	template <>
	UINT32 GetD3D10ResourceSize( const D3D10FormatMap& fmtMap, ID3D10Texture2D* tex )
	{
		D3D10_TEXTURE2D_DESC desc;
		tex->GetDesc( &desc );

		// only allow POW 2 textures for now
		MD_FERROR_ON_FALSE( (desc.Width		& ~desc.Width	+ 1) == desc.Width	);
		MD_FERROR_ON_FALSE( (desc.Height	& ~desc.Height	+ 1) == desc.Height	);

		return GetElementsByteSizeChecked( fmtMap.GetFormat( desc.Format ), desc.Width * desc.Height );
	}

	template <>
	UINT32 GetD3D10ResourceSize( const D3D10FormatMap& fmtMap, ID3D10Texture3D* tex )
	{
		D3D10_TEXTURE3D_DESC desc;
		tex->GetDesc( &desc );

		// only allow POW 3 textures for now
		MD_FERROR_ON_FALSE( desc.Width	& ~desc.Width	+ 1 == desc.Width	);
		MD_FERROR_ON_FALSE( desc.Height	& ~desc.Height	+ 1 == desc.Height	);
		MD_FERROR_ON_FALSE( desc.Depth	& ~desc.Depth	+ 1 == desc.Depth	);

		return GetElementsByteSizeChecked( fmtMap.GetFormat( desc.Format ), desc.Width * desc.Height );
	}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

}

#endif