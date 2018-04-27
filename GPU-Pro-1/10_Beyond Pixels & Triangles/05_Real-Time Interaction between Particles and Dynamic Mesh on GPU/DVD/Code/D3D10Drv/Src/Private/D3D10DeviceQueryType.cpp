#include "Precompiled.h"

#include "Wrap3D/Src/DeviceQuery.h"

#include "D3D10DeviceQueryType.h"

namespace Mod
{

	//------------------------------------------------------------------------

	D3D10DeviceQueryType::D3D10DeviceQueryType( D3D10_QUERY q, const type_info& dataTI, UINT64 size ) : 
	mValue( q ),
	mDataTypeInfo( dataTI ),
	mDeviceDataSize( size )
	{

	}

	//------------------------------------------------------------------------

	D3D10DeviceQueryType::~D3D10DeviceQueryType()
	{

	}

	//------------------------------------------------------------------------

	D3D10_QUERY
	D3D10DeviceQueryType::GetValue() const
	{
		return mValue;
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceQueryType::RemapData( void* dest, const void* src ) const
	{
		RemapDataImpl( dest, src );
	}

	//------------------------------------------------------------------------

	UINT64
	D3D10DeviceQueryType::GetDeviceDataSize() const
	{
		return mDeviceDataSize;
	}

	//------------------------------------------------------------------------

	const type_info&
	D3D10DeviceQueryType::GetTypeInfoImpl() const
	{
		return mDataTypeInfo;
	}

	//------------------------------------------------------------------------
	//------------------------------------------------------------------------

	template < D3D10_QUERY q >
	struct QueryDesc;

#define MD_CREATE_QUERY_DESC_SPEC(qtype,datatype,d3ddatatype)	\
	template <>													\
	struct QueryDesc<qtype>										\
	{															\
		static const D3D10_QUERY query = qtype;					\
		typedef datatype DataType;								\
		typedef d3ddatatype D3D10DataType;						\
	};

	MD_CREATE_QUERY_DESC_SPEC(D3D10_QUERY_SO_STATISTICS,DQ_SOStatistics,D3D10_QUERY_DATA_SO_STATISTICS)

#undef MD_CREATE_QUERY_DESC_SPEC


	//------------------------------------------------------------------------

	template <D3D10_QUERY q>
	D3D10DeviceQueryTypeImpl<q>::D3D10DeviceQueryTypeImpl() : 
	Base( q, typeid(QueryDesc<q>::DataType ), sizeof( QueryDesc<q>::D3D10DataType ) )
	{
		
	}

	//------------------------------------------------------------------------

	template <D3D10_QUERY q>
	D3D10DeviceQueryTypeImpl<q>::~D3D10DeviceQueryTypeImpl()
	{

	}

	//------------------------------------------------------------------------

	template <>
	void
	D3D10DeviceQueryTypeImpl<D3D10_QUERY_SO_STATISTICS>::RemapDataImpl( void* dest, const void* src ) const
	{
		typedef const QueryDesc<Query>::D3D10DataType ConstD3D10DataType;
		DQ_SOStatistics&		dss = *static_cast<DQ_SOStatistics*>( dest );
		ConstD3D10DataType&		sss = *static_cast<ConstD3D10DataType*>( src );

		dss.numStoredElems	= sss.NumPrimitivesWritten;
		dss.numTotalElems	= sss.PrimitivesStorageNeeded;
	}

	//------------------------------------------------------------------------

	template class D3D10DeviceQueryTypeImpl<D3D10_QUERY_SO_STATISTICS>;

}