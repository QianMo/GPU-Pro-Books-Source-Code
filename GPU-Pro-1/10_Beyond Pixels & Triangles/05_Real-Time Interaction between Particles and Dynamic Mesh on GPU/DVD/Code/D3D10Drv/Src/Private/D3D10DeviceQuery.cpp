#include "Precompiled.h"

#include "Wrap3D/Src/DeviceQueryConfig.h"

#include "D3D10DeviceQuery.h"
#include "D3D10DeviceQueryType.h"

#include "D3D10Exception.h"

namespace Mod
{

	D3D10DeviceQuery::D3D10DeviceQuery( const DeviceQueryConfig& cfg, ID3D10Device * dev ) :
	Base( cfg )
	{
		D3D10_QUERY_DESC desc = {};
		
		desc.Query = static_cast<const D3D10DeviceQueryType*>(cfg.type)->GetValue();

		ID3D10Query *result;
		D3D10_THROW_IF( dev->CreateQuery( &desc, &result ) );

		mResource.set( result );
		
	}

	//------------------------------------------------------------------------

	D3D10DeviceQuery::~D3D10DeviceQuery()
	{

	}

	//------------------------------------------------------------------------

	void D3D10DeviceQuery::BeginImpl()
	{
		mResource->Begin();
	}

	//------------------------------------------------------------------------

	void
	D3D10DeviceQuery::EndImpl()
	{
		mResource->End();
	}

	//------------------------------------------------------------------------

	bool D3D10DeviceQuery::GetDataImpl( void * data ) const
	{
		const D3D10DeviceQueryType* type = static_cast<const D3D10DeviceQueryType*>(GetConfig().type);
		return  mResource->GetData( data, static_cast<UINT>(type->GetDeviceDataSize()), 0) == S_OK;
	}

}