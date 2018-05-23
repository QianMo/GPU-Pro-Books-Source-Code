/*******************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*******************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beQuery.h"
#include <beGraphics/DX/beError.h>

namespace beGraphics
{

namespace DX11
{

// Creates a timing query.
lean::com_ptr<ID3D11Query, true> CreateTimingQuery(ID3D11Device *device)
{
	lean::com_ptr<ID3D11Query> query;

	D3D11_QUERY_DESC desc;
	desc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
	desc.MiscFlags = 0;

	BE_THROW_DX_ERROR_MSG(
		device->CreateQuery(&desc, query.rebind()),
		"ID3D11Device::CreateQuery()" );

	return query.transfer();
}

// Creates a timestamp query.
lean::com_ptr<ID3D11Query, true> CreateTimestampQuery(ID3D11Device *device)
{
	lean::com_ptr<ID3D11Query> query;

	D3D11_QUERY_DESC desc;
	desc.Query = D3D11_QUERY_TIMESTAMP;
	desc.MiscFlags = 0;

	BE_THROW_DX_ERROR_MSG(
		device->CreateQuery(&desc, query.rebind()),
		"ID3D11Device::CreateQuery()" );

	return query.transfer();
}

// Gets the frequency from the given timing query.
uint8 GetTimingFrequency(ID3D11DeviceContext *context, ID3D11Query *timingQuery)
{
	D3D11_QUERY_DATA_TIMESTAMP_DISJOINT data;

	while (context->GetData(timingQuery, &data, sizeof(data), 0) == S_FALSE);

	return data.Frequency;
}

// Gets the time stamp from the given timer query.
uint8 GetTimestamp(ID3D11DeviceContext *context, ID3D11Query *timerQuery)
{
	UINT64 data;

	while (context->GetData(timerQuery, &data, sizeof(data), 0) == S_FALSE);

	return data;
}

} // namespace

} // namespace