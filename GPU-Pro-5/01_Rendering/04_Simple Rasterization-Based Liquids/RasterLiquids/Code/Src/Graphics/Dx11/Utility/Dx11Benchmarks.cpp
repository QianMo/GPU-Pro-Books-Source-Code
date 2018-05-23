
#include <Graphics/Dx11/Utility/Dx11Benchmarks.hpp>

TimeStampQuery::TimeStampQuery(ID3D11Device* _pDevice)
{
	D3D11_QUERY_DESC	m_QueryDesc = {D3D11_QUERY_TIMESTAMP,0};
	HRESULT cc = _pDevice->CreateQuery(&m_QueryDesc, &m_pQueryTimeStamp);
	ASSERT(cc==S_OK, "Failed to create count.  ");

	m_QueryDesc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
	cc = _pDevice->CreateQuery(&m_QueryDesc, &m_pQueryTimeStampDisjoint);
	ASSERT(cc==S_OK, "Failed to create count.  ");
}

TimeStampQuery::~TimeStampQuery()
{
	M::Release(&m_pQueryTimeStamp);
	M::Release(&m_pQueryTimeStampDisjoint);
}

void TimeStampQuery::Begin(ID3D11DeviceContext* _pDeviceContext)
{
	_pDeviceContext->Begin(m_pQueryTimeStampDisjoint);
	
	_pDeviceContext->End(m_pQueryTimeStamp);

	while( S_OK != _pDeviceContext->GetData(m_pQueryTimeStamp, &m_ticksBegin, sizeof(UINT64), 0) )
	{
	
	}

}

float32 TimeStampQuery::End(ID3D11DeviceContext* _pDeviceContext)
{

	_pDeviceContext->End(m_pQueryTimeStamp);
	
	while( S_OK != _pDeviceContext->GetData(m_pQueryTimeStamp, &m_ticksEnd, sizeof(UINT64), 0) )
	{
	}

	_pDeviceContext->End(m_pQueryTimeStampDisjoint);
	

	D3D10_QUERY_DATA_TIMESTAMP_DISJOINT frequency;
	while( S_OK != _pDeviceContext->GetData(m_pQueryTimeStampDisjoint, &frequency, sizeof(D3D10_QUERY_DATA_TIMESTAMP_DISJOINT), 0) )
	{
	}

	if (!frequency.Disjoint)
	{
		return static_cast<float32>(m_ticksEnd-m_ticksBegin)/static_cast<float32>(frequency.Frequency);
	}

	return 0;

}