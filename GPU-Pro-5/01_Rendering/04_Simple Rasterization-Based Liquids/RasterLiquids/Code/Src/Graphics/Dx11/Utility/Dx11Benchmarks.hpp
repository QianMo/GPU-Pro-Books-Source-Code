

#ifndef __GPU_BENCHMARKS_HPP__
#define __GPU_BENCHMARKS_HPP__

#include <d3dx11.h>
#include <Common/Common.hpp>

///<
class TimeStampQuery
{	

	ID3D11Query*		m_pQueryTimeStamp;
	ID3D11Query*		m_pQueryTimeStampDisjoint;
	uint64				m_ticksBegin;
	uint64				m_ticksEnd;

public:


    ///<
    TimeStampQuery(ID3D11Device* _pDevice);

    ~TimeStampQuery();

    void	Begin	(ID3D11DeviceContext* _pDeviceContext);
    float32 End		(ID3D11DeviceContext* _pDeviceContext);
};

#endif

