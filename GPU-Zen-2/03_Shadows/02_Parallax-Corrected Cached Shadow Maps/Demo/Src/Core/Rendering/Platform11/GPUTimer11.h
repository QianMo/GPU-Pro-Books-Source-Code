#ifndef __GPU_TIMER11
#define __GPU_TIMER11

#include <stdint.h>
#include "Platform11.h"

class GPUTimer
{
public:
  class Scope
  {
  public:
    finline Scope(GPUTimer* t) : m_Timer(t) { m_Timer->Begin(); }
    finline ~Scope() { m_Timer->End(); }
  protected:
    GPUTimer* m_Timer;
  };

  GPUTimer()
  {
    m_pTimeQuery[0] = m_pTimeQuery[1] = m_pFreqQuery = NULL;
    m_Time[0] = m_Time[1] = m_Freq = 1;
    m_Processed = true;
    m_QueryInProgress = false;
  }
  HRESULT Init()
  {
    D3D11_QUERY_DESC timeDesc = { D3D11_QUERY_TIMESTAMP, 0 };
    HRESULT hr = Platform::GetD3DDevice()->CreateQuery(&timeDesc, &m_pTimeQuery[0]);
    hr = SUCCEEDED(hr) ? Platform::GetD3DDevice()->CreateQuery(&timeDesc, &m_pTimeQuery[1]) : hr;
    D3D11_QUERY_DESC freqDesc = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
    hr = SUCCEEDED(hr) ? Platform::GetD3DDevice()->CreateQuery(&freqDesc, &m_pFreqQuery) : hr;
    return hr;
  }
  void Clear()
  {
    SAFE_RELEASE(m_pTimeQuery[0]);
    SAFE_RELEASE(m_pTimeQuery[1]);
    SAFE_RELEASE(m_pFreqQuery);
  }
  void Begin(DeviceContext11& dc = Platform::GetImmediateContext(), bool forceNewQuery = false)
  {
    if(forceNewQuery | m_Processed)
    {
      m_Processed = false;
      m_QueryInProgress = true;
      dc.DoNotFlushToDevice()->End(m_pTimeQuery[0]);
      dc.DoNotFlushToDevice()->Begin(m_pFreqQuery);
    }
  }
  void End(DeviceContext11& dc = Platform::GetImmediateContext())
  {
    if(m_QueryInProgress)
    {
      dc.DoNotFlushToDevice()->End(m_pTimeQuery[1]);
      dc.DoNotFlushToDevice()->End(m_pFreqQuery);
      m_QueryInProgress = false;
    }
  }
  bool IsProcessed(DeviceContext11& dc = Platform::GetImmediateContext())
  {
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT d;
    if(!(m_Processed | m_QueryInProgress) &&
       dc.DoNotFlushToDevice()->GetData(m_pTimeQuery[0], &m_Time[0], sizeof(uint64_t), 0)==S_OK &&
       dc.DoNotFlushToDevice()->GetData(m_pTimeQuery[1], &m_Time[1], sizeof(uint64_t), 0)==S_OK &&
       dc.DoNotFlushToDevice()->GetData(m_pFreqQuery, &d, sizeof(d), 0)==S_OK)
    {
      m_Freq = d.Frequency;
      m_Processed = true;
    }
    return m_Processed;
  }
  finline double GetTime()
  {
    return IsProcessed() ? (double)(m_Time[1] - m_Time[0])/(double)m_Freq : FLT_MAX;
  }

protected:
  uint64_t m_Time[2], m_Freq;
  ID3D11Query* m_pTimeQuery[2];
  ID3D11Query* m_pFreqQuery;
  bool m_Processed, m_QueryInProgress;
};

#endif //#ifndef __GPU_TIMER11
