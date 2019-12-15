#ifndef __RADIX_SORT
#define __RADIX_SORT

#include "../Math/Math.h"

template<size_t c_KeyOffset, size_t c_KeySize, unsigned char c_OrderMask, class T>
  inline T* RadixSort(size_t NItems, T* pItems, T* pTemp)
{
  const size_t c_HistogramBufferSize = c_KeySize*256*sizeof(size_t);
  size_t* pHistogramBuffer = (size_t*)alloca(c_HistogramBufferSize);
  memset(pHistogramBuffer, 0, c_HistogramBufferSize);
  unsigned char* restrict pRawKey = ((unsigned char*)pItems) + c_KeyOffset;
  for(size_t i=0; i<NItems; ++i)
  {
    size_t* restrict pHistogram = pHistogramBuffer;
    for(size_t j=0; j<c_KeySize; ++j, pHistogram+=256)
      ++pHistogram[c_OrderMask ^ pRawKey[j]];
    pRawKey += sizeof(T);
  }
  size_t* restrict pHistogram = pHistogramBuffer;
  T* restrict pSrc = pItems;
  T* restrict pDst = pTemp;
  for(size_t i=0; i<c_KeySize; ++i, pHistogram+=256)
  {
    size_t offset = 0;
    for(size_t j=0; j<256; ++j)
    {
      size_t k = pHistogram[j];
      pHistogram[j] = offset;
      offset += k;
    }
    unsigned char* restrict pRawKey = ((unsigned char*)pSrc) + c_KeyOffset + i;
    for(size_t j=0; j<NItems; ++j)
    {
      pDst[pHistogram[c_OrderMask ^ *pRawKey]++] = pSrc[j];
      pRawKey += sizeof(T);
    }
    std::swap(pDst, pSrc);
  }
  return pSrc;
}

template<size_t c_KeyOffset, size_t c_KeySize, class T>
  inline T* RadixSort_Ascending(size_t NItems, T *pItems, T* pTemp)
{
  return RadixSort<c_KeyOffset, c_KeySize, 0>(NItems, pItems, pTemp);
}

template<size_t c_KeyOffset, size_t c_KeySize, class T>
  inline T* RadixSort_Descending(size_t NItems, T *pItems, T* pTemp)
{
  return RadixSort<c_KeyOffset, c_KeySize, 255>(NItems, pItems, pTemp);
}

#endif //#ifndef __RADIX_SORT
