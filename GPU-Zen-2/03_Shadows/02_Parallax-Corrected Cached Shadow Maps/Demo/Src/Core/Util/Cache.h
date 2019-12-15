#ifndef __CACHE_H
#define __CACHE_H

#include "tbb/include/tbb/null_rw_mutex.h"
#include <algorithm>
#include "Merger.h"

template<class T, class COMPARE = PODCompare<typename T::Description, 8, 64> > class Cache
{
public:
  typedef typename T::Description Description;
  typedef typename T::CacheEntry CacheEntry;
  typedef typename T::UserParam UserParam;

  Cache(size_t sizeLimit = 0) : m_MaxIndex(sizeLimit - 1)
  {
    m_CacheEntries.reserve(sizeLimit);
  }
  ~Cache()
  {
    Clear();
  }
  void Init(const UserParam& p, bool bAllocate)
  {
    m_UserParam = p;
    if(bAllocate && m_Descriptions.size()>0)
      GetByIndex(m_Descriptions.size() - 1);
  }
  bool IsEmpty() const
  {
    return m_Descriptions.size()==0;
  }
  void Clear()
  {
    for(CacheEntries::iterator it=m_CacheEntries.begin(); it!=m_CacheEntries.end(); ++it)
      T::Free(m_UserParam, *it);
    m_CacheEntries.clear();
  }
  void Reset()
  {
    Clear();
    m_Descriptions.clear();
  }
  finline const CacheEntry& Get(const Description& d)
  {
    return GetByIndex(GetIndex(d));
  }
  finline size_t GetIndex(const Description& d)
  {
    size_t index = m_Descriptions.emit(d);
    _ASSERT(index<=m_MaxIndex && "cache capacity is exceeded");
    return std::min(index, m_MaxIndex);
  }
  finline const CacheEntry& GetByIndex(size_t index)
  {
    tbb::null_rw_mutex null;
    return GetByIndex(index, null, null);
  }

protected:
  typedef Merger<Description, COMPARE> Descriptions;
  typedef std::vector<CacheEntry> CacheEntries;

  Descriptions m_Descriptions;
  CacheEntries m_CacheEntries;
  UserParam m_UserParam;
  size_t m_MaxIndex;

  template<class M> finline const CacheEntry& GetByIndex(size_t index, M& entriesMutex, M& descMutex)
  {
    // Dinkumware's vector::size() is generally non-atomic, but calling it without taking the lock is okay 
    // for the moment as only fixed-size concurrent caches are supported.
    _ASSERT(index<m_Descriptions.size() && "invalid index");
    _ASSERT(index<=m_MaxIndex && "invalid index");
    if(index>=m_CacheEntries.size())
    {
      M::scoped_lock entriesLock(entriesMutex, true);
      for(size_t i=m_CacheEntries.size(); i<=index; ++i)
      {
        M::scoped_lock descLock(descMutex, false);
        const Description* pDesc = NULL;
        for(Descriptions::const_iterator it=m_Descriptions.begin(); it!=m_Descriptions.end(); ++it)
          if(it->second==i) { pDesc = &it->first; break; }
        _ASSERT(pDesc!=NULL && "internal structure is broken");
        m_CacheEntries.resize(m_CacheEntries.size() + 1);
        T::Allocate(m_UserParam, *pDesc, m_CacheEntries.back());
      }
    }
    return m_CacheEntries[index];
  }
};

template<class T, class COMPARE = PODCompare<typename T::Description, 8, 64> > class ConcurrentCache : public Cache<T, COMPARE>
{
public:
  ConcurrentCache(size_t sizeLimit) : Cache(sizeLimit)
  {
   // The size of concurrent cache must be limited as the cache returns unguarded references and does not 
   // provide mechanisms for locking the cache when reading from it.
    _ASSERT(sizeLimit>0);
  }
  finline const CacheEntry& ConcurrentGet(const Description& d)
  {
    return ConcurrentGetByIndex(ConcurrentGetIndex(d));
  }
  finline size_t ConcurrentGetIndex(const Description& d)
  {
    size_t index = m_Descriptions.emit(d, m_DescMutex);
    _ASSERT(index<=m_MaxIndex && "cache capacity is exceeded");
    return std::min(index, m_MaxIndex);
  }
  finline const CacheEntry& ConcurrentGetByIndex(size_t index)
  {
    return GetByIndex(index, m_EntriesMutex, m_DescMutex);
  }

protected:
  tbb::spin_rw_mutex m_EntriesMutex;
  tbb::spin_rw_mutex m_DescMutex;
};

#endif //#ifndef __CACHE_H
