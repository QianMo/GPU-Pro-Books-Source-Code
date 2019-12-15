#ifndef __MERGER_H
#define __MERGER_H

#include "tbb/include/tbb/spin_rw_mutex.h"
#include "../Math/Math.h"
#include <hash_map>
#include "Hasher.h"

template<class T, int BUCKET_SIZE = 4, int MIN_BUCKETS = 1024> class PODCompare
{
public:
  enum { bucket_size = BUCKET_SIZE, min_buckets = MIN_BUCKETS };

  finline size_t operator() (const T& a) const
  {
    Hasher h;
    return h.Hash(a);
  }
  finline bool operator() (const T& a, const T& b) const
  {
    return memcmp(&a, &b, sizeof(T))<0;
  }
};

template<class T, class COMPARE=PODCompare<T> > class Merger : public stdext::hash_map<T, size_t, COMPARE>
{
public:
  size_t emit(const T& e)
  {
    const_iterator it = find(e);
    if(it!=end())
      return it->second;
    size_t index = size();
    insert(std::pair<T, size_t>(e, index));
    return index;
  }
  size_t emit(const T& e, tbb::spin_rw_mutex& mutex)
  {
    for(bool bWrite=false;; bWrite=true)
    {
      tbb::spin_rw_mutex::scoped_lock lock(mutex, bWrite);
      const_iterator it = find(e);
      if(it!=end())
        return it->second;
      if(!bWrite)
        continue;
      size_t index = size();
      insert(std::pair<T, size_t>(e, index));
      return index;
    }
  }
  template<class VECTOR> void copy_data(VECTOR& vector) const
  {
    vector.resize(size());
    for(const_iterator it=begin(); it!=end(); ++it)
      vector[it->second] = it->first;
  }
};

template<class T> void PrepareTriangle(T *i)
{
  if(i[1]<i[0] && i[1]<i[2])
  {
    std::swap(i[0], i[1]);
    std::swap(i[1], i[2]);
  }
  else if(i[2]<i[0] && i[2]<i[1])
  {
    std::swap(i[0], i[1]);
    std::swap(i[0], i[2]);
  }
}

#endif //#ifndef __MERGER_H
