/* Unordered set of pointers:
     - O(1) insertion and removal;
     - intrusive implementation. */

#ifndef __PTR_SET
#define __PTR_SET

#include <vector>

class ptr_set_handle
{
public:
  ptr_set_handle() : m_Index(-1) { }
  bool is_inserted() const { return m_Index>=0; }
  void set(int i) { m_Index = i; }
  int get() const { return m_Index; }

protected:
  int m_Index;
};

template<class T, ptr_set_handle& (T::*get_handle)()> class ptr_set : public std::vector<T*>
{
public:
  inline void insert(T* t, bool MayBeAlreadyInserted = false)
  {
    ptr_set_handle& h = (t->*get_handle)();
    if(h.is_inserted())
    {
      _ASSERT(MayBeAlreadyInserted && "trying to insert a pointer that was already inserted into the set");
    }
    else
    {
      h.set(size());
      push_back(t);
    }
  }
  inline void remove(T* t, bool MayBeNotInserted = false)
  {
    ptr_set_handle& h = (t->*get_handle)();
    if(h.is_inserted())
    {
      (back()->*get_handle)() = h;
      at(h.get()) = back();
      pop_back();
      h.set(-1);
    }
    else
    {
      _ASSERT(MayBeNotInserted && "trying to remove a pointer that was not inserted into the set");
    }
  }
};

#endif //#ifndef __PTR_SET
