#ifndef __DELEGATE_H
#define __DELEGATE_H

#define SRUTIL_DELEGATE_PREFERRED_SYNTAX
#pragma push_macro("private")
#define private protected // need access to internals
#include "delegate.hpp"
#pragma pop_macro("private")

template<typename T> class Delegate : public srutil::delegate<T>
{
public:
  inline Delegate() { }
  inline Delegate(const srutil::delegate<T>& d) : srutil::delegate<T>::delegate(d) { }
  inline bool operator == (const Delegate& a) const { return object_ptr==a.object_ptr && stub_ptr==a.stub_ptr; }
  inline bool operator != (const Delegate& a) const { return object_ptr!=a.object_ptr || stub_ptr!=a.stub_ptr; }
};

#endif //#ifndef __DELEGATE_H
