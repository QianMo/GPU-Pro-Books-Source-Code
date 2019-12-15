#ifndef __ATTRIBUTE_H
#define __ATTRIBUTE_H

#include <string>
#include "../../Util/Hasher.h"

class Attribute
{
public:
  typedef Delegate<void (const char*)> Setter;

  Attribute(const char* pszName) : m_pszName(pszName) { m_Hasher.Hash(pszName); }
  Attribute(const char* pszName, const Setter& setter) : m_pszName(pszName), m_Setter(setter) { m_Hasher.Hash(pszName); }
  finline void Set(const char* pValue) const { m_Setter(pValue); }
  finline bool operator< (const Attribute& a) const { return Compare(a)<0; }
  finline bool operator== (const Attribute& a) const { return Compare(a)==0; }

protected:
  Setter m_Setter;
  const char* m_pszName;
  Hasher m_Hasher;

  finline int Compare(const Attribute& a) const
  {
    int r = m_Hasher.Compare(a.m_Hasher);
    if(!r) r = strcmp(m_pszName, a.m_pszName);
    return r;
  }
};

class Attributes
{
public:
  Attributes(size_t n, Attribute* p) : m_nAttr(n), m_pAttr(p)
  {
    std::sort(m_pAttr, m_pAttr + m_nAttr);
  }
  Attribute* Find(const char* pszName) const
  {
    Attribute a(pszName);
    Attribute* p = std::lower_bound(m_pAttr, m_pAttr + m_nAttr, a);
    return (p!=(m_pAttr + m_nAttr) && *p==a) ? p : NULL;
  }

protected:
  Attribute* m_pAttr;
  size_t m_nAttr;
};

#endif //#ifndef __ATTRIBUTE_H
