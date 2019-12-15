#ifndef __VERTEX_FORMAT
#define __VERTEX_FORMAT

#include <vector>
#include <d3d11.h>
#include "../../Math/Math.h"

class MemoryBuffer;
struct _D3DVERTEXELEMENT9;

class VertexIterator
{
public:
  void Next() { m_pData = (char*)m_pData + m_Stride; }
  finline const Vec4 Get() const { return Vec4(GetFloat(0), GetFloat(1), GetFloat(2), GetFloat(3)); }
  finline void Set(const Vec4& a) const { SetFloat(0, a.x); SetFloat(1, a.y); SetFloat(2, a.z); SetFloat(3, a.w);}
  finline void SetSignedNormalized(const Vec4& a) const { Set(Vec4::Max(Vec4(-1.0f), Vec4::Min(Vec4(1.0f), a))); }
  finline void SetUnsignedNormalized(const Vec4& a) const { Set(Vec4::Max(Vec4::Zero(), Vec4::Min(Vec4(1.0f), a))); }

  virtual float __fastcall GetFloat(unsigned) const PURE;
  virtual int __fastcall GetInteger(unsigned) const PURE;
  virtual void __fastcall SetFloat(unsigned, float) const PURE;
  virtual void __fastcall SetInteger(unsigned, int) const PURE;
  virtual size_t __fastcall GetNumberOfElements() const PURE;

protected:
  void* m_pData;
  size_t m_Stride;
};

class VertexFormatDesc : public std::vector<D3D11_INPUT_ELEMENT_DESC>
{
public:
  VertexFormatDesc() { }
  VertexFormatDesc(const _D3DVERTEXELEMENT9*);
  VertexFormatDesc(const D3D11_INPUT_ELEMENT_DESC*, size_t);
  finline const D3D11_INPUT_ELEMENT_DESC* ptr() const { return size() ? &front() : NULL; }

  VertexIterator* new_iterator(int, void*, size_t) const;
  int find(const char*, unsigned) const;
  size_t GetByteOffset(int) const;
  size_t GetMinVertexSize() const;
  void Serialize(MemoryBuffer&) const;
  void Deserialize(MemoryBuffer&);
  void FinishAssembly();
  void GetDesc9(_D3DVERTEXELEMENT9*) const;

  finline int compare(const VertexFormatDesc& a) const
  {
    int r = (size()>a.size()) - (size()<a.size());
    if(!r) r = memcmp(ptr(), a.ptr(), size()*sizeof(D3D11_INPUT_ELEMENT_DESC));
    return r;
  }
};

#endif //#ifndef __VERTEX_FORMAT
