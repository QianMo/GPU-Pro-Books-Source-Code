#include "PreCompile.h"
#include "VertexFormat.h"
#include "../../Util/MemoryBuffer.h"
#include "../Platform11/Platform11.h"

#pragma warning(push)
#pragma warning(disable:4244) // suppress int-to-float conversion warning in D3DXFLOAT16 ctor

template<class T, unsigned c_NElements, unsigned c_NormalizationConst> class VertexIteratorImpl : public VertexIterator
{
public:
  VertexIteratorImpl(void *pData, size_t stride)
  {
    m_pData = pData;
    m_Stride = stride;
  }
  virtual int __fastcall GetInteger(unsigned i) const
  {
    return i<c_NElements ? int(*((T*)m_pData + i)) : 0;
  }
  virtual void __fastcall SetInteger(unsigned i, int n) const
  {
    if(i>=c_NElements) return;
    *((T*)m_pData + i) = T(n);
  }
  virtual float __fastcall GetFloat(unsigned i) const
  {
    if(i>=c_NElements) return 0.0f;
    float f = (float)*((T*)m_pData + i);
    return c_NormalizationConst!=1 ? f/float(c_NormalizationConst) : f;
  }
  virtual void __fastcall SetFloat(unsigned i, float f) const
  {
    if(i>=c_NElements) return;
    *((T*)m_pData + i) = T(c_NormalizationConst!=1 ? f*(float)c_NormalizationConst : f);
  }
  virtual size_t __fastcall GetNumberOfElements() const
  {
    return c_NElements;
  }
};

#pragma warning(pop)

VertexIterator* VertexFormatDesc::new_iterator(int elementIndex, void* pData, size_t stride) const
{
  pData = (char*)pData + GetByteOffset(elementIndex);
  switch(elementIndex>=0 ? at(elementIndex).Format : DXGI_FORMAT_UNKNOWN)
  {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return new VertexIteratorImpl<float, 4, 1>(pData, stride);
    case DXGI_FORMAT_R32G32B32_FLOAT:    return new VertexIteratorImpl<float, 3, 1>(pData, stride);
    case DXGI_FORMAT_R32G32_FLOAT:       return new VertexIteratorImpl<float, 2, 1>(pData, stride);
    case DXGI_FORMAT_R32_FLOAT:          return new VertexIteratorImpl<float, 1, 1>(pData, stride);
    case DXGI_FORMAT_R32G32B32A32_UINT:  return new VertexIteratorImpl<unsigned, 4, 1>(pData, stride);
    case DXGI_FORMAT_R32G32B32_UINT:     return new VertexIteratorImpl<unsigned, 3, 1>(pData, stride);
    case DXGI_FORMAT_R32G32_UINT:        return new VertexIteratorImpl<unsigned, 2, 1>(pData, stride);
    case DXGI_FORMAT_R32_UINT:           return new VertexIteratorImpl<unsigned, 1, 1>(pData, stride);
    case DXGI_FORMAT_R32G32B32A32_SINT:  return new VertexIteratorImpl<int, 4, 1>(pData, stride);
    case DXGI_FORMAT_R32G32B32_SINT:     return new VertexIteratorImpl<int, 3, 1>(pData, stride);
    case DXGI_FORMAT_R32G32_SINT:        return new VertexIteratorImpl<int, 2, 1>(pData, stride);
    case DXGI_FORMAT_R32_SINT:           return new VertexIteratorImpl<int, 1, 1>(pData, stride);
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return new VertexIteratorImpl<D3DXFLOAT16, 4, 1>(pData, stride);
    case DXGI_FORMAT_R16G16_FLOAT:       return new VertexIteratorImpl<D3DXFLOAT16, 2, 1>(pData, stride);
    case DXGI_FORMAT_R16_FLOAT:          return new VertexIteratorImpl<D3DXFLOAT16, 1, 1>(pData, stride);
    case DXGI_FORMAT_R16G16B16A16_UNORM: return new VertexIteratorImpl<unsigned short, 4, USHRT_MAX>(pData, stride);
    case DXGI_FORMAT_R16G16_UNORM:       return new VertexIteratorImpl<unsigned short, 2, USHRT_MAX>(pData, stride);
    case DXGI_FORMAT_R16_UNORM:          return new VertexIteratorImpl<unsigned short, 1, USHRT_MAX>(pData, stride);
    case DXGI_FORMAT_R16G16B16A16_UINT:  return new VertexIteratorImpl<unsigned short, 4, 1>(pData, stride);
    case DXGI_FORMAT_R16G16_UINT:        return new VertexIteratorImpl<unsigned short, 2, 1>(pData, stride);
    case DXGI_FORMAT_R16_UINT:           return new VertexIteratorImpl<unsigned short, 1, 1>(pData, stride);
    case DXGI_FORMAT_R16G16B16A16_SNORM: return new VertexIteratorImpl<short, 4, SHRT_MAX>(pData, stride);
    case DXGI_FORMAT_R16G16_SNORM:       return new VertexIteratorImpl<short, 2, SHRT_MAX>(pData, stride);
    case DXGI_FORMAT_R16_SNORM:          return new VertexIteratorImpl<short, 1, SHRT_MAX>(pData, stride);
    case DXGI_FORMAT_R16G16B16A16_SINT:  return new VertexIteratorImpl<short, 4, 1>(pData, stride);
    case DXGI_FORMAT_R16G16_SINT:        return new VertexIteratorImpl<short, 2, 1>(pData, stride);
    case DXGI_FORMAT_R16_SINT:           return new VertexIteratorImpl<short, 1, 1>(pData, stride);
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:     return new VertexIteratorImpl<unsigned char, 4, UCHAR_MAX>(pData, stride);
    case DXGI_FORMAT_R8G8_UNORM:         return new VertexIteratorImpl<unsigned char, 2, UCHAR_MAX>(pData, stride);
    case DXGI_FORMAT_R8_UNORM:           return new VertexIteratorImpl<unsigned char, 1, UCHAR_MAX>(pData, stride);
    case DXGI_FORMAT_R8G8B8A8_UINT:      return new VertexIteratorImpl<unsigned char, 4, 1>(pData, stride);
    case DXGI_FORMAT_R8G8_UINT:          return new VertexIteratorImpl<unsigned char, 2, 1>(pData, stride);
    case DXGI_FORMAT_R8_UINT:            return new VertexIteratorImpl<unsigned char, 1, 1>(pData, stride);
    case DXGI_FORMAT_R8G8B8A8_SNORM:     return new VertexIteratorImpl<char, 4, CHAR_MAX>(pData, stride);
    case DXGI_FORMAT_R8G8_SNORM:         return new VertexIteratorImpl<char, 2, CHAR_MAX>(pData, stride);
    case DXGI_FORMAT_R8_SNORM:           return new VertexIteratorImpl<char, 1, CHAR_MAX>(pData, stride);
    case DXGI_FORMAT_R8G8B8A8_SINT:      return new VertexIteratorImpl<char, 4, 1>(pData, stride);
    case DXGI_FORMAT_R8G8_SINT:          return new VertexIteratorImpl<char, 2, 1>(pData, stride);
    case DXGI_FORMAT_R8_SINT:            return new VertexIteratorImpl<char, 1, 1>(pData, stride);
  }
  return NULL;
}

int VertexFormatDesc::find(const char* pszSemanticName, unsigned semanticIndex) const
{
  for(size_t i=0; i<size(); ++i)
    if(!strcmp(at(i).SemanticName, pszSemanticName) && at(i).SemanticIndex==semanticIndex)
      return i;
  return -1;
}

size_t VertexFormatDesc::GetByteOffset(int elementIndex) const
{
  size_t offset = 0;
  for(int i=elementIndex; i>=0; --i)
  {
    if(i!=elementIndex)
      offset += Platform::GetFormatBitsPerPixel(at(i).Format)/8;
    if(at(i).AlignedByteOffset!=D3D11_APPEND_ALIGNED_ELEMENT)
      return offset + at(i).AlignedByteOffset;
  }
  return offset;
}

size_t VertexFormatDesc::GetMinVertexSize() const
{
  size_t s = 0;
  for(const_iterator it=begin(); it!=end(); ++it)
    s += Platform::GetFormatBitsPerPixel(it->Format)/8;
  return s;
}

void VertexFormatDesc::Serialize(MemoryBuffer& out) const
{
  out.Write<unsigned>(size());
  for(const_iterator it=begin(); it!=end(); ++it)
  {
    D3D11_INPUT_ELEMENT_DESC d = *it;
    out.Write(d.SemanticName);
    d.SemanticName = NULL;
    out.Write(d);
  }
}

void VertexFormatDesc::Deserialize(MemoryBuffer& in)
{
  resize(in.Read<unsigned>());
  for(iterator it=begin(); it!=end(); ++it)
  {
    char* pSemanticName = in.Ptr<char>();
    in.Seek(in.Position() + strlen(pSemanticName) + 1);
    in.Read(*it);
    it->SemanticName = pSemanticName;
  }
  FinishAssembly();
}

VertexFormatDesc::VertexFormatDesc(const D3D11_INPUT_ELEMENT_DESC* pDesc, size_t nElements) :
  std::vector<D3D11_INPUT_ELEMENT_DESC>::vector(pDesc, pDesc + nElements)
{
  FinishAssembly();
}

class StringsPool
{
public:
  ~StringsPool()
  {
    for(auto it=m_Pool.begin(); it!=m_Pool.end(); ++it)
      delete[] *it;
  }
  const char* Add(const std::string& a)
  {
    for(auto it=m_Pool.begin(); it!=m_Pool.end(); ++it)
      if(!strcmp(*it, a.c_str()))
        return *it;
    m_Pool.push_back(new char[a.length() + 1]);
    strcpy(m_Pool.back(), a.c_str());
    return m_Pool.back();
  }

protected:
  std::vector<char*> m_Pool;
} s_StringsPool;

void VertexFormatDesc::FinishAssembly()
{
  for(iterator it = begin(); it!=end(); ++it)
    it->SemanticName = s_StringsPool.Add(it->SemanticName);
}

static DXGI_FORMAT s_TypeToFormat[] = 
{
    DXGI_FORMAT_R32_FLOAT,
    DXGI_FORMAT_R32G32_FLOAT,
    DXGI_FORMAT_R32G32B32_FLOAT,
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    DXGI_FORMAT_B8G8R8A8_UNORM,
    DXGI_FORMAT_R8G8B8A8_UINT,
    DXGI_FORMAT_R16G16_SINT,
    DXGI_FORMAT_R16G16B16A16_SINT,
    DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R16G16_SNORM,
    DXGI_FORMAT_R16G16B16A16_SNORM,
    DXGI_FORMAT_R16G16_UNORM,
    DXGI_FORMAT_R16G16B16A16_UNORM,
    DXGI_FORMAT_UNKNOWN,
    DXGI_FORMAT_UNKNOWN,
    DXGI_FORMAT_R16G16_FLOAT,
    DXGI_FORMAT_R16G16B16A16_FLOAT,
    DXGI_FORMAT_UNKNOWN,
};

static const char* s_pszSemantics[] = 
{
    "POSITION",
    "BLENDWEIGHT",
    "BLENDINDICES",
    "NORMAL",
    "PSIZE",
    "TEXCOORD",
    "TANGENT",
    "BINORMAL",
    "TESSFACTOR",
    "POSITIONT",
    "COLOR",
    "FOG",
    "DEPTH",
    "SAMPLE",
};

static D3D11_INPUT_ELEMENT_DESC Convert(const D3DVERTEXELEMENT9& a)
{
    D3D11_INPUT_ELEMENT_DESC r = { };
    r.SemanticName = s_pszSemantics[a.Usage];
    r.SemanticIndex = a.UsageIndex;
    r.Format = s_TypeToFormat[a.Type];
    r.InputSlot = a.Stream;
    r.AlignedByteOffset = a.Offset;
    r.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
    return r;
}

static const D3DVERTEXELEMENT9 c_End = D3DDECL_END();

VertexFormatDesc::VertexFormatDesc(const D3DVERTEXELEMENT9* pDesc)
{
    while(memcmp(pDesc, &c_End, sizeof(D3DVERTEXELEMENT9)))
        push_back(Convert(*pDesc++));
    FinishAssembly();
}

static D3DVERTEXELEMENT9 Convert(const D3D11_INPUT_ELEMENT_DESC& a, size_t byteOffset)
{
    D3DVERTEXELEMENT9 r = { };
    bool bFound = false;
    for(size_t i=0; i<ARRAYSIZE(s_pszSemantics); ++i)
        if(!strcmp(s_pszSemantics[i], a.SemanticName))
            { r.Usage = (BYTE)i; bFound = true; break; }
    _ASSERT(bFound);
    r.UsageIndex = (BYTE)a.SemanticIndex;
    bFound = false;
    for(size_t i=0; i<ARRAYSIZE(s_TypeToFormat); ++i)
        if(s_TypeToFormat[i]==a.Format)
            { r.Type = (BYTE)i; bFound = true; break; }
    _ASSERT(bFound);
    r.Stream = (WORD)a.InputSlot;
    r.Offset = (WORD)byteOffset;
    return r;
}

void VertexFormatDesc::GetDesc9(D3DVERTEXELEMENT9* pDesc) const
{
    for(size_t i=0; i<size(); ++i)
       *pDesc++ = Convert(at(i), GetByteOffset(i));
    *pDesc = c_End;
}
