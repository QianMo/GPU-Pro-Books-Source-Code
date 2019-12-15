#ifndef DX12_ROOT_SIGNATURE_H
#define DX12_ROOT_SIGNATURE_H

#include <render_states.h>

BITFLAGS_ENUM(UINT, rootSignatureFlags)
{
  NONE_ROOT_SIGNATURE_FLAG                = D3D12_ROOT_SIGNATURE_FLAG_NONE,
  ALLOW_INPUT_LAYOUT_ROOT_SIGNATURE_FLAG  = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
  DENY_VS_ACCESS_ROOT_SIGNATURE_FLAG      = D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS,
  DENY_HS_ACCESS_ROOT_SIGNATURE_FLAG      = D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS,
  DENY_DS_ACCESS_ROOT_SIGNATURE_FLAG      = D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS,
  DENY_GS_ACCESS_ROOT_SIGNATURE_FLAG      = D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS,
  DENY_PS_ACCESS_ROOT_SIGNATURE_FLAG      = D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS,
  ALLOW_STREAM_OUTPUT_ROOT_SIGNATURE_FLAG = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT
};

enum shaderVisibilities
{
  ALL_SHADER_VIS = D3D12_SHADER_VISIBILITY_ALL,
  VS_SHADER_VIS  = D3D12_SHADER_VISIBILITY_VERTEX,
  HS_SHADER_VIS  = D3D12_SHADER_VISIBILITY_HULL,
  DS_SHADER_VIS  = D3D12_SHADER_VISIBILITY_DOMAIN,
  GS_SHADER_VIS  = D3D12_SHADER_VISIBILITY_GEOMETRY,
  PS_SHADER_VIS  = D3D12_SHADER_VISIBILITY_PIXEL,
  CS_SHADER_VIS  = ALL_SHADER_VIS
};

enum rootDescTableRangeTypes
{
  SRV_ROOT_DESC_TABLE_RANGE     = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
  UAV_ROOT_DESC_TABLE_RANGE     = D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
  CBV_ROOT_DESC_TABLE_RANGE     = D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
  SAMPLER_ROOT_DESC_TABLE_RANGE = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER
};

struct RootDescTableRange
{
  bool operator== (const RootDescTableRange &desc) const
  {
    return ((rangeType == desc.rangeType) && (numDescs == desc.numDescs) && (baseShaderReg == desc.baseShaderReg) &&
            (regSpace == desc.regSpace) && (descTableOffset == desc.descTableOffset));
  }

  bool operator!= (const RootDescTableRange &desc) const
  {
    return !((*this) == desc);
  }

  rootDescTableRangeTypes rangeType;
  UINT numDescs;
  UINT baseShaderReg;
  UINT regSpace;
  UINT descTableOffset;
};

struct RootDescTableDesc
{
  bool operator== (const RootDescTableDesc &desc) const
  {
    if(numRanges != desc.numRanges)
      return false;
    for(UINT i=0; i<numRanges; i++)
    {
      if(ranges[i] != desc.ranges[i])
        return false;
    }
    return true;
  }

  bool operator!= (const RootDescTableDesc &desc) const
  {
    return !((*this) == desc);
  }

  RootDescTableRange ranges[MAX_NUM_ROOT_DESC_TABLE_RANGES];
  UINT numRanges;
};

struct RootConstDesc
{
  bool operator== (const RootConstDesc &desc) const
  {
    return ((shaderReg == desc.shaderReg) && (regSpace == desc.regSpace) && (numConsts == desc.numConsts));
  }

  bool operator!= (const RootConstDesc &desc) const
  {
    return !((*this) == desc);
  }

  UINT shaderReg;
  UINT regSpace;
  UINT numConsts;
};

struct RootDesc
{
  bool operator== (const RootDesc &desc) const
  {
    return ((shaderReg == desc.shaderReg) && (regSpace == desc.regSpace));
  }

  bool operator!= (const RootDesc &desc) const
  {
    return !((*this) == desc);
  }

  UINT shaderReg;
  UINT regSpace;
};

struct RootParamDesc
{
  bool operator== (const RootParamDesc &desc) const
  {
    if((rootParamType != desc.rootParamType) || (shaderVisibility != desc.shaderVisibility))
    {
      return false;
    }
    switch(rootParamType)
    {
    case DESC_TABLE_ROOT_PARAM:
      if(rootDescTableDesc != desc.rootDescTableDesc)
        return false;
      break;
    case CONST_ROOT_PARAM:
      if(rootConstDesc != desc.rootConstDesc)
        return false;
      break;
    case CBV_ROOT_PARAM:
    case SRV_ROOT_PARAM:
    case UAV_ROOT_PARAM:
      if(rootDesc != desc.rootDesc)
        return false;
      break;
    }
    return true;
  }

  bool operator!= (const RootParamDesc &desc) const
  {
    return !((*this) == desc);
  }

  rootParamTypes rootParamType;
  shaderVisibilities shaderVisibility;
  union
  {
    RootDescTableDesc rootDescTableDesc;
    RootConstDesc rootConstDesc;
    RootDesc rootDesc;
  };
};

struct SamplerDesc
{
  SamplerDesc() :
    filter(MIN_MAG_LINEAR_MIP_POINT_FILTER),
    adressU(CLAMP_TEX_ADDRESS),
    adressV(CLAMP_TEX_ADDRESS),
    adressW(CLAMP_TEX_ADDRESS),
    lodBias(0.0f),
    maxAnisotropy(2),
    compareFunc(LEQUAL_CMP_FUNC),
    minLOD(0.0f),
    maxLOD(FLT_MAX)
  {
  }

  bool operator== (const SamplerDesc &desc) const
  {
    if(filter != desc.filter)
      return false;
    if(adressU != desc.adressU)
      return false;
    if(adressV != desc.adressV)
      return false;
    if(adressW != desc.adressW)
      return false;
    if(maxAnisotropy != desc.maxAnisotropy)
      return false;
    if(compareFunc != desc.compareFunc)
      return false;
    if(!IS_EQUAL(lodBias, desc.lodBias))
      return false;
    if(!IS_EQUAL(minLOD, desc.minLOD))
      return false;
    if(!IS_EQUAL(maxLOD, desc.maxLOD))
      return false;
    if(borderColor != desc.borderColor)
      return false;
    return true;
  }

  bool operator!= (const SamplerDesc &desc) const
  {
    return !((*this) == desc);
  }

  filterModes filter;
  texAddressModes adressU;
  texAddressModes adressV;
  texAddressModes adressW;
  float lodBias;
  UINT maxAnisotropy;
  comparisonFuncs compareFunc;
  Color borderColor;
  float minLOD;
  float maxLOD;
};

struct StaticSamplerDesc
{
  StaticSamplerDesc() :
    filter(MIN_MAG_LINEAR_MIP_POINT_FILTER),
    adressU(CLAMP_TEX_ADDRESS),
    adressV(CLAMP_TEX_ADDRESS),
    adressW(CLAMP_TEX_ADDRESS),
    lodBias(0.0f),
    maxAnisotropy(2),
    compareFunc(LEQUAL_CMP_FUNC),
    borderColor(TRANSPARENT_BLACK_STATIC_BORDER_COLOR),
    minLOD(0.0f),
    maxLOD(FLT_MAX),
    shaderReg(0),
    regSpace(0),
    shaderVisibility(ALL_SHADER_VIS)
  {
  }

  bool operator== (const StaticSamplerDesc &desc) const
  {
    if(filter != desc.filter)
      return false;
    if(adressU != desc.adressU)
      return false;
    if(adressV != desc.adressV)
      return false;
    if(adressW != desc.adressW)
      return false;
    if(maxAnisotropy != desc.maxAnisotropy)
      return false;
    if(compareFunc != desc.compareFunc)
      return false;
    if(borderColor != desc.borderColor)
      return false;
    if(shaderReg != desc.shaderReg)
      return false;
    if(regSpace != desc.regSpace)
      return false;
    if(shaderVisibility != desc.shaderVisibility)
      return false;
    if(!IS_EQUAL(lodBias, desc.lodBias))
      return false;
    if(!IS_EQUAL(minLOD, desc.minLOD))
      return false;
    if(!IS_EQUAL(maxLOD, desc.maxLOD))
      return false;
    return true;
  }

  bool operator!= (const StaticSamplerDesc &desc) const
  {
    return !((*this) == desc);
  }

  filterModes filter;
  texAddressModes adressU;
  texAddressModes adressV;
  texAddressModes adressW;
  float lodBias;
  UINT maxAnisotropy;
  comparisonFuncs compareFunc;
  staticBorderColors borderColor;
  float minLOD;
  float maxLOD;
  UINT shaderReg;
  UINT regSpace;
  shaderVisibilities shaderVisibility;
};

struct RootSignatureDesc
{
  RootSignatureDesc() :
    numRootParamDescs(0),
    numSamplerDescs(0),
    flags(NONE_ROOT_SIGNATURE_FLAG)
  {
    memset(rootParamDescs, 0, sizeof(RootParamDesc) * MAX_NUM_ROOT_PARAMS);
  }

  bool operator== (const RootSignatureDesc &desc) const
  {
    if((numRootParamDescs != desc.numRootParamDescs) || (numSamplerDescs != desc.numSamplerDescs) || (flags != desc.flags))
    {
      return false;
    }
    for(UINT i=0; i<numRootParamDescs; i++)
    {
      if(rootParamDescs[i] != desc.rootParamDescs[i])
        return false;
    }
    for(UINT i=0; i<numSamplerDescs; i++)
    {
      if(samplerDescs[i] != desc.samplerDescs[i])
        return false;
    }
    return true;
  }

  bool operator!= (const RootSignatureDesc &desc) const
  {
    return !((*this) == desc);
  }

  RootParamDesc rootParamDescs[MAX_NUM_ROOT_PARAMS];
  UINT numRootParamDescs;
  StaticSamplerDesc samplerDescs[MAX_NUM_STATIC_SAMPLERS];
  UINT numSamplerDescs;
  rootSignatureFlags flags;
};

// DX12_RootSignature
//
class DX12_RootSignature
{
public:
  bool Create(const RootSignatureDesc &desc, const char *name);

  ID3D12RootSignature* GetRootSignature() const
  {
    return rootSignature.Get();
  }

  const RootSignatureDesc& GetDesc() const
  {
    return desc;
  }

private:
  ComPtr<ID3D12RootSignature> rootSignature;
  RootSignatureDesc desc;

};

#endif
