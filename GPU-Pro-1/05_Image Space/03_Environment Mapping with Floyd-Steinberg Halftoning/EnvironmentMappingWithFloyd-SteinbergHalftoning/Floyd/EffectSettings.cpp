#include "DXUT.h"
#include "EffectSettings.h"

void EffectSettings::setShaderResource(ID3D10EffectShaderResourceVariable* resourceVariable, ID3D10ShaderResourceView* resourceView)
{
	shaderResourceSettings[resourceVariable] = resourceView;
}

void EffectSettings::setVector(ID3D10EffectVectorVariable* vectorVariable, const D3DXVECTOR4& value)
{
	vectorSettings[vectorVariable] = value;
}

bool EffectSettings::applyShaderResourceSetting(ID3D10EffectShaderResourceVariable* resourceVariable) const
{
	ShaderResourceSettings::const_iterator iShaderResourceSetting = shaderResourceSettings.find(resourceVariable);
	if(iShaderResourceSetting == shaderResourceSettings.end())
		return false;
	resourceVariable->SetResource(iShaderResourceSetting->second);
	return true;
}

bool EffectSettings::applyVectorSetting(ID3D10EffectVectorVariable* vectorVariable) const
{
	VectorSettings::const_iterator iVectorSetting = vectorSettings.find(vectorVariable);
	if(iVectorSetting == vectorSettings.end())
		return false;
	vectorVariable->SetFloatVector((float*)&iVectorSetting->second);
	return true;
}

