#pragma once

#include "Directory.h"

class EffectSettings
{
protected:
	typedef std::map<ID3D10EffectShaderResourceVariable*, ID3D10ShaderResourceView*, ComparePointers>		ShaderResourceSettings;
	typedef std::map<ID3D10EffectVectorVariable*, D3DXVECTOR4, ComparePointers>				VectorSettings;

	/// Name-value pairs for texture variables in the effect file.
	ShaderResourceSettings shaderResourceSettings;
	/// Name-value pairs for float4 variables in the effect file.
	VectorSettings vectorSettings;

public:

	/// Adds a name-value pair to the texture effect variables directory.
	void setShaderResource(ID3D10EffectShaderResourceVariable* resourceVariable, ID3D10ShaderResourceView* resourceView);
	/// Adds a name-value pair to the vector effect variables directory.
	void setVector(ID3D10EffectVectorVariable* vectorVariable, const D3DXVECTOR4& value);


	bool applyShaderResourceSetting(ID3D10EffectShaderResourceVariable* resourceVariable) const;
	bool applyVectorSetting(ID3D10EffectVectorVariable* vectorVariable) const;

};
