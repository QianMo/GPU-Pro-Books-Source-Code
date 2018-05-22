#include "DXUT.h"
#include "Mesh/Material.h"


void Mesh::Material::saveVariable(ID3DX11EffectVariable* variable)
{
	EffectVariableSettings::iterator i = effectVariableSettings.find(variable);
	if(i != effectVariableSettings.end())
	{
		delete i->second;
		effectVariableSettings.erase(i);
	}

	D3DX11_EFFECT_TYPE_DESC typeDesc;
	variable->GetType()->GetDesc(&typeDesc);

	if(typeDesc.PackedSize > 0)
	{
		void* d = new char[typeDesc.PackedSize];
		variable->GetRawValue(d, 0, typeDesc.PackedSize);

		effectVariableSettings[variable] = d;
	}
	else
	{
		ID3D11ShaderResourceView* srv;
		variable->AsShaderResource()->GetResource(&srv);
		effectVariableSettings[variable] = srv;
	}
}

Mesh::Material::~Material()
{
	EffectVariableSettings::iterator i = effectVariableSettings.begin();
	EffectVariableSettings::iterator e = effectVariableSettings.end();
	while(i!=e)
	{
		D3DX11_EFFECT_TYPE_DESC typeDesc;
		i->first->GetType()->GetDesc(&typeDesc);
		if(typeDesc.PackedSize > 0)
			delete i->second;
		else
		{
			ID3D11ShaderResourceView* srv = ((ID3D11ShaderResourceView*)i->second);
			if(srv)
				srv->Release();
		}
		i++;
	}
}

void Mesh::Material::apply(ID3D11DeviceContext* context)
{
	EffectVariableSettings::iterator i = effectVariableSettings.begin();
	EffectVariableSettings::iterator e = effectVariableSettings.end();
	while(i!=e)
	{
		D3DX11_EFFECT_TYPE_DESC typeDesc;
		i->first->GetType()->GetDesc(&typeDesc);
		if(typeDesc.PackedSize > 0)
			i->first->SetRawValue(i->second, 0, typeDesc.PackedSize);
		else
			i->first->AsShaderResource()->SetResource((ID3D11ShaderResourceView*)i->second);
		i++;
	}
	pass->Apply(flags, context);
}

