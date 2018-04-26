#include "DXUT.h"
#include "InputLayoutManager.h"

InputLayoutManager::InputLayoutManager()
{
}

InputLayoutManager::~InputLayoutManager()
{
	InputConfigurationList::iterator i = inputConfigurationList.begin();
	InputConfigurationList::iterator e = inputConfigurationList.end();
	while(i != e)
	{
		i->inputLayout->Release();
		i++;
	}
}

InputLayoutManager::InputConfiguration::InputConfiguration(ID3D10EffectTechnique* technique, ID3DX10Mesh* mesh)
{
	technique->GetPassByIndex(0)->GetDesc(&passDesc);
	mesh->GetVertexDescription(&elements, &nElements);
}

HRESULT InputLayoutManager::InputConfiguration::createInputLayout(ID3D10Device* device)
{
	return device->CreateInputLayout(elements, nElements, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &inputLayout);
}

bool InputLayoutManager::InputConfiguration::isCompatible(const InputLayoutManager::InputConfiguration& other) const
{
	if(passDesc.IAInputSignatureSize != other.passDesc.IAInputSignatureSize)
		return false;
	if(0 != memcmp((const void*)passDesc.pIAInputSignature, (const void*)other.passDesc.pIAInputSignature, passDesc.IAInputSignatureSize))
		return false;
	for(unsigned int i=0; i<nElements; i++)
	{
		for(unsigned int u=0; u<other.nElements; u++)
		{
			if(elements[i].SemanticIndex == other.elements[u].SemanticIndex &&
				(0 == strcmp(elements[i].SemanticName, other.elements[u].SemanticName)))
			{
				if(elements[i].AlignedByteOffset != other.elements[u].AlignedByteOffset)
					return false;
				if(elements[i].Format != other.elements[u].Format)
					return false;
				if(elements[i].InputSlot != other.elements[u].InputSlot)
					return false;
				if(elements[i].InputSlotClass != other.elements[u].InputSlotClass)
					return false;
				if(elements[i].InstanceDataStepRate != other.elements[u].InstanceDataStepRate)
					return false;
			}
		}
	}
	return true;
}

ID3D10InputLayout* InputLayoutManager::getCompatibleInputLayout(ID3D10Device* device, ID3D10EffectTechnique* technique, ID3DX10Mesh* mesh)
{
	InputConfiguration tentativeInputConfiguration(technique, mesh);
	InputConfigurationList::iterator i = inputConfigurationList.begin();
	InputConfigurationList::iterator e = inputConfigurationList.end();
	while(i != e)
	{
		if(i->isCompatible(tentativeInputConfiguration))
			return i->inputLayout;
		i++;
	}
	tentativeInputConfiguration.createInputLayout(device);
	inputConfigurationList.push_back(tentativeInputConfiguration);
	return tentativeInputConfiguration.inputLayout;
}