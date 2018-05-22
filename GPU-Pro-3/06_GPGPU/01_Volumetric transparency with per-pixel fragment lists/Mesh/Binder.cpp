#include "DXUT.h"
#include "Mesh/Binder.h"

Mesh::Binder::Binder()
{
}

Mesh::Binder::~Binder()
{
	InputConfigurationList::iterator i = inputConfigurationList.begin();
	InputConfigurationList::iterator e = inputConfigurationList.end();
	while(i != e)
	{
		(*i)->inputLayout->Release();
		delete *i;
		i++;
	}
}

Mesh::Binder::InputConfiguration::InputConfiguration(const D3DX11_PASS_DESC& passDesc, Mesh::Geometry::P geometry)
	:passDesc(passDesc)
{
	const D3D11_INPUT_ELEMENT_DESC* delements;
	unsigned int dnElements;
	geometry->getElements(delements, dnElements);

	nElements = dnElements;

	D3D11_INPUT_ELEMENT_DESC* elements = new D3D11_INPUT_ELEMENT_DESC[nElements];
	memcpy(elements, delements, nElements * sizeof(D3D11_INPUT_ELEMENT_DESC));

	for(int i=0; i<nElements; i++)
	{
		char* semanticName = new char[strlen(delements[i].SemanticName)+1];
		strcpy(semanticName, delements[i].SemanticName);
		elements[i].SemanticName = semanticName;
	}
	this->elements = elements;
}

Mesh::Binder::InputConfiguration::~InputConfiguration()
{
	for(int i=0; i<nElements; i++)
		if(elements[i].SemanticName)
			delete elements[i].SemanticName;
	delete [] elements;
}

HRESULT Mesh::Binder::InputConfiguration::createInputLayout(ID3D11Device* device)
{
	return device->CreateInputLayout(elements, nElements, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &inputLayout);
}

bool Mesh::Binder::InputConfiguration::isCompatible(const Binder::InputConfiguration& other) const
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

ID3D11InputLayout* Mesh::Binder::getCompatibleInputLayout(ID3D11Device* device, const D3DX11_PASS_DESC& passDesc, Mesh::Geometry::P geometry)
{
	InputConfiguration* tentativeInputConfiguration = new InputConfiguration(passDesc, geometry);
	InputConfigurationList::iterator i = inputConfigurationList.begin();
	InputConfigurationList::iterator e = inputConfigurationList.end();
#ifdef DEBUG
	if(tentativeInputConfiguration->createInputLayout(device) != S_OK)
	{
		MessageBox(NULL, L"Failed to create an input layout. Invalid mesh-technique combination.", L"Error!", 0);
		DXUTShutdown(-1);
	}
	while(i != e)
	{
		if((*i)->isCompatible(*tentativeInputConfiguration))
		{
			tentativeInputConfiguration->inputLayout->Release();
			delete tentativeInputConfiguration;
			return (*i)->inputLayout;
		}
		i++;
	}
#else
	while(i != e)
	{
		if((*i)->isCompatible(*tentativeInputConfiguration))
			return (*i)->inputLayout;
		i++;
	}
	tentativeInputConfiguration->createInputLayout(device);
#endif
	inputConfigurationList.push_back(tentativeInputConfiguration);
	return tentativeInputConfiguration->inputLayout;
}

Mesh::Bound::P Mesh::Binder::bind(ID3D11Device* device, const D3DX11_PASS_DESC& passDesc, Mesh::Geometry::P geometry)
{
	ID3D11InputLayout* inputLayout = getCompatibleInputLayout(device, passDesc, geometry);
	return Mesh::Bound::make(geometry, inputLayout);
}
