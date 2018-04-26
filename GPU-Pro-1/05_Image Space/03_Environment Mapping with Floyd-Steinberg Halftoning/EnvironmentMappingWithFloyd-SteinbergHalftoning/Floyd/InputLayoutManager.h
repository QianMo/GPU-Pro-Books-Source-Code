#pragma once

class InputLayoutManager
{
	class InputConfiguration
	{
		friend class InputLayoutManager;

		const D3D10_INPUT_ELEMENT_DESC* elements;
		unsigned int nElements;
		D3D10_PASS_DESC passDesc;

		ID3D10InputLayout* inputLayout;

		InputConfiguration(ID3D10EffectTechnique* technique, ID3DX10Mesh* mesh);

		/// Returns true if input signatures are identical and elements with shared semantics are also identical.
		bool isCompatible(const InputConfiguration& other) const;

		HRESULT createInputLayout(ID3D10Device* device);
	};

	typedef std::vector<InputConfiguration> InputConfigurationList;
	InputConfigurationList inputConfigurationList;

public:
	InputLayoutManager();
	~InputLayoutManager();

	ID3D10InputLayout* getCompatibleInputLayout(ID3D10Device* device, ID3D10EffectTechnique* technique, ID3DX10Mesh* mesh);
};
