#ifndef DX11_STRUCTURED_BUFFER_H
#define DX11_STRUCTURED_BUFFER_H

#include <render_states.h>

// DX11_STRUCTURED_BUFFER
//   Manages a DirectX 11 structured buffer.
class DX11_STRUCTURED_BUFFER
{
public:
	DX11_STRUCTURED_BUFFER()
	{
		bindingPoint = 0;
		elementCount = 0;
		elementSize = 0;
	  structuredBuffer = NULL;
		unorderedAccessView = NULL;
    shaderResourceView = NULL;
	}

	~DX11_STRUCTURED_BUFFER()
	{
		Release();
	}

	void Release();

	bool Create(int bindingPoint,int elementCount,int elementSize);

	void Bind(shaderTypes shaderType=VERTEX_SHADER) const;

	ID3D11UnorderedAccessView* GetUnorderdAccessView() const;

	int GetBindingPoint() const
	{
		return bindingPoint;
	}

	int GetElementCount() const
	{
		return elementCount;
	}

	int GetElementSize() const
	{
		return elementCount;
	}

private:
	int bindingPoint; // shader binding point
	int elementCount; // number of structured elements in buffer
	int elementSize; // size of 1 structured element in bytes
	ID3D11Buffer *structuredBuffer;
	ID3D11UnorderedAccessView *unorderedAccessView;
	ID3D11ShaderResourceView *shaderResourceView;

};

#endif