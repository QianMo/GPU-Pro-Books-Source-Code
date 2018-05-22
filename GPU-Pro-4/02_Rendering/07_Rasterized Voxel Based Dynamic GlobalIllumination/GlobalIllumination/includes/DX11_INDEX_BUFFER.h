#ifndef DX11_INDEX_BUFFER_H
#define DX11_INDEX_BUFFER_H

#include <LIST.h>

// DX11_INDEX_BUFFER
//   Manages an index buffer.  
class DX11_INDEX_BUFFER
{
public:
	DX11_INDEX_BUFFER()
	{
		dynamic = false;
		maxIndexCount = 0;
		indexBuffer = NULL;
	}

	~DX11_INDEX_BUFFER()
	{
		Release();
	}

	void Release();

	bool Create(bool dynamic,int maxIndexCount);

	void Clear()
	{
		indices.Clear();
	}

	int AddIndices(int numIndices,const int *indices);

	bool Update();

	void Bind() const;

	int GetIndexCount() const
	{
		return indices.GetSize();
	}

	bool IsDynamic() const
	{
		return dynamic;
	}

private:
	LIST<int> indices; // list of all indices
	bool dynamic; 
	int maxIndexCount; // max count of indices that indexBuffer can handle
	ID3D11Buffer *indexBuffer;

};

#endif