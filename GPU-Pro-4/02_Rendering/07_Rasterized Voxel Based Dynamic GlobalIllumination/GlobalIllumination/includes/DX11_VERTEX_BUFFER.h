#ifndef DX11_VERTEX_BUFFER_H
#define DX11_VERTEX_BUFFER_H

#include <vertex_types.h>
#include <VERTEX_LIST.h>

// DX11_VERTEX_BUFFER
//   Manages a vertex buffer.
class DX11_VERTEX_BUFFER
{
public:
	DX11_VERTEX_BUFFER()
	{
		vertexSize = 0;
		dynamic = false;
		vertexBuffer = NULL;
		inputLayout = NULL;
	}

	~DX11_VERTEX_BUFFER()
	{
		Release();
	}

	void Release();

	void Clear()
	{
		vertices.Clear();
	}

	bool Create(const VERTEX_ELEMENT_DESC *vertexElementDescs,int numVertexElementDescs,bool dynamic,int maxVertexCount);

	int AddVertices(int numVertices,const float *newVertices);

	bool Update();

	void Bind() const;

	VERTEX_LIST& GetVertices() 
	{
		return vertices;
	}

	int GetVertexCount() const
	{
		return vertices.GetSize();
	}

	int GetVertexSize() const
	{
		return vertexSize;
	}

	bool IsDynamic() const
	{
		return dynamic;
	}

private:
	VERTEX_LIST vertices; // list of all vertices
	int vertexSize; // size of 1 vertex
	bool dynamic;
	int maxVertexCount; // max count of vertices that vertexBuffer can handle
	ID3D11Buffer *vertexBuffer;
	ID3D11InputLayout *inputLayout;

};

#endif