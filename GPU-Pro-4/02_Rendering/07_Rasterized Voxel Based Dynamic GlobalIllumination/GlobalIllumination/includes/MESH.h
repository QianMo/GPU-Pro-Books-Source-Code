#ifndef MESH_H
#define MESH_H

class DX11_VERTEX_BUFFER;
class DX11_INDEX_BUFFER;

// MESH
//   Simple generic mesh class.
class MESH
{
public:
	MESH()
	{
		vertexBuffer = NULL;
		indexBuffer = NULL;
		primitiveType = TRIANGLES_PRIMITIVE;
	}

  bool Create(primitiveTypes primitiveType,const VERTEX_ELEMENT_DESC *vertexElementDescs,
		          int numVertexElementDescs,bool dynamic,int numVertices,int numIndices);

	DX11_VERTEX_BUFFER *vertexBuffer;
	DX11_INDEX_BUFFER *indexBuffer;
	primitiveTypes primitiveType;

};

#endif