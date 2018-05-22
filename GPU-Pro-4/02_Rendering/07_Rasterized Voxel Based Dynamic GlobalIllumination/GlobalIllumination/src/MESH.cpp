#include <stdafx.h>
#include <DEMO.h>
#include <MESH.h>

bool MESH::Create(primitiveTypes primitiveType,const VERTEX_ELEMENT_DESC *vertexElementDescs,
									int numVertexElementDescs,bool dynamic,int numVertices,int numIndices)
{
	this->primitiveType = primitiveType;
	if(numVertices<1)
		return false;

	// create vertex-buffer
	vertexBuffer = DEMO::renderer->CreateVertexBuffer(vertexElementDescs,numVertexElementDescs,dynamic,numVertices);
	if(!vertexBuffer)
		return false;

	// create index-buffer
	if(numIndices>0)
	{
		indexBuffer = DEMO::renderer->CreateIndexBuffer(dynamic,numIndices);
		if(!indexBuffer)
			return false;
	}
	
  return true;
}
