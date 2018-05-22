#ifndef vertex_types_H
#define vertex_types_H

enum vertexElements
{
	POSITION_ELEMENT=0,
	TEXCOORDS_ELEMENT,
	NORMAL_ELEMENT,
	TANGENT_ELEMENT,
	COLOR_ELEMENT
};

enum elementFormats
{
  R32_FLOAT_EF=0,
  R32G32_FLOAT_EF,
  R32G32B32_FLOAT_EF,
  R32G32B32A32_FLOAT_EF  
};

struct VERTEX_ELEMENT_DESC
{
	vertexElements vertexElement;
	elementFormats format;
	int offset;
};

// this vertex-type is used by demo-meshes
struct GEOMETRY_VERTEX
{
	VECTOR3D position;
	VECTOR2D texCoords;  
	VECTOR3D normal;
	VECTOR4D tangent;
};

// this vertex-type is used by quads
struct QUAD_VERTEX
{
	VECTOR3D position;
	VECTOR2D texCoords;
}; 

// this vertex-type is used by fonts
struct FONT_VERTEX
{
	VECTOR3D position;
	VECTOR2D texCoords;
	COLOR color;
}; 

#endif