/*
****************************************************************************
 * GeometryLoader class - loader for an own (raw) mesh format (.dgb)
 *
 * @author: László Szécsi
 * Used with permission.
****************************************************************************
*/

#pragma once

/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////

class GeometryLoader
{
public:
	GeometryLoader(void);
public:
	~GeometryLoader(void);

	static HRESULT LoadMeshFromFile(const wchar_t* dgbFileName, ID3D10Device* device, ID3DX10Mesh** mesh);
	static HRESULT CreateMeshFromMemory(ID3D10Device* device, BYTE* data, unsigned int nBytes, ID3DX10Mesh** mesh);
};
