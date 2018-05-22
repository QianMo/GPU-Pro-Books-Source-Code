#pragma once

#include <d3d11.h>
#include "..\Effects11\Inc\d3dx11effect.h"

#define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=NULL; } }
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=NULL; } }
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=NULL; } }
#define PI 3.1415926535897932384626433832795f

#ifndef WIN64
bool ProcessorSupports3DNowProf();
#endif

extern D3D11_INPUT_ELEMENT_DESC CoreUtilsProceduralGeomertryLayout[1];

CoreResult GenerateSphere(UINT widthSegs, UINT heightSegs, float radius, CoreVector3** outVertices, WORD** outIndices, UINT* outNumVertices, UINT* outNumIndices);

UINT GetNumberOfBytesFromDXGIFormt(DXGI_FORMAT format);

interface CoreIncludeHandler
{
	// Note: Cannot handle unicode filenames because of effect file limitation; do not end the file with a \0!
	virtual CoreResult Open(D3D10_INCLUDE_TYPE IncludeType, std::wstring fileName, void** outData, UINT* outNumBytes) { return CORE_MISC_ERROR; }
	// Data = the pointer returned in Open
	virtual void Close(void *data) {}
};

CoreResult LoadEffectFromStream(Core *core, std::istream& in, CoreIncludeHandler *includeHandler, UINT hlslFlags, UINT fxFlags, ID3D10Blob** outErrors, ID3DX11Effect** outEffect);

