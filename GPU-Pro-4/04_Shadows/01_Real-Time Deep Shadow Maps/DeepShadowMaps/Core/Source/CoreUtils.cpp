#include "Core.h"
#include "D3DX11async.h"

D3D11_INPUT_ELEMENT_DESC CoreUtilsProceduralGeomertryLayout[] =
{
	{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
};

#ifndef WIN64
bool ProcessorSupports3DNowProf()
{
  unsigned AMD3dNow, AMD3dNowEx;

	_asm 
	{
		xor			eax, eax									
		cpuid															
		test			eax, eax
		jz				$No3dNow

		mov			eax, 80000000h							
		cpuid
		cmp			eax, 80000001h								
		jl				$No3dNow									

		mov			eax, 80000001h							
		cpuid
		mov			ecx, edx
		and			edx, 80000000h
		shr			edx, 0000001Fh
		mov			dword ptr [AMD3dNow], edx			
		mov			edx, ecx
		and			edx, 40000000h
		shr			edx, 0000001Eh
		mov			dword ptr [AMD3dNowEx], edx			

		jmp			$EndCode						
	$No3dNow:
		mov			dword ptr [AMD3dNow], 0
		mov			dword ptr [AMD3dNowEx], 0
	$EndCode:
	}

	if(AMD3dNowEx)
		return true;
	else
		return false;
}
#endif

void sincosf(float angle, float* psin, float* pcos)
{
    *psin = sinf(angle);
    *pcos = cosf(angle);
}

CoreResult GenerateSphere(UINT widthSegs, UINT heightSegs, float radius, CoreVector3** outVertices, WORD** outIndices, UINT* outNumVertices, UINT* outNumIndices)
{
	UINT i, j;

	*outNumVertices = ( heightSegs - 1 ) * widthSegs + 2;
	*outVertices = new CoreVector3[*outNumVertices];
	 if(!*outVertices)
		 return CORE_OUTOFMEM;

	 *outNumIndices = 6 * ( heightSegs - 1 ) * widthSegs;
	 *outIndices = new WORD[*outNumIndices];
	

	 if(!*outIndices)
	 {
		 delete outVertices;
		 return CORE_OUTOFMEM;
	 }

    // Sin/Cos caches
    float *sinI = new float[widthSegs], *cosI = new float[widthSegs];
    float *sinJ = new float[heightSegs], *cosJ = new float[heightSegs];

    for( i = 0; i < widthSegs; i++ )
        sincosf( 2.0f * PI * i / widthSegs, sinI + i, cosI + i );
	

    for( j = 0; j < heightSegs; j++ )
        sincosf( PI * j / heightSegs, sinJ + j, cosJ + j );

    // Generate vertices
    CoreVector3* pVertex = *outVertices;

    // +Z pole
    *pVertex = CoreVector3( 0.0f, 0.0f, radius );
    pVertex++;

    // Stacks
    for( j = 1; j < heightSegs; j++ )
        for( i = 0; i < widthSegs; i++ )
        {
            CoreVector3 norm(sinI[i] * sinJ[j], cosI[i] * sinJ[j], cosJ[j]);
			*pVertex = norm * radius;

            pVertex++;
        }

    // Z- pole
    *pVertex = CoreVector3( 0.0f, 0.0f, -radius );
    pVertex++;



    // Generate indices
    WORD* pwFace = *outIndices;
    UINT uRowA, uRowB;

    // Z+ pole
    uRowA = 0;
    uRowB = 1;

    for( i = 0; i < widthSegs - 1; i++ )
    {
        pwFace[0] = ( WORD )( uRowA );
        pwFace[1] = ( WORD )( uRowB + i + 1 );
        pwFace[2] = ( WORD )( uRowB + i );
        pwFace += 3;
    }

    pwFace[0] = ( WORD )( uRowA );
    pwFace[1] = ( WORD )( uRowB );
    pwFace[2] = ( WORD )( uRowB + i );
    pwFace += 3;

    // Interior stacks
    for( j = 1; j < heightSegs - 1; j++ )
    {
        uRowA = 1 + ( j - 1 ) * widthSegs;
        uRowB = uRowA + widthSegs;

        for( i = 0; i < widthSegs - 1; i++ )
        {
            pwFace[0] = ( WORD )( uRowA + i );
            pwFace[1] = ( WORD )( uRowA + i + 1 );
            pwFace[2] = ( WORD )( uRowB + i );
            pwFace += 3;

            pwFace[0] = ( WORD )( uRowA + i + 1 );
            pwFace[1] = ( WORD )( uRowB + i + 1 );
            pwFace[2] = ( WORD )( uRowB + i );
            pwFace += 3;
        }

        pwFace[0] = ( WORD )( uRowA + i );
        pwFace[1] = ( WORD )( uRowA );
        pwFace[2] = ( WORD )( uRowB + i );
        pwFace += 3;

        pwFace[0] = ( WORD )( uRowA );
        pwFace[1] = ( WORD )( uRowB );
        pwFace[2] = ( WORD )( uRowB + i );
        pwFace += 3;
    }

    // Z- pole
    uRowA = 1 + ( heightSegs - 2 ) * widthSegs;
    uRowB = uRowA + widthSegs;

    for( i = 0; i < widthSegs - 1; i++ )
    {
        pwFace[0] = ( WORD )( uRowA + i );
        pwFace[1] = ( WORD )( uRowA + i + 1 );
        pwFace[2] = ( WORD )( uRowB );
        pwFace += 3;
    }

    pwFace[0] = ( WORD )( uRowA + i );
    pwFace[1] = ( WORD )( uRowA );
    pwFace[2] = ( WORD )( uRowB );
    pwFace += 3;
	
	delete sinI;
	delete sinJ;
	delete cosI;
	delete cosJ;

	return CORE_OK;
}

UINT GetNumberOfBytesFromDXGIFormt(DXGI_FORMAT format)
{
	//TODO: Add more formats
	switch(format)
	{
		case DXGI_FORMAT_R8G8B8A8_UNORM:   return 4;
		case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return 4;
		case DXGI_FORMAT_R32G32B32A32_FLOAT: return 4 * sizeof(float);
		case DXGI_FORMAT_R32G32_FLOAT: return 2 * sizeof(float);
		default: return 8;
	};
}

// Should not be visible to others
class CoreD3DIncludeHandler : public ID3D10Include
{
public:
	CoreD3DIncludeHandler(CoreIncludeHandler *handler)
	{
		includeHandler = handler;
	}

	STDMETHOD(Open)(D3D10_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID *ppData, UINT *pBytes)
	{
		int fileNameLen = (int)strlen(pFileName) + 1;
		wchar_t *wstr = new wchar_t[fileNameLen];

		MultiByteToWideChar(CP_ACP, 0, pFileName, -1,  wstr, fileNameLen);
		std::wstring fileNameW = std::wstring(wstr);
		delete wstr;
		
		if(includeHandler->Open(IncludeType, fileNameW, (void **)ppData, pBytes) == CORE_OK)
			return S_OK;
		else
			return E_FAIL;
	}

	STDMETHOD(Close)(LPCVOID pData)
	{
		includeHandler->Close((void *)pData);
		return S_OK;
	}
private:
	CoreIncludeHandler *includeHandler;
};

CoreResult LoadEffectFromStream(Core *core, std::istream& in, CoreIncludeHandler *includeHandler, UINT hlslFlags, UINT fxFlags, ID3D10Blob** outErrors, ID3DX11Effect** outEffect)
{
	if(!in.good())
	{
		CoreLog::Information(L"in is not good, skipping!");
		return CORE_INVALIDARGS;
	}

	std::string text;
	std::string line;
	while (!in.eof())
    {
      getline (in, line);
      text += line + "\n";
    }

	ID3D10Include *d3dIncludeHandler = NULL;
	if(includeHandler)
		d3dIncludeHandler = new CoreD3DIncludeHandler(includeHandler);

	ID3D10Blob* compiled;
	HRESULT result = D3DX11CompileFromMemory((LPCSTR)text.c_str(), text.length(), NULL, NULL, d3dIncludeHandler, NULL, "fx_5_0", hlslFlags, fxFlags, NULL, &compiled, outErrors, NULL);


	if(FAILED(result))
	{
		SAFE_DELETE(d3dIncludeHandler);
		CoreLog::Information(L"Error compiling the effect, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}

	result = D3DX11CreateEffectFromMemory(compiled->GetBufferPointer(), compiled->GetBufferSize(), fxFlags, core->GetDevice(), outEffect);

	compiled->Release();
	SAFE_DELETE(d3dIncludeHandler);

	if(FAILED(result))
	{
		CoreLog::Information(L"Error creating the effect, HRESULT = %x", result);
		return CORE_MISC_ERROR;
	}
	return CORE_OK;
}
