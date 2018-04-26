#include <windows.h>
#include <d3d11.h>
#include <d3dx11.h>

#include <D3DX11tex.h>
#include <D3DX11.h>
#include <D3DX11core.h>
#include <D3DX11async.h>
#include <D3DCompiler.h>
#include <strsafe.h>
#include <iostream>

#define ITEM_BUFFER_DIMENSION_XY 512
#define ITEM_BUFFER_TILING_XY 2
#define ITEM_BUFFER_ITEM_COUNT 256


UINT Ceillog2( const UINT v ) {

	if( v == 0 )
		return (UINT)-1;

	UINT tmp = v;
	UINT clog=0;
	while( tmp > 1 ) {
		clog++;
		tmp = tmp >> 1;
	}

	return ( ( 1 << clog ) == v ? clog : clog+1 );
}

#define SAFE_RELEASE(x) if(x) { x->Release(); x = NULL; }

ID3D11Device*				gDevice11 = NULL;
ID3D11DeviceContext*		gDeviceContext11 = NULL;
ID3D11Texture2D*			gItemBuffer = NULL;
ID3D11ShaderResourceView*	gItemBufferSR = NULL;
ID3D11RenderTargetView*		gItemBufferRT =NULL;
ID3D11Texture2D*			gItemHistogram = NULL;
ID3D11UnorderedAccessView*	gItemHistogramUAV = NULL;
ID3D11ComputeShader*		gCSBuildItemHistogram = NULL;
ID3D11Texture2D*			gReadBackTexture = NULL;
ID3D11Texture2D*			gHierarchyTexture = NULL;

HRESULT CompileShaderFromFile( WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut )
{
    HRESULT hr = S_OK;

    // find the file
    WCHAR str[MAX_PATH];
	StringCchPrintf( str, MAX_PATH, TEXT("%s"), szFileName );

    // open the file
    HANDLE hFile = CreateFile( str, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
        FILE_FLAG_SEQUENTIAL_SCAN, NULL );
    if( INVALID_HANDLE_VALUE == hFile )
        return E_FAIL;

    // Get the file size
    LARGE_INTEGER FileSize;
    GetFileSizeEx( hFile, &FileSize );

    // create enough space for the file data
    BYTE* pFileData = new BYTE[ FileSize.LowPart ];
    if( !pFileData )
        return E_OUTOFMEMORY;

    // read the data in
    DWORD BytesRead;
    if( !ReadFile( hFile, pFileData, FileSize.LowPart, &BytesRead, NULL ) )
        return E_FAIL; 

    CloseHandle( hFile );

    // Compile the shader
    char pFilePathName[MAX_PATH];        
    WideCharToMultiByte(CP_ACP, 0, str, -1, pFilePathName, MAX_PATH, NULL, NULL);
    ID3DBlob* pErrorBlob;
    hr = D3DCompile( pFileData, FileSize.LowPart, pFilePathName, NULL, NULL, szEntryPoint, szShaderModel, D3D10_SHADER_ENABLE_STRICTNESS, 0, ppBlobOut, &pErrorBlob );

    delete []pFileData;

    if( FAILED(hr) )
    {
        OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
        SAFE_RELEASE( pErrorBlob );
        return hr;
    }
    SAFE_RELEASE( pErrorBlob );

    return S_OK;
}


HRESULT CreateDevice11( ID3D11Device** d3dDevice, ID3D11DeviceContext** d3dContext ) {
	D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
	return D3D11CreateDevice(
		NULL,						// adapter
		D3D_DRIVER_TYPE_REFERENCE,	// driver type
		NULL,						// handle to software rasterizer (WARP)
		0,
		&featureLevel,
		1,
		D3D11_SDK_VERSION,
		d3dDevice,
		NULL,
		d3dContext );
}

void FreeResources() {
	SAFE_RELEASE( gItemBuffer );
	SAFE_RELEASE( gItemBufferSR );
	SAFE_RELEASE( gItemBufferRT );
	SAFE_RELEASE( gItemHistogram );
	SAFE_RELEASE( gItemHistogramUAV );
	SAFE_RELEASE( gCSBuildItemHistogram );
	SAFE_RELEASE( gHierarchyTexture );
}

void DestroyDevice11( ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dContext ) {
	SAFE_RELEASE( d3dDevice );
	SAFE_RELEASE( d3dContext );
}


HRESULT CreateItemBuffer( ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dContext ) {

	HRESULT hr = E_FAIL;

	D3D11_TEXTURE2D_DESC td;
	td.ArraySize		= 1;
	td.Width			= ITEM_BUFFER_DIMENSION_XY;
	td.Height			= ITEM_BUFFER_DIMENSION_XY;
	td.Format			= DXGI_FORMAT_R32_UINT;
	td.BindFlags		= D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	td.CPUAccessFlags	= 0;
	td.MipLevels		= 1;
	td.MiscFlags		= D3D11_RESOURCE_MISC_GENERATE_MIPS;
	td.SampleDesc.Count = 1;
	td.SampleDesc.Quality=0;
	td.Usage			= D3D11_USAGE_DEFAULT;

	

	hr = d3dDevice->CreateTexture2D( &td, NULL, &gItemBuffer );
	if( FAILED( hr ) )
		return hr;

	gItemBuffer->GetDesc( &td );
	D3D11_SHADER_RESOURCE_VIEW_DESC sd;
	sd.ViewDimension	= D3D11_SRV_DIMENSION_TEXTURE2D;
	sd.Texture2D.MipLevels= td.MipLevels;
	sd.Texture2D.MostDetailedMip=0;
	sd.Format	= td.Format;

	hr = d3dDevice->CreateShaderResourceView( gItemBuffer, &sd, &gItemBufferSR );
	if( FAILED( hr ) )
		return hr;

	// we fill the item buffer with random data instead of
	// rendering anything into it.
	UINT* itemBufferData = new UINT[ ITEM_BUFFER_DIMENSION_XY * ITEM_BUFFER_DIMENSION_XY ];
	for( UINT i = 0; i < ITEM_BUFFER_DIMENSION_XY * ITEM_BUFFER_DIMENSION_XY; ++i ) {
		UINT ID = ( rand() * ITEM_BUFFER_ITEM_COUNT ) / (RAND_MAX+1);
				
		itemBufferData[ i ] = ID;
	}


	D3D11_BOX box;
	box.back	= 1;
	box.front	= 0;
	box.left	= 0;
	box.right	= ITEM_BUFFER_DIMENSION_XY;
	box.top		= 0;
	box.bottom	= ITEM_BUFFER_DIMENSION_XY;

	d3dContext->UpdateSubresource( gItemBuffer, D3D11CalcSubresource( 0, 0, 0 ), &box, itemBufferData, ITEM_BUFFER_DIMENSION_XY, 0 );
	delete[] itemBufferData;



	D3D11_RENDER_TARGET_VIEW_DESC rd;
	rd.ViewDimension	= D3D11_RTV_DIMENSION_TEXTURE2D;
	rd.Format			= td.Format;
	rd.Texture2D.MipSlice=0;
	hr = d3dDevice->CreateRenderTargetView( gItemBuffer, &rd, &gItemBufferRT );
	if( FAILED( hr ) )
		return hr;

	// item histogram texture
	ZeroMemory( &td, sizeof( td ) );
	td.ArraySize		= ITEM_BUFFER_ITEM_COUNT;
	td.Width			= ITEM_BUFFER_TILING_XY;
	td.Height			= ITEM_BUFFER_TILING_XY;
	td.Format			= DXGI_FORMAT_R32_UINT;
	td.BindFlags		= D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
	td.CPUAccessFlags	= 0;
	td.MipLevels		= 0;
	td.MiscFlags		= 0;
	td.SampleDesc.Count = 1;
	td.SampleDesc.Quality=0;
	td.Usage			= D3D11_USAGE_DEFAULT;

	hr = d3dDevice->CreateTexture2D( &td, NULL, &gItemHistogram );
	if( FAILED( hr ) )
		return hr;

	D3D11_UNORDERED_ACCESS_VIEW_DESC uad;
	uad.ViewDimension					= D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
	uad.Format							= td.Format;
	uad.Texture2DArray.ArraySize		= ITEM_BUFFER_ITEM_COUNT;
	uad.Texture2DArray.FirstArraySlice	= 0;
	uad.Texture2DArray.MipSlice			= 0;

	hr = d3dDevice->CreateUnorderedAccessView( gItemHistogram, &uad, &gItemHistogramUAV );
	if( FAILED( hr ) )
		return hr;


	td.BindFlags		= 0;
	td.Format			= DXGI_FORMAT_R32_UINT;
	td.CPUAccessFlags	= D3D11_CPU_ACCESS_READ;
	td.Usage			= D3D11_USAGE_STAGING;
	td.MiscFlags		= 0;
	hr = d3dDevice->CreateTexture2D( &td, NULL, &gReadBackTexture );
	if( FAILED( hr ) )
		return hr;



	ZeroMemory( &td, sizeof( td ) );
	td.ArraySize		= ITEM_BUFFER_ITEM_COUNT;
	td.Width			= ITEM_BUFFER_TILING_XY;
	td.Height			= ITEM_BUFFER_TILING_XY;
	td.Format			= DXGI_FORMAT_R32_FLOAT;
	td.BindFlags		= D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	td.CPUAccessFlags	= 0;
	td.MipLevels		= 1;
	td.MiscFlags		= D3D11_RESOURCE_MISC_GENERATE_MIPS;
	td.MiscFlags		= 0;
	td.SampleDesc.Count = 1;
	td.SampleDesc.Quality=0;
	td.Usage			= D3D11_USAGE_DEFAULT;

	hr = d3dDevice->CreateTexture2D( &td, NULL, &gHierarchyTexture );
	if( FAILED( hr ) )
		return hr;



	return hr;
}

void DestroyItemBuffer() {
	SAFE_RELEASE( gItemBuffer );
	SAFE_RELEASE( gItemBufferSR );
	SAFE_RELEASE( gItemHistogram );
	SAFE_RELEASE( gItemHistogramUAV );
}

HRESULT LoadComputeShader( ID3D11Device* d3dDevice ) {
	HRESULT hr = E_FAIL;
	ID3DBlob* blob = NULL;
	hr = CompileShaderFromFile( TEXT("..\\effects\\CSItemHistogram.csh"), "CSMain", "cs_5_0", &blob );
	if( FAILED( hr ) ) {
		MessageBox( NULL, TEXT("Failed to compile or load shader. Check if working directory is correct"), TEXT("ERROR"), MB_ICONERROR | MB_OK );
		exit(-1);
	}
	hr = d3dDevice->CreateComputeShader( blob->GetBufferPointer(), blob->GetBufferSize(), NULL, &gCSBuildItemHistogram );
	SAFE_RELEASE( blob );

	return hr;
}
int main( int argc, char** argv ) {

	
	HRESULT hr;
	std::cout << "Creating Device and Context: ";
	hr = CreateDevice11( &gDevice11, &gDeviceContext11 );
	if( SUCCEEDED( hr ) )
		std::cout << "OK" << std::endl;
	else {
		std::cout << "FAILED" << std::endl;
		return -1;
	}


	// Create and initialize the item buffer
	// For the sake of demonstrating the algorithm with
	// the D3D11 reference device, we do not render anything into
	// hte item buffer, but initialize it with random identifiers.
	std::cout << "Creating item buffer and histogram texture: ";
	hr = CreateItemBuffer( gDevice11, gDeviceContext11 );
	if( SUCCEEDED( hr ) )
		std::cout << "OK" << std::endl;
	else {
		std::cout << "FAILED" << std::endl;
		return -1;
	}
	
	// Load and compile the shader
	std::cout << "Loading and compling compute shader: ";
	hr = LoadComputeShader( gDevice11 );
	if( SUCCEEDED( hr ) )
			std::cout << "OK" << std::endl;
	else {
		std::cout << "FAILED" << std::endl;
		return -1;
	}


	// Clear histogram texture
	UINT clearValues[] = { 0, 0, 0, 0 };
	gDeviceContext11->ClearUnorderedAccessViewUint( gItemHistogramUAV, clearValues );

	// Bind item buffer as shader resource to CS
	gDeviceContext11->CSSetShaderResources( 0, 1, &gItemBufferSR );

	// Bind histogram texture as unordered access view to CS
	UINT iCnt = 1;
	ID3D11UnorderedAccessView* arr[] = { gItemHistogramUAV };
	gDeviceContext11->CSSetUnorderedAccessViews( 0, 1, &gItemHistogramUAV, &iCnt );

	// Set and Launch CS
	std::cout << "Executing compute shader: ";
	gDeviceContext11->CSSetShader( gCSBuildItemHistogram, NULL, 0 );
	gDeviceContext11->Dispatch( ITEM_BUFFER_DIMENSION_XY/32, ITEM_BUFFER_DIMENSION_XY/32, 1 );
	std::cout << "DONE" << std::endl;
	

	ID3D11UnorderedAccessView* tmp2[] = { NULL, NULL };
	gDeviceContext11->CSSetUnorderedAccessViews( 0, 2, tmp2, &iCnt );

	// Copy result into a CPU mappable texture
	gDeviceContext11->CopyResource( gReadBackTexture, gItemHistogram );

	std::cout << "Writing result to result.txt ";
	// Write result into a TXT file.
	const UINT mipLevels	= Ceillog2( ITEM_BUFFER_TILING_XY ) + 1;
	HANDLE hFile = CreateFile( TEXT("result.txt"),
		GENERIC_WRITE, 0,
		NULL,
		CREATE_ALWAYS,
		FILE_FLAG_SEQUENTIAL_SCAN,
		NULL );

	for( UINT i = 0; i < ITEM_BUFFER_ITEM_COUNT; ++i ) {

		char str[ 256 ];
		size_t strLen;
		DWORD written = 0;
		ZeroMemory( str, sizeof( char ) * 256 );
		StringCchPrintfA( str, 256, "------------------------------------------------------------\r\n");
		StringCchLengthA( str, 256, &strLen );
		WriteFile( hFile, str, (DWORD)strLen, &written, NULL );
		
		StringCchPrintfA( str, 256, "Item: %i\r\n", i );
		StringCchLengthA( str, 256, &strLen );
		WriteFile( hFile, str, (DWORD)strLen, &written, NULL );

		StringCchPrintfA( str, 256, "------------------------------------------------------------\r\n");
		StringCchLengthA( str, 256, &strLen );
		WriteFile( hFile, str, (DWORD)strLen, &written, NULL );



		D3D11_MAPPED_SUBRESOURCE ms;
		gDeviceContext11->Map( gReadBackTexture, D3D11CalcSubresource( 0, i, mipLevels ), D3D11_MAP_READ, 0, &ms );
		UINT* tmp = (UINT*)ms.pData;

		for( UINT y = 0; y < ITEM_BUFFER_TILING_XY; ++ y ) {
			for( UINT x = 0; x < ITEM_BUFFER_TILING_XY; ++x ) {
				StringCchPrintfA( str, 256, "tile(%i,%i) = %i\r\n", x, y, tmp[ y * ITEM_BUFFER_TILING_XY + x ] );
				StringCchLengthA( str, 256, &strLen );
				WriteFile( hFile, str, (DWORD)strLen, &written, NULL );
			}
		}
		
		gDeviceContext11->Unmap( gReadBackTexture, D3D11CalcSubresource( 0, i, mipLevels ) );
	}

	CloseHandle( hFile );
	std::cout << "DONE" << std::endl;

	// cleanup
	std::cout << "Freeing resources and device ";
	FreeResources();
	DestroyDevice11( gDevice11, gDeviceContext11 );
	std::cout << "DONE" << std::endl;
	return 0;
}