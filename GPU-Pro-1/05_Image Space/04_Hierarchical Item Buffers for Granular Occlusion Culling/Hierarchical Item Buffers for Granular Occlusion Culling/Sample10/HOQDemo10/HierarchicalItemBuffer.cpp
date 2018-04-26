#include "HOQDemo10.h"
#include "HierarchicalItemBuffer.h"

HierarchicalItemBuffer::HierarchicalItemBuffer() 
{
	bucketCount = 0;
	itemCount	= 0;
	width		= 0;
	height		= 0;
	
	levelCount	= 0;

	scatterTexture			= NULL;
	scatterRenderTarget		= NULL;
	scatterShaderResource	= NULL;

	inputTexture			= NULL;
	itemInputRenderTarget	= NULL;
	itemInputResource		= NULL;
	pointBuffer				= NULL;

	scatterPass3			= NULL;
	scatterLayout3			= NULL;

	effect					= NULL;
}

HRESULT HierarchicalItemBuffer::CreateRenderTargetAndResource(
	ID3D10Device* d3dDevice, 
	size_t width, 
	size_t height, 
	DXGI_FORMAT format,
	ID3D10RenderTargetView** rtvOut,
	ID3D10ShaderResourceView** srvOut,
	UINT miscFlags ) const
{
	HRESULT hr;

	D3D10_TEXTURE2D_DESC td;
	td.ArraySize		= 1;
	td.BindFlags		= D3D10_BIND_SHADER_RESOURCE | D3D10_BIND_RENDER_TARGET;
	td.CPUAccessFlags	= 0;
	td.Format			= format;
	td.Height			= (UINT)height;
	td.Width			= (UINT)width;
	td.Usage			= D3D10_USAGE_DEFAULT;
	td.SampleDesc.Count	= 1;
	td.SampleDesc.Quality=0;
	td.MiscFlags		= miscFlags;
	td.MipLevels		= 0;

	ID3D10Texture2D* texture;
	V_RETURN( d3dDevice->CreateTexture2D( &td, NULL, &texture ) );
	
	texture->GetDesc( &td );

	D3D10_SHADER_RESOURCE_VIEW_DESC sd;
	sd.ViewDimension				= D3D10_SRV_DIMENSION_TEXTURE2D;
	sd.Texture2D.MipLevels			= td.MipLevels;
	sd.Texture2D.MostDetailedMip	= 0;
	sd.Format						= td.Format;

	V_RETURN( d3dDevice->CreateShaderResourceView( texture, &sd, srvOut ) );

	D3D10_RENDER_TARGET_VIEW_DESC rd;
	rd.ViewDimension				= D3D10_RTV_DIMENSION_TEXTURE2D;
	rd.Format						= td.Format;
	rd.Texture2D.MipSlice			= 0;

	V_RETURN( d3dDevice->CreateRenderTargetView( texture, &rd, rtvOut ) );

	SAFE_RELEASE( texture );
	return S_OK;

}


HRESULT HierarchicalItemBuffer::Init( ID3D10Device* d3dDevice, size_t itemWidth, size_t itemHeight, size_t histoWidth, size_t histoHeight )
{
	// create input texture
	HRESULT hr;
	itemCount	= itemWidth * itemHeight;
	width		= itemWidth;
	height		= itemHeight;

	bucketWidth	= histoWidth;
	bucketHeight= histoHeight;
	this->bucketCount	= histoWidth * histoHeight;

	// create the input texture, render target and shader resource view
	V_RETURN( CreateRenderTargetAndResource( d3dDevice, width, height, DXGI_FORMAT_R16G16_FLOAT, &itemInputRenderTarget, &itemInputResource ) );

	// create the histogram texture
	V_RETURN( CreateRenderTargetAndResource( d3dDevice, histoWidth, histoHeight, DXGI_FORMAT_R32_FLOAT,
		&scatterRenderTarget, &scatterShaderResource, D3D10_RESOURCE_MISC_GENERATE_MIPS ) );

	// load effect file
	V_RETURN( CreateEffect( d3dDevice, &effect, D3D10_SHADER_OPTIMIZATION_LEVEL3 ) );

	BindEffectVariables( effect );

	
	V_RETURN( CreatePointVertexBuffer( d3dDevice, &pointBuffer, itemWidth, itemHeight, 16 ) );
	V_RETURN( CreateScatterLayout3( d3dDevice, &scatterLayout3, scatterPass3 ) );
	
	return S_OK;
}

void HierarchicalItemBuffer::Destroy() {
	SAFE_RELEASE( scatterTexture );
	SAFE_RELEASE( scatterRenderTarget );
	SAFE_RELEASE( scatterShaderResource );
	
	SAFE_RELEASE( itemInputRenderTarget );
	SAFE_RELEASE( itemInputResource );
	SAFE_RELEASE( effect );

	SAFE_RELEASE( pointBuffer );
	SAFE_RELEASE( scatterLayout3 );
}

HRESULT HierarchicalItemBuffer::CreatePointVertexBuffer( ID3D10Device* d3dDevice, ID3D10Buffer** vbOut, size_t width, size_t height, size_t blocking ) const
{
	struct vtx {
		USHORT x;
		USHORT y;
	};
	vtx* pbuffer = new vtx[ width * height ];

	size_t blockCntX = width / blocking;
	size_t blockCntY = height / blocking;


	USHORT xx = 0;
	USHORT yy = 0;
	size_t vertex = 0;
	for( size_t by = 0; by < blockCntY; ++by ) {
		for( size_t bx = 0; bx < blockCntX; ++bx ) {

			for( size_t y = 0; y < blocking; ++y ) {
				for( size_t x = 0; x < blocking; ++x ) {
					pbuffer[ vertex ].x = (USHORT)(bx * blocking + x);
					pbuffer[ vertex ].y = (USHORT)(by * blocking + y);
					vertex++;
				}
			}
		}
	}

	D3D10_BUFFER_DESC bd;
	bd.BindFlags	= D3D10_BIND_VERTEX_BUFFER;
	bd.ByteWidth	= (UINT)(sizeof( vtx ) * width * height);
	bd.CPUAccessFlags	= 0;
	bd.MiscFlags = 0;
	bd.Usage = D3D10_USAGE_IMMUTABLE;

	D3D10_SUBRESOURCE_DATA sd;
	sd.pSysMem = (void*)pbuffer;
	sd.SysMemPitch = 0;
	sd.SysMemSlicePitch = 0;

	HRESULT hr = d3dDevice->CreateBuffer( &bd, &sd, vbOut );
	
	delete[] pbuffer;
	return hr;
}


HRESULT HierarchicalItemBuffer::CreateEffect(
	ID3D10Device* d3dDevice,
	ID3D10Effect** effectOut,
	UINT compileFlags ) const 
{
	HRESULT hr = E_FAIL;
	TCHAR fileName[ 512 ];
	V_RETURN( StringCchPrintf( fileName, 512, TEXT("%s\\HierarchicalItemBuffer.fx"), EFFECT_PATH ) );

	ID3D10Blob* errorMsg = NULL;
	hr = D3DX10CreateEffectFromFile( fileName, NULL, NULL, "fx_4_0", compileFlags, 0,
		d3dDevice, NULL, NULL, effectOut, &errorMsg, NULL );

	if( FAILED( hr ) ) {
		if( errorMsg ) 
			MessageBoxA( NULL, (LPCSTR)errorMsg->GetBufferPointer(), "ERROR", MB_ICONERROR | MB_OK );
	}

	SAFE_RELEASE( errorMsg );
	return hr;
}


bool HierarchicalItemBuffer::BindEffectVariables( ID3D10Effect* d3dEffect ) {
	
	inputTexture = d3dEffect->GetVariableByName( "tInputTexture" )->AsShaderResource();
	if( !inputTexture->IsValid() )
		return false;

	scatterPass2 = d3dEffect->GetTechniqueByName( "tScatterToBuckets2" )->GetPassByIndex( 0 );
	if( !scatterPass2->IsValid() )
		return false;

	scatterPass3 = d3dEffect->GetTechniqueByName( "tScatterToBuckets3" )->GetPassByIndex( 0 );
	if( !scatterPass3->IsValid() )
		return false;

	return true;
}

HRESULT HierarchicalItemBuffer::CreateScatterLayout3(
	ID3D10Device* d3dDevice,
	ID3D10InputLayout** inputLayout,
	ID3D10EffectPass* effect ) const
{

	D3D10_INPUT_ELEMENT_DESC scatter3Desc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R16G16_UINT, 0, 0,D3D10_INPUT_PER_VERTEX_DATA , 0 }
	};

	HRESULT hr;
	D3D10_PASS_DESC pd;
	effect->GetDesc( &pd );
	V_RETURN( d3dDevice->CreateInputLayout( scatter3Desc, LENGTHOF( scatter3Desc ), 
		pd.pIAInputSignature, pd.IAInputSignatureSize, inputLayout ) );


	return S_OK;

}

HRESULT HierarchicalItemBuffer::ScatterToHistogram( ID3D10Device* d3dDevice, BOOL genMips )
{

	D3D10_VIEWPORT vpOld[ 8 ];
	UINT vpOldCount = 8;
	d3dDevice->RSGetViewports( &vpOldCount, &vpOld[0] );

	D3D10_VIEWPORT vp;
	vp.Height	= (UINT)bucketHeight;
	vp.Width	= (UINT)bucketWidth;
	vp.MaxDepth	= 1.f;
	vp.MinDepth	= 0.f;
	vp.TopLeftX	= 0;
	vp.TopLeftY	= 0;


	// bind the histogram texture to the output merger
	d3dDevice->ClearRenderTargetView( scatterRenderTarget, D3DXCOLOR( 0, 0, 0, 0 ) );
	d3dDevice->OMSetRenderTargets( 1, &scatterRenderTarget, NULL );

	d3dDevice->RSSetViewports( 1, &vp );


	
	if( 1 ) {
		// the vertex shader of this branch
		// reads the item buffer line by line

		ID3D10Buffer* nullBuffer = NULL;
		UINT offset	= 0;
		UINT stride = 0;

		d3dDevice->IASetVertexBuffers( 0, 1, &nullBuffer, &stride, &offset );
		d3dDevice->IASetIndexBuffer( nullBuffer, DXGI_FORMAT_R16_UINT, 0 );
		d3dDevice->IASetInputLayout( NULL );
		d3dDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_POINTLIST );
		inputTexture->SetResource( itemInputResource );

		scatterPass2->Apply( 0 );
		d3dDevice->Draw( (UINT)itemCount, 0 );

		ID3D10ShaderResourceView* nullView = NULL;
		d3dDevice->Draw( (UINT)itemCount, 0 );

		d3dDevice->VSSetShaderResources( 0, 1, &nullView );


	}
	else {
		// the vertex shader of this branch reads the
		// item buffer blockwise
		UINT stride = sizeof( UINT );
		UINT offset = 0;

		
		d3dDevice->IASetVertexBuffers( 0, 1, &pointBuffer, &stride, &offset );
		d3dDevice->IASetInputLayout( scatterLayout3 );
		d3dDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_POINTLIST );
		inputTexture->SetResource( itemInputResource );


		scatterPass3->Apply( 0 );
		d3dDevice->Draw( (UINT)itemCount, 0 );

		ID3D10ShaderResourceView* nullView = NULL;
		d3dDevice->VSSetShaderResources( 0, 1, &nullView );

	}
	d3dDevice->RSSetViewports( vpOldCount, &vpOld[0] );

	ID3D10RenderTargetView* nullRTV = NULL;
	d3dDevice->OMSetRenderTargets( 1, &nullRTV, NULL );
	d3dDevice->GenerateMips( scatterShaderResource );
	return S_OK;
}