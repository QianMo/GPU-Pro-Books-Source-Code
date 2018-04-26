#include "HOQDemo10.h"
#include "HierarchicalItemBuffer.h"
#include <fstream>

#define HIB_WIDTH 512
#define HIB_HEIGHT 512

#define MAX_CASTER_COUNT	510
#define MAX_RADIUS			20.f
#define MAX_FRAMES			300

// supported render paths
HRESULT CALLBACK ItemBufferRenderPath( ID3D10Device* d3dDevice );
HRESULT CALLBACK PredicatedRenderPath( ID3D10Device* d3dDevice );
HRESULT CALLBACK DefaultRenderPath( ID3D10Device* d3dDevice );
HRESULT CALLBACK InstancedRenderPath( ID3D10Device* d3dDevice );


RENDER_PATH_DESC gRenderPaths[4];
UINT gActiveRenderPath			= 3;
UINT gActiveCasters				= 250;
bool gShowVolumes				= true;

CDXUTTextHelper* gTextHelper	= NULL;
ID3DX10Sprite* gTextSprites		= NULL;
ID3DX10Font* gTextFont			= NULL;

float gFrameTimes[ MAX_FRAMES ];
UINT gCurrentFrame = 0;


// wooden box mesh
struct {
	CDXUTSDKMesh*			mesh;
	ID3D10InputLayout*		zfillLayout;
	ID3D10InputLayout*		texturedLightLayout;

	D3DXMATRIX				object2World;
	D3DXMATRIX				normal2World;
} woodenBox = { NULL, NULL, NULL };

// shadow caster mesh
struct {
	CDXUTSDKMesh*			mesh;
	ID3D10InputLayout*		zfillLayout;
	ID3D10InputLayout*		lightLayout;

	D3DXMATRIX*				object2World;
	D3DXMATRIX*				normal2World;

	Particle				particles[ MAX_CASTER_COUNT ];
	ID3D10Predicate*		predicates[ MAX_CASTER_COUNT ];
	
} shadowCaster = { NULL, NULL, NULL, NULL, NULL };


// D3D effects
struct {
	ID3D10Effect*				effect;

	ID3D10EffectTechnique*		zfillBoxTechnique;
	ID3D10EffectTechnique*		zfillCasterTechnique;

	ID3D10EffectTechnique*		volumeToStencilBoxTechnique;
	ID3D10EffectTechnique*		volumeToStencilCasterTechnique;


	ID3D10EffectTechnique*		lightBoxTechnique;
	ID3D10EffectTechnique*		ambientBoxTechnique;
	
	ID3D10EffectTechnique*		lightCasterTechnique;
	ID3D10EffectTechnique*		ambientCasterTechnique;

	ID3D10EffectTechnique*		buildItemBufferCasterTechnique;
	ID3D10EffectTechnique*		volumeToStencilCasterVisibilityTechnique;

	ID3D10EffectMatrixVariable*	object2WorldBox;
	ID3D10EffectMatrixVariable* normal2WorldBox;

	ID3D10EffectMatrixVariable* object2WorldCaster;
	ID3D10EffectMatrixVariable* normal2WorldCaster;

	ID3D10EffectMatrixVariable*	viewProjMat;
	ID3D10EffectVectorVariable*	lightPos;
	ID3D10EffectVectorVariable* eyePos;
	ID3D10EffectVectorVariable* histogramDimension;

	ID3D10EffectShaderResourceVariable* texture;
	ID3D10EffectShaderResourceVariable* itemHistogramTexture;

} mainEffect = { NULL, NULL, NULL, NULL, NULL };


struct {
	ID3D10RenderTargetView*		rtvItemBuffer;
	ID3D10RenderTargetView*		rtvHistogram;
	ID3D10ShaderResourceView*	srvHistogram;
	ID3D10Buffer*				vbItemBuffer;
} HIB = { NULL, NULL, NULL, NULL };


Camera gViewerCam;
Camera gLightCam;

float gLightAccel	= 12.f;
float gLightAngle	= 0.f;
float gLightRadius	= 15.f;
bool  gPlayAnimation= true;
int   gFixLightAngle=5;

ID3D10DepthStencilView* gOcclusionDepthBuffer = NULL;
HierarchicalItemBuffer* gItemBuffer = NULL;

//*****************************************************************************
// Wooden box mesh loader
//*****************************************************************************
HRESULT LoadWoodBox( ID3D10Device* d3dDevice )
{
	HRESULT hr = E_FAIL;

	TCHAR pathName[512];
	hr = StringCchPrintf( &pathName[0], sizeof( pathName ) / sizeof( pathName[0] ), TEXT("%s\\woodbox.sdkmesh"), ASSET_PATH );
	if( FAILED( hr ) )
		return hr;
	
	woodenBox.mesh	= new CDXUTSDKMesh();
	if( woodenBox.mesh == NULL )
		return E_OUTOFMEMORY;


	hr = woodenBox.mesh->Create( d3dDevice, pathName, true );
	if( FAILED( hr ) )
		return hr;
	
	D3DXMatrixScaling( &woodenBox.object2World, 5, 5, 5 );

	float det = D3DXMatrixDeterminant( &woodenBox.object2World );
	D3DXMATRIX tmp;
	D3DXMatrixInverse( &tmp, &det, &woodenBox.object2World );
	D3DXMatrixTranspose( &woodenBox.normal2World, &tmp );

	return hr;
}

void FreeWoodBoxResources() {
	SAFE_DELETE( woodenBox.mesh );
	SAFE_RELEASE( woodenBox.zfillLayout );
	SAFE_RELEASE( woodenBox.texturedLightLayout );
}

//*****************************************************************************
// Shadow Caster mesh loader
//*****************************************************************************
HRESULT LoadShadowCaster( ID3D10Device* d3dDevice ) 
{
	HRESULT hr	= E_FAIL;

	// assemble the full file name
	TCHAR pathName[ 512 ];
	hr = StringCchPrintf( &pathName[0], LENGTHOF( pathName ), TEXT("%s\\sc02.sdkmesh"), ASSET_PATH );
	if( FAILED( hr ) )
		return hr;

	// allocate a mesh object for the shadow caster
	shadowCaster.mesh	= new CDXUTSDKMesh();
	if( shadowCaster.mesh == NULL )
		return E_OUTOFMEMORY;

	// load the caster's mesh and tell DXUT to generate adjacency information
	hr = shadowCaster.mesh->Create( d3dDevice, pathName, true );
	if( FAILED( hr ) )
		return hr;


	shadowCaster.object2World	= new D3DXMATRIX[ MAX_CASTER_COUNT ];
	shadowCaster.normal2World	= new D3DXMATRIX[ MAX_CASTER_COUNT ];


	// init particle structs
	for( SIZE_T i = 0; i < MAX_CASTER_COUNT; ++i ) {
		float r = float(rand()) / float(0x7fff) * 2.f * D3DX_PI;
		float a = float(rand()) / float(0x7fff) * 2.f * D3DX_PI;
		float v = max( 0.2f, float(rand()) / float(0x7fff) * 0.4f );
		float h = float(rand()) / float(0x7fff) * 2.f * D3DX_PI;

		shadowCaster.particles[i].angle	= a;
		shadowCaster.particles[i].angularAccel	= v;
		shadowCaster.particles[i].radius		= r;
		shadowCaster.particles[i].height		=h ;
		shadowCaster.particles[i].radiusAccel	= max( 0.3f, float(rand()) / float(0x7fff) * 1.2f );
		shadowCaster.particles[i].heightAccel	= max( 0.3f, float(rand()) / float(0x7fff) * 0.6f );
	}

	// create query objects 
	D3D10_QUERY_DESC qd;
	qd.MiscFlags	= D3D10_QUERY_MISC_PREDICATEHINT;
	qd.Query		= D3D10_QUERY_OCCLUSION_PREDICATE;

	for( SIZE_T i = 0; i < MAX_CASTER_COUNT; ++i ) {
		CreateTransformationsFromParticle( shadowCaster.object2World+i, shadowCaster.normal2World+i, shadowCaster.particles+i );
		V_RETURN( d3dDevice->CreatePredicate( &qd, &shadowCaster.predicates[i] ) );
	}

	
	return hr;
}


void FreeShadowCasterResources()
{
	SAFE_DELETE( shadowCaster.mesh );
	SAFE_DELETE_ARRAY( shadowCaster.object2World );
	SAFE_DELETE_ARRAY( shadowCaster.normal2World );
	SAFE_RELEASE( shadowCaster.zfillLayout );
	SAFE_RELEASE( shadowCaster.lightLayout );
	
	for( SIZE_T i = 0 ; i < MAX_CASTER_COUNT; ++i ) {
		SAFE_RELEASE( shadowCaster.predicates[i] );
	}
}

// helper function to save a particle configuration to file
void StoreParticleConfiguration( LPCTSTR fileName ) {
	TCHAR tmp[ 256 ];
	StringCchPrintf( tmp, LENGTHOF( tmp ), L"%s\\%s", ASSET_PATH, fileName );
	HANDLE hFile = CreateFile(
		tmp,
		GENERIC_WRITE,
		0,
		NULL,
		CREATE_ALWAYS,
		FILE_FLAG_SEQUENTIAL_SCAN,
		NULL );

	if( hFile == INVALID_HANDLE_VALUE )
		return;

	for( UINT i = 0; i < MAX_CASTER_COUNT; ++i ) {
		DWORD bytesWritten = 0;
		WriteFile( hFile, &shadowCaster.particles[i], sizeof( Particle ), &bytesWritten, NULL );
	}

	CloseHandle( hFile );
}

// helper function to load a particle configuration from file
void LoadParticleConfiguration( LPCTSTR fileName ) {
	TCHAR tmp[ 256 ];
	StringCchPrintf( tmp, LENGTHOF( tmp ), L"%s\\%s", ASSET_PATH, fileName );
	HANDLE hFile = CreateFile(
		tmp,
		GENERIC_READ,
		0,
		NULL,
		OPEN_EXISTING,
		FILE_FLAG_SEQUENTIAL_SCAN,
		NULL );

	if( hFile == INVALID_HANDLE_VALUE )
		return;

	for( UINT i = 0; i < MAX_CASTER_COUNT; ++i ) {
		DWORD bytesRead = 0;
		ReadFile( hFile, &shadowCaster.particles[i], sizeof( Particle ), &bytesRead, NULL );
	}

	CloseHandle( hFile );

}
//*****************************************************************************
// Effect initialization
//*****************************************************************************
HRESULT LoadEffects( ID3D10Device* d3dDevice ) 
{
	HRESULT hr	= E_FAIL;

	TCHAR pathName[ 512 ];
	hr = StringCchPrintf( pathName, LENGTHOF( pathName ), TEXT("%s\\Main.fx"), EFFECT_PATH );
	if( FAILED( hr ) )
		return hr;

	// load effect from file
	UINT hlslFlags = 0;
	ID3D10Blob* errorMsg = NULL;

#if defined(DEBUG) || defined(_DEBUG)
	hlslFlags |= D3D10_SHADER_DEBUG | D3D10_SHADER_SKIP_OPTIMIZATION;
#else
	hlslFlags |= D3D10_SHADER_OPTIMIZATION_LEVEL3;
#endif
	hr = D3DX10CreateEffectFromFile( 
		pathName, NULL, NULL, "fx_4_0", hlslFlags, 0, d3dDevice, 
		NULL, NULL, &(mainEffect.effect), &errorMsg, NULL );

	if( FAILED( hr ) ) {
		if( errorMsg ) {
			MessageBoxA( NULL, (LPCSTR)errorMsg->GetBufferPointer(), "FX ERROR", MB_ICONERROR | MB_OK );
		}
		else
			MessageBox( NULL, TEXT("FX compilation failed for some unknown reason"), TEXT("FX ERROR"), MB_ICONERROR | MB_OK );
	}

	SAFE_RELEASE( errorMsg );

	return hr;
}

void FreeEffectResources()
{
	SAFE_RELEASE( mainEffect.effect );
}

//*****************************************************************************
// bind shader techniques and variables
//*****************************************************************************
HRESULT BindShaderVariables() {
	ID3D10Effect* const fxTmp	= mainEffect.effect;
	

	// bind zfill techniques
	mainEffect.zfillBoxTechnique	= fxTmp->GetTechniqueByName( "tZFILLBox" );
	if( !mainEffect.zfillBoxTechnique->IsValid() )
		return E_FAIL;

	mainEffect.zfillCasterTechnique	= fxTmp->GetTechniqueByName( "tZFILLCaster" );
	if( !mainEffect.zfillCasterTechnique->IsValid() )
		return E_FAIL;



	// bind volume to stencil techniques
	mainEffect.volumeToStencilBoxTechnique = fxTmp->GetTechniqueByName( "tVolumesToStencilBox" );
	if( !mainEffect.volumeToStencilBoxTechnique->IsValid() )
		return E_FAIL;

	mainEffect.volumeToStencilCasterTechnique = fxTmp->GetTechniqueByName( "tVolumesToStencilCaster" );
	if( !mainEffect.volumeToStencilCasterTechnique->IsValid() )
		return E_FAIL;


	mainEffect.volumeToStencilCasterVisibilityTechnique = fxTmp->GetTechniqueByName( "tVolumesToStencilCasterVisibility" );
	if( !mainEffect.volumeToStencilCasterVisibilityTechnique->IsValid() )
		return E_FAIL;

	// bind light/shadow pass techniques
	mainEffect.lightBoxTechnique		= fxTmp->GetTechniqueByName( "tLightBox" );
	if( !mainEffect.lightBoxTechnique->IsValid() )
		return E_FAIL;

	mainEffect.lightCasterTechnique		= fxTmp->GetTechniqueByName( "tLightCaster" );
	if( !mainEffect.lightBoxTechnique->IsValid() )
		return E_FAIL;


	// ambient passes
	mainEffect.ambientBoxTechnique		= fxTmp->GetTechniqueByName( "tAmbientBox" );
	if( !mainEffect.ambientBoxTechnique->IsValid() )
		return E_FAIL;

	mainEffect.ambientCasterTechnique	= fxTmp->GetTechniqueByName( "tAmbientCaster" );
	if( !mainEffect.ambientCasterTechnique->IsValid() )
		return E_FAIL;

	mainEffect.buildItemBufferCasterTechnique	= fxTmp->GetTechniqueByName( "tBuildItemBufferCaster" );
	if( !mainEffect.buildItemBufferCasterTechnique->IsValid() )
		return E_FAIL;



	// variables
	mainEffect.object2WorldBox			= fxTmp->GetVariableByName( "mObject2WorldBox" )->AsMatrix();
	if( !mainEffect.object2WorldBox->IsValid() )
		return E_FAIL;

	mainEffect.normal2WorldBox	= fxTmp->GetVariableByName( "mNormal2WorldBox" )->AsMatrix();
	if( !mainEffect.normal2WorldBox->IsValid() )
		return E_FAIL;


	mainEffect.object2WorldCaster		= fxTmp->GetVariableByName( "mObject2WorldCaster" )->AsMatrix();
	if( !mainEffect.object2WorldCaster->IsValid() )
		return E_FAIL;

	mainEffect.normal2WorldCaster		= fxTmp->GetVariableByName( "mNormal2WorldCaster" )->AsMatrix();
	if( !mainEffect.normal2WorldCaster->IsValid() )
		return E_FAIL;

	

	mainEffect.viewProjMat		= fxTmp->GetVariableByName( "mViewProjMat" )->AsMatrix();
	if( !mainEffect.viewProjMat->IsValid() )
		return E_FAIL;

	mainEffect.lightPos			= fxTmp->GetVariableByName( "vLightPos" )->AsVector();
	if( !mainEffect.lightPos->IsValid() )
		return E_FAIL;

	mainEffect.eyePos			= fxTmp->GetVariableByName( "vEyePos" )->AsVector();
	if( !mainEffect.eyePos->IsValid() )
		return E_FAIL;


	mainEffect.histogramDimension= fxTmp->GetVariableByName( "vHistogramDimension" )->AsVector();
	if( !mainEffect.histogramDimension->IsValid() )
		return E_FAIL;

	mainEffect.texture			= fxTmp->GetVariableByName( "diffTexture" )->AsShaderResource();
	if( !mainEffect.texture->IsValid() )
		return E_FAIL;

	mainEffect.itemHistogramTexture	= fxTmp->GetVariableByName( "itemHistogram" )->AsShaderResource();
	if( !mainEffect.itemHistogramTexture->IsValid() )
		return E_FAIL;

	
	return S_OK;
}

//*****************************************************************************
// input layout initialization
//*****************************************************************************
HRESULT CreateInputLayout( ID3D10Device* d3dDevice )
{
	HRESULT hr = E_FAIL;

	// ZFILL layout
	D3D10_INPUT_ELEMENT_DESC zfillLayoutDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 }
	};

	D3D10_PASS_DESC pd;
	mainEffect.zfillBoxTechnique->GetPassByIndex(0)->GetDesc( &pd );
	
	V_RETURN( d3dDevice->CreateInputLayout( zfillLayoutDesc, LENGTHOF( zfillLayoutDesc ), pd.pIAInputSignature, pd.IAInputSignatureSize, &woodenBox.zfillLayout ) );

	mainEffect.zfillCasterTechnique->GetPassByIndex(0)->GetDesc( &pd );
	V_RETURN( d3dDevice->CreateInputLayout( zfillLayoutDesc, LENGTHOF( zfillLayoutDesc ), pd.pIAInputSignature, pd.IAInputSignatureSize, &shadowCaster.zfillLayout ) );

	// layout for wooden box
	D3D10_INPUT_ELEMENT_DESC lightBoxInputDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D10_INPUT_PER_VERTEX_DATA, 0 }
	};

	mainEffect.lightBoxTechnique->GetPassByIndex(0)->GetDesc( &pd );
	V_RETURN( d3dDevice->CreateInputLayout( lightBoxInputDesc, LENGTHOF( lightBoxInputDesc ) ,
		pd.pIAInputSignature, pd.IAInputSignatureSize, &woodenBox.texturedLightLayout ) );


	// layout for caster
	D3D10_INPUT_ELEMENT_DESC lightCasterDesc[] = 
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D10_INPUT_PER_VERTEX_DATA, 0 },
	};

	mainEffect.lightCasterTechnique->GetPassByIndex(0)->GetDesc( &pd );
	V_RETURN( d3dDevice->CreateInputLayout( lightCasterDesc, LENGTHOF( lightCasterDesc ),
		pd.pIAInputSignature, pd.IAInputSignatureSize, &shadowCaster.lightLayout ) );
	return S_OK;
}

//*****************************************************************************
// Camera setup
//*****************************************************************************
void InitCameras() {
	
	gViewerCam.SetLookAt( float3( -37.476f, 25.2259f, -54.1214f ), float3( 0.f, 0.f, 0.f ), float3( 0.f, 1.f, 0.f ) );
	gViewerCam.SetPerspective( 1.f, 45.f, 0.01f, 300.f );
	gViewerCam.SetViewport( 100, 100 );
	
	gLightCam.SetLookAt( float3( 0.f, 30.f, 0.01f ), float3( 0.f, 0.f, 0.f ), float3( 0.f, 1.f, 0.f ) );
	gLightCam.SetPerspective( 1.f, 110.f, 0.01f, 1000.f );
	gLightCam.SetViewport( HIB_WIDTH, HIB_HEIGHT );
}
//*****************************************************************************
// render paths initalization
//*****************************************************************************
void InitRenderPaths() {
	StringCchCopy( gRenderPaths[0].funcName, LENGTHOF( gRenderPaths[0].funcName ), TEXT("Default Render Path") );
	gRenderPaths[0].renderFunc	= &DefaultRenderPath;

	StringCchCopy( gRenderPaths[1].funcName, LENGTHOF( gRenderPaths[1].funcName ), TEXT("Instanced Render Path") );
	gRenderPaths[1].renderFunc = &InstancedRenderPath;

	StringCchCopy( gRenderPaths[2].funcName, LENGTHOF( gRenderPaths[2].funcName ), TEXT("Predicated Render Path") );
	gRenderPaths[2].renderFunc = &PredicatedRenderPath;

	StringCchCopy( gRenderPaths[3].funcName, LENGTHOF( gRenderPaths[3].funcName ), TEXT("Item Buffer Render Path") );
	gRenderPaths[3].renderFunc = &ItemBufferRenderPath;

}

HRESULT RenderCaster( ID3D10Device* d3dDevice, ID3D10InputLayout* layout, ID3D10EffectTechnique* technique ) {
	d3dDevice->IASetInputLayout( layout );
	for( SIZE_T i = 0; i < gActiveCasters; ++i ) {
		mainEffect.object2WorldCaster->SetMatrix( shadowCaster.object2World[i] );
		mainEffect.normal2WorldCaster->SetMatrix( shadowCaster.normal2World[i] );

		shadowCaster.mesh->Render( d3dDevice, technique );
	}

	return S_OK;
}


HRESULT RenderCasterAdjacent( ID3D10Device* d3dDevice, ID3D10InputLayout* layout, ID3D10EffectTechnique* technique ) {
	d3dDevice->IASetInputLayout( layout );
	for( SIZE_T i = 0; i < gActiveCasters; ++i ) {
		mainEffect.object2WorldCaster->SetMatrix( shadowCaster.object2World[i] );
		mainEffect.normal2WorldCaster->SetMatrix( shadowCaster.normal2World[i] );

		shadowCaster.mesh->RenderAdjacent( d3dDevice, technique );
	}

	return S_OK;
}

HRESULT RenderCasterPredicatedAdjacent( ID3D10Device* d3dDevice, ID3D10InputLayout* layout, ID3D10EffectTechnique* technique ) {
	d3dDevice->IASetInputLayout( layout );
	for( SIZE_T i = 0; i < gActiveCasters; ++i ) {
		mainEffect.object2WorldCaster->SetMatrix( shadowCaster.object2World[i] );
		mainEffect.normal2WorldCaster->SetMatrix( shadowCaster.normal2World[i] );
		d3dDevice->SetPredication( shadowCaster.predicates[i], FALSE );
		shadowCaster.mesh->RenderAdjacent( d3dDevice, technique );
		d3dDevice->SetPredication( NULL, FALSE );
	}

	return S_OK;
}

HRESULT RenderCasterPredicated( ID3D10Device* d3dDevice, ID3D10InputLayout* layout, ID3D10EffectTechnique* technique ) { 
	d3dDevice->IASetInputLayout( layout );
	for( SIZE_T i = 0; i < gActiveCasters; ++i ) {
		mainEffect.object2WorldCaster->SetMatrix( shadowCaster.object2World[i] );
		mainEffect.normal2WorldCaster->SetMatrix( shadowCaster.normal2World[i] );
		//d3dDevice->SetPredication( shadowCaster.predicates[i], FALSE );
		shadowCaster.mesh->Render( d3dDevice, technique );
		//d3dDevice->SetPredication( NULL, FALSE );
	}

	return S_OK;
}

HRESULT LoadCasterTransformations() {
	mainEffect.object2WorldCaster->SetMatrixArray( (float*)(&shadowCaster.object2World[0]), 0, gActiveCasters );
	mainEffect.normal2WorldCaster->SetMatrixArray( (float*)(&shadowCaster.normal2World[0]), 0, gActiveCasters );

	return S_OK;
}

HRESULT RenderCasterInstanced( ID3D10Device* d3dDevice, ID3D10InputLayout* layout, ID3D10EffectTechnique* technique ) {
	ID3D10Buffer* ib = shadowCaster.mesh->GetIB10(0);
	ID3D10Buffer* vb = shadowCaster.mesh->GetVB10(0,0);
	UINT stride		= 32;
	UINT offset		= 0;

	UINT indexCount	= (UINT)shadowCaster.mesh->GetNumIndices(0);
	d3dDevice->IASetInputLayout( layout );
	d3dDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
	d3dDevice->IASetVertexBuffers( 0, 1, &vb, &stride, &offset );
	d3dDevice->IASetIndexBuffer( ib, DXGI_FORMAT_R32_UINT, 0 );

	technique->GetPassByIndex(0)->Apply(0);
	d3dDevice->DrawIndexedInstanced( indexCount, gActiveCasters, 0, 0, 0 );

	return S_OK;
}

HRESULT RenderCasterInstancedAdjacent( ID3D10Device* d3dDevice, ID3D10InputLayout* layout, ID3D10EffectTechnique* technique ) {
	ID3D10Buffer* ib = shadowCaster.mesh->GetAdjIB10(0);
	ID3D10Buffer* vb = shadowCaster.mesh->GetVB10(0,0);
	UINT stride		= 32;
	UINT offset		= 0;

	UINT indexCount	= (UINT)shadowCaster.mesh->GetNumIndices(0);
	d3dDevice->IASetInputLayout( layout );
	d3dDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ );
	d3dDevice->IASetVertexBuffers( 0, 1, &vb, &stride, &offset );
	d3dDevice->IASetIndexBuffer( ib, DXGI_FORMAT_R32_UINT, 0 );

	technique->GetPassByIndex(0)->Apply(0);
	d3dDevice->DrawIndexedInstanced( 2*indexCount, gActiveCasters, 0, 0, 0 );

	return S_OK;

}

//*****************************************************************************
// animate light position
//*****************************************************************************
void AnimateLightPosition( float elapsedTime ) {
	gLightAngle += elapsedTime * gLightAccel;
	gLightAngle = ( gLightAngle > 360.f ) ? gLightAngle-360.f : gLightAngle;

	float x	= gLightRadius * cos( D3DX_PI * gLightAngle / 180.f );
	float z = gLightRadius * sin( D3DX_PI * gLightAngle / 180.f );

	gLightCam.SetLookAt( D3DXVECTOR3( x, gLightCam.eye.y, z ), gLightCam.at, gLightCam.up );

}

void AnimateCasters( float elapsedTime ) {
	
	for( SIZE_T i = 0; i < MAX_CASTER_COUNT; ++i ) {
		float angle = shadowCaster.particles[i].angle + elapsedTime * shadowCaster.particles[i].angularAccel;
		shadowCaster.particles[i].angle	= ( angle > 2.f * D3DX_PI ) ? angle - 2.f * D3DX_PI : angle;
		float radius = shadowCaster.particles[i].radius + elapsedTime * shadowCaster.particles[i].radiusAccel;
		shadowCaster.particles[i].radius = ( radius > 2.f * D3DX_PI ) ? radius - 2.f * D3DX_PI : radius;

		float height = shadowCaster.particles[i].height + elapsedTime * shadowCaster.particles[i].heightAccel;
		shadowCaster.particles[i].height = ( radius > 2.f * D3DX_PI ) ? height - 2.f * D3DX_PI : height;
	}

	for( SIZE_T i = 0; i < gActiveCasters; ++i )
		CreateTransformationsFromParticle( shadowCaster.object2World+i, shadowCaster.normal2World+i, shadowCaster.particles+i );
}


//*****************************************************************************
// visibility pass for predicated rendering
//*****************************************************************************
HRESULT LaunchPredicates( ID3D10Device* d3dDevice ) {

	
	// bind low resolution depth buffer
	ID3D10RenderTargetView* rtv;
	ID3D10DepthStencilView* dsv;
	d3dDevice->OMGetRenderTargets( 1, &rtv, &dsv );
	d3dDevice->OMSetRenderTargets( 0, NULL, gOcclusionDepthBuffer );
	d3dDevice->ClearDepthStencilView( gOcclusionDepthBuffer, D3D10_CLEAR_DEPTH | D3D10_CLEAR_STENCIL, 1.f, 0 );

	// store current viewport settings
	D3D10_VIEWPORT vp;
	UINT vpCount = 1;
	d3dDevice->RSGetViewports( &vpCount, &vp );

	// viewport for occlusion queries
	D3D10_VIEWPORT occlVp;
	occlVp.Height	= HIB_HEIGHT;
	occlVp.Width	= HIB_WIDTH;
	occlVp.MaxDepth	= 1.f;
	occlVp.MinDepth	= 0.f;
	occlVp.TopLeftX	= 0;
	occlVp.TopLeftY	= 0;

	d3dDevice->RSSetViewports( 1, &occlVp );

	D3DXMATRIX viewProjMat;
	gLightCam.GetViewProjMat( &viewProjMat );
	mainEffect.viewProjMat->SetMatrix( viewProjMat );

	// render box to depth buffer
	d3dDevice->IASetInputLayout( woodenBox.zfillLayout );
	mainEffect.object2WorldBox->SetMatrix( woodenBox.object2World );
	mainEffect.normal2WorldBox->SetMatrix( woodenBox.normal2World );
	woodenBox.mesh->Render( d3dDevice, mainEffect.zfillBoxTechnique );


	// launch one predicate for each shadow caster and render against depth buffer
	d3dDevice->IASetInputLayout( shadowCaster.zfillLayout );
	for( SIZE_T i = 0; i < gActiveCasters; ++i ) {
		mainEffect.object2WorldCaster->SetMatrix( shadowCaster.object2World[i] );
		mainEffect.normal2WorldCaster->SetMatrix( shadowCaster.normal2World[i] );
		shadowCaster.predicates[i]->Begin();
		shadowCaster.mesh->Render( d3dDevice, mainEffect.zfillCasterTechnique );
		shadowCaster.predicates[i]->End();
	}

	// restore old depth buffer
	d3dDevice->OMSetRenderTargets( 1, &rtv, dsv );
	d3dDevice->RSSetViewports( 1, &vp );
	SAFE_RELEASE( rtv );
	SAFE_RELEASE( dsv );
	return S_OK;
}

HRESULT RenderItemBuffer( ID3D10Device* d3dDevice ) {

	// bind low resolution depth buffer
	ID3D10RenderTargetView* rtv;
	ID3D10DepthStencilView* dsv;
	ID3D10RenderTargetView* htv = gItemBuffer->GetItemBufferRenderTargetView();

	d3dDevice->OMGetRenderTargets( 1, &rtv, &dsv );
	d3dDevice->OMSetRenderTargets( 1, &htv, gOcclusionDepthBuffer );
	d3dDevice->ClearDepthStencilView( gOcclusionDepthBuffer, D3D10_CLEAR_DEPTH | D3D10_CLEAR_STENCIL, 1.f, 0 );
	d3dDevice->ClearRenderTargetView( htv, D3DXCOLOR( -2, -2, -2, -2 ) );

	// store current viewport settings
	D3D10_VIEWPORT vp[8];
	UINT vpCount = 8;
	d3dDevice->RSGetViewports( &vpCount, vp );


	// viewport for occlusion queries
	D3D10_VIEWPORT occlVp;
	occlVp.Height	= HIB_HEIGHT;
	occlVp.Width	= HIB_WIDTH;
	occlVp.MaxDepth	= 1.f;
	occlVp.MinDepth	= 0.f;
	occlVp.TopLeftX	= 0;
	occlVp.TopLeftY	= 0;

	d3dDevice->RSSetViewports( 1, &occlVp );

	D3DXMATRIX viewProjMat;
	gLightCam.GetViewProjMat( &viewProjMat );
	mainEffect.viewProjMat->SetMatrix( viewProjMat );

	// render box to depth buffer
	d3dDevice->IASetInputLayout( woodenBox.zfillLayout );
	mainEffect.object2WorldBox->SetMatrix( woodenBox.object2World );
	mainEffect.normal2WorldBox->SetMatrix( woodenBox.normal2World );
	woodenBox.mesh->Render( d3dDevice, mainEffect.zfillBoxTechnique );


	// draw into the item buffer
	D3DXVECTOR4 histoViewport;
	gItemBuffer->GetHistogramDimensions( histoViewport.x, histoViewport.y );
	histoViewport.z = 1.f / histoViewport.x;
	histoViewport.w = 1.f / histoViewport.y;

	mainEffect.histogramDimension->SetFloatVector( histoViewport );
	RenderCasterInstanced( d3dDevice, shadowCaster.zfillLayout, mainEffect.buildItemBufferCasterTechnique );


	// build histogram
	gItemBuffer->ScatterToHistogram( d3dDevice, FALSE );


	// restore previous RS and OM
	d3dDevice->OMSetRenderTargets( 1, &rtv, dsv );
	d3dDevice->RSSetViewports( vpCount, vp );
	SAFE_RELEASE( rtv );
	SAFE_RELEASE( dsv );


	return S_OK;

}

//*****************************************************************************
// different render paths
//*****************************************************************************
HRESULT CALLBACK DefaultRenderPath( ID3D10Device* d3dDevice ) 
{
	// viewproj setup
	D3DXMATRIX viewProjMat;
	gViewerCam.GetViewProjMat( &viewProjMat );
	mainEffect.viewProjMat->SetMatrix( viewProjMat );
	

	// render box to depth buffer
	d3dDevice->IASetInputLayout( woodenBox.zfillLayout );
	mainEffect.object2WorldBox->SetMatrix( woodenBox.object2World );
	mainEffect.normal2WorldBox->SetMatrix( woodenBox.normal2World );
	woodenBox.mesh->Render( d3dDevice, mainEffect.zfillBoxTechnique );

	// render shadow casters
	RenderCaster( d3dDevice, shadowCaster.zfillLayout, mainEffect.zfillCasterTechnique );


	// set light position
	D3DXVECTOR4 lPos;
	gLightCam.GetEyePos( &lPos );
	mainEffect.lightPos->SetFloatVector( lPos );

	gViewerCam.GetEyePos( &lPos );
	mainEffect.eyePos->SetFloatVector( lPos );


	// draw box volume
	d3dDevice->IASetInputLayout( woodenBox.zfillLayout );
	woodenBox.mesh->RenderAdjacent( d3dDevice, mainEffect.volumeToStencilBoxTechnique);


	// draw caster volumes
	RenderCasterAdjacent( d3dDevice, shadowCaster.zfillLayout, mainEffect.volumeToStencilCasterTechnique );


	// ambient pass
	d3dDevice->IASetInputLayout( woodenBox.texturedLightLayout );
	woodenBox.mesh->Render( d3dDevice, mainEffect.ambientBoxTechnique, mainEffect.texture );

	RenderCaster( d3dDevice, shadowCaster.lightLayout, mainEffect.ambientCasterTechnique );


	// shadow pass
	d3dDevice->IASetInputLayout( woodenBox.texturedLightLayout );
	woodenBox.mesh->Render( d3dDevice, mainEffect.lightBoxTechnique, mainEffect.texture );

	RenderCaster( d3dDevice, shadowCaster.lightLayout, mainEffect.lightCasterTechnique );

	if( gShowVolumes )
		RenderCasterAdjacent( d3dDevice, shadowCaster.zfillLayout, mainEffect.effect->GetTechniqueByName("tRenderVolumes") );
	
	return S_OK;
}


HRESULT CALLBACK InstancedRenderPath( ID3D10Device* d3dDevice )
{
	// view projection setup
	D3DXMATRIX viewProjMat;
	gViewerCam.GetViewProjMat( &viewProjMat );
	mainEffect.viewProjMat->SetMatrix( viewProjMat );
	
	
	// load caster transformation matrices into a constant buffer
	LoadCasterTransformations();

	// load box transformation matrices
	mainEffect.object2WorldBox->SetMatrix( woodenBox.object2World );
	mainEffect.normal2WorldBox->SetMatrix( woodenBox.normal2World );


	// draw box to depth buffer
	d3dDevice->IASetInputLayout( woodenBox.zfillLayout );
	woodenBox.mesh->Render( d3dDevice, mainEffect.zfillBoxTechnique );

	// render shadow casters
	RenderCasterInstanced( d3dDevice, shadowCaster.zfillLayout, mainEffect.zfillCasterTechnique );


	// set light position
	D3DXVECTOR4 lPos;
	gLightCam.GetEyePos( &lPos );
	mainEffect.lightPos->SetFloatVector( lPos );

	gViewerCam.GetEyePos( &lPos );
	mainEffect.eyePos->SetFloatVector( lPos );


	// draw box volumes
	d3dDevice->IASetInputLayout( woodenBox.zfillLayout );
	woodenBox.mesh->RenderAdjacent( d3dDevice, mainEffect.volumeToStencilBoxTechnique);


	// draw caster volumes
	RenderCasterInstancedAdjacent( d3dDevice, shadowCaster.zfillLayout, mainEffect.volumeToStencilCasterTechnique );

	// ambient pass
	d3dDevice->IASetInputLayout( woodenBox.texturedLightLayout );
	woodenBox.mesh->Render( d3dDevice, mainEffect.ambientBoxTechnique, mainEffect.texture );

	RenderCasterInstanced( d3dDevice, shadowCaster.lightLayout, mainEffect.ambientCasterTechnique );


	// shadow pass
	d3dDevice->IASetInputLayout( woodenBox.texturedLightLayout );
	woodenBox.mesh->Render( d3dDevice, mainEffect.lightBoxTechnique, mainEffect.texture );


	RenderCasterInstanced( d3dDevice, shadowCaster.lightLayout, mainEffect.lightCasterTechnique );
	
	if( gShowVolumes )
		RenderCasterInstancedAdjacent( d3dDevice, shadowCaster.zfillLayout, mainEffect.effect->GetTechniqueByName("tRenderVolumes") );

	return S_OK;
}


HRESULT CALLBACK PredicatedRenderPath( ID3D10Device* d3dDevice )
{
	LaunchPredicates( d3dDevice );

	// viewproj setup
	D3DXMATRIX viewProjMat;
	gViewerCam.GetViewProjMat( &viewProjMat );
	mainEffect.viewProjMat->SetMatrix( viewProjMat );
	
	// draw box to depth buffer
	d3dDevice->IASetInputLayout( woodenBox.zfillLayout );
	mainEffect.object2WorldBox->SetMatrix( woodenBox.object2World );
	mainEffect.normal2WorldBox->SetMatrix( woodenBox.normal2World );
	woodenBox.mesh->Render( d3dDevice, mainEffect.zfillBoxTechnique );

	// render shadow casters
	RenderCaster( d3dDevice, shadowCaster.zfillLayout, mainEffect.zfillCasterTechnique );

	// set light position
	D3DXVECTOR4 lPos;
	gLightCam.GetEyePos( &lPos );
	mainEffect.lightPos->SetFloatVector( lPos );

	gViewerCam.GetEyePos( &lPos );
	mainEffect.eyePos->SetFloatVector( lPos );


	// draw box volumes
	d3dDevice->IASetInputLayout( woodenBox.zfillLayout );
	woodenBox.mesh->RenderAdjacent( d3dDevice, mainEffect.volumeToStencilBoxTechnique);

	// draw caster volumes
	RenderCasterPredicatedAdjacent( d3dDevice, shadowCaster.zfillLayout, mainEffect.volumeToStencilCasterTechnique );

	// ambient pass
	d3dDevice->IASetInputLayout( woodenBox.texturedLightLayout );
	woodenBox.mesh->Render( d3dDevice, mainEffect.ambientBoxTechnique, mainEffect.texture );

	RenderCaster( d3dDevice, shadowCaster.lightLayout, mainEffect.ambientCasterTechnique );


	// shadow pass
	d3dDevice->IASetInputLayout( woodenBox.texturedLightLayout );
	woodenBox.mesh->Render( d3dDevice, mainEffect.lightBoxTechnique, mainEffect.texture );


	RenderCaster( d3dDevice, shadowCaster.lightLayout, mainEffect.lightCasterTechnique );
	
	if( gShowVolumes )
		RenderCasterPredicatedAdjacent( d3dDevice, shadowCaster.zfillLayout, mainEffect.effect->GetTechniqueByName("tRenderVolumes") );
	

	return S_OK;
}

HRESULT CALLBACK ItemBufferRenderPath( ID3D10Device* d3dDevice ) {

	// build item buffer
	LoadCasterTransformations();
	RenderItemBuffer( d3dDevice );

	// viewproj setup
	D3DXMATRIX viewProjMat;
	gViewerCam.GetViewProjMat( &viewProjMat );
	mainEffect.viewProjMat->SetMatrix( viewProjMat );

	// load box transformation
	mainEffect.object2WorldBox->SetMatrix( woodenBox.object2World );
	mainEffect.normal2WorldBox->SetMatrix( woodenBox.normal2World );


	// render box 
	d3dDevice->IASetInputLayout( woodenBox.zfillLayout );
	woodenBox.mesh->Render( d3dDevice, mainEffect.zfillBoxTechnique );

	// render shadow casters
	RenderCasterInstanced( d3dDevice, shadowCaster.zfillLayout, mainEffect.zfillCasterTechnique );


	// set light position
	D3DXVECTOR4 lPos;
	gLightCam.GetEyePos( &lPos );
	mainEffect.lightPos->SetFloatVector( lPos );

	gViewerCam.GetEyePos( &lPos );
	mainEffect.eyePos->SetFloatVector( lPos );


	// draw box volumes
	d3dDevice->IASetInputLayout( woodenBox.zfillLayout );
	woodenBox.mesh->RenderAdjacent( d3dDevice, mainEffect.volumeToStencilBoxTechnique);


	// draw caster volumes
	mainEffect.itemHistogramTexture->SetResource( gItemBuffer->GetHistogramShaderResourceView() );
	RenderCasterInstancedAdjacent( d3dDevice, shadowCaster.zfillLayout, mainEffect.volumeToStencilCasterVisibilityTechnique );
	ID3D10ShaderResourceView* nullView = NULL;
	d3dDevice->GSSetShaderResources( 0, 1, &nullView );

	// ambient pass
	d3dDevice->IASetInputLayout( woodenBox.texturedLightLayout );
	woodenBox.mesh->Render( d3dDevice, mainEffect.ambientBoxTechnique, mainEffect.texture );

	RenderCasterInstanced( d3dDevice, shadowCaster.lightLayout, mainEffect.ambientCasterTechnique );


	// shadow pass
	d3dDevice->IASetInputLayout( woodenBox.texturedLightLayout );
	woodenBox.mesh->Render( d3dDevice, mainEffect.lightBoxTechnique, mainEffect.texture );


	RenderCasterInstanced( d3dDevice, shadowCaster.lightLayout, mainEffect.lightCasterTechnique );

	if( gShowVolumes ) {
		mainEffect.itemHistogramTexture->SetResource( gItemBuffer->GetHistogramShaderResourceView() );
		RenderCasterInstancedAdjacent( d3dDevice, shadowCaster.zfillLayout, mainEffect.effect->GetTechniqueByName("tRenderVolumesCasterVisibility") );
	}

	d3dDevice->GSSetShaderResources( 0, 1, &nullView );



	return S_OK;
}



//*****************************************************************************
// default callback functions
//*****************************************************************************
void CALLBACK OnFrameMove( 
    double time,
	float elapsedTime, 
	void* userContext ) 
{
	if( gFixLightAngle == 0 ) {
		gLightAngle = 261.72f;
		elapsedTime = 0;
	}
	else if( gFixLightAngle == 1 ) {
		gLightAngle = 31.86f;
		elapsedTime = 0;
	}
	else if( gFixLightAngle == 2 ) {
		gLightAngle = 117.14f;
		elapsedTime = 0;
	}
	else if( gFixLightAngle == 3 ) {
		gLightAngle = 181.90f;
		elapsedTime = 0;
	}

	if( gPlayAnimation ) {
		AnimateLightPosition( elapsedTime );
		AnimateCasters( elapsedTime );
	}
}

void CALLBACK OnKeyboard( 
    UINT nChar, 
	bool keyDown, 
	bool altDown, 
	void* userContext )
{
	if( nChar == 72 && keyDown )
		gShowVolumes = !gShowVolumes;
	else if( nChar == 71 && keyDown ) {
		LoadParticleConfiguration( TEXT("particles.dat") );
		gFixLightAngle = ( gFixLightAngle + 1 ) % 5;
	}

	else if( nChar == 77 && keyDown )
		StoreParticleConfiguration( TEXT("particles.dat") );
	else if( nChar == 78 && keyDown )
		LoadParticleConfiguration( TEXT("particles.dat") );
}

void CALLBACK OnMouse( 
   bool leftButtonDown, 
   bool rightButtonDown, 
   bool middleButtonDown,
   bool sideButtonDown1,
   bool sidebuttonDown2,
   int	mouseWheelDelta, 
   int	xPos, 
   int	yPos, 
   void* userContext )
{
}


LRESULT CALLBACK OnMsgProc(
	HWND hwnd,
	UINT msg,
	WPARAM wParam,
	LPARAM lParam,
	bool* noFurtherProcessing,
	void* userContext )
{

	if( msg == WM_KEYDOWN ) {
		if( wParam == VK_SPACE ) {
			gActiveRenderPath = ( gActiveRenderPath + 1 ) % 4;
			SetWindowText( hwnd, gRenderPaths[gActiveRenderPath].funcName );
		}
		else if( wParam == VK_F4 ) {
			gPlayAnimation = !gPlayAnimation;
		}
		else if( wParam == VK_ADD ) {
			gActiveCasters = min( MAX_CASTER_COUNT, gActiveCasters+10 );
		}

		else if( wParam == VK_SUBTRACT ) {
			gActiveCasters = max( 10, gActiveCasters-10 );
		}
	}

	return 0;
}

bool CALLBACK OnModifyDeviceSettings(
	DXUTDeviceSettings* deviceSettings,
	void* userContext )
{
	deviceSettings->d3d10.AutoDepthStencilFormat	= DXGI_FORMAT_D24_UNORM_S8_UINT;
	//deviceSettings->d3d10.AutoDepthStencilFormat	= DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
	return true;
}

bool CALLBACK OnDeviceRemoved( void* userContext )
{
	return true;
}



// Direct3D 10 callbacks
bool CALLBACK OnIsDeviceAcceptable10( 
    UINT adapter, 
	UINT output, 
	D3D10_DRIVER_TYPE deviceType, 
	DXGI_FORMAT backBufferFormat, 
	bool windowed, 
	void* userContext ) 
{
	return true;
}

HRESULT CALLBACK OnCreateDevice10( 
    ID3D10Device* d3dDevice, 
	const DXGI_SURFACE_DESC* backBufferSurfaceDesc,
	void* userContext )
{

	srand( 47110815 );
	HRESULT hr = E_FAIL;

	V_RETURN( D3DX10CreateFont( d3dDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, DEFAULT_QUALITY,
		DEFAULT_PITCH | FF_DONTCARE, L"Calibri", &gTextFont ) );

	V_RETURN( D3DX10CreateSprite( d3dDevice, 4096, &gTextSprites ) );
	gTextHelper = new CDXUTTextHelper(  gTextFont, gTextSprites, 16 );


	InitCameras();
	InitRenderPaths();
	V_RETURN( LoadWoodBox( d3dDevice ) );
	V_RETURN( LoadShadowCaster( d3dDevice ) );
	V_RETURN( LoadEffects( d3dDevice ) );
	V_RETURN( BindShaderVariables() );
	V_RETURN( CreateInputLayout( d3dDevice ) );
	
	// create a depth buffer
	D3D10_TEXTURE2D_DESC td;
	td.ArraySize	= 1;
	td.BindFlags	= D3D10_BIND_DEPTH_STENCIL;
	td.Width		= HIB_WIDTH;
	td.Height		= HIB_HEIGHT;
	td.CPUAccessFlags=0;
	td.Format		= DXGI_FORMAT_D24_UNORM_S8_UINT;
	td.MipLevels	= 0;
	td.MiscFlags	= 0;
	td.SampleDesc.Count	= 1;
	td.SampleDesc.Quality = 0;
	td.Usage		= D3D10_USAGE_DEFAULT;

	ID3D10Texture2D* depthTexture = NULL;
	V_RETURN( d3dDevice->CreateTexture2D( &td, NULL, &depthTexture ) );

	D3D10_DEPTH_STENCIL_VIEW_DESC dsv;
	dsv.Format		= td.Format;
	dsv.ViewDimension= D3D10_DSV_DIMENSION_TEXTURE2D;
	dsv.Texture2D.MipSlice=0;
	
	V_RETURN( d3dDevice->CreateDepthStencilView( depthTexture, &dsv, &gOcclusionDepthBuffer ) );

	SAFE_RELEASE( depthTexture );

	// init item buffer
	gItemBuffer = new HierarchicalItemBuffer();
	float hx = ceilf( sqrtf( MAX_CASTER_COUNT ) );

	gItemBuffer->Init( d3dDevice, HIB_WIDTH, HIB_HEIGHT, (size_t)hx, (size_t)hx );

	return S_OK;
}

HRESULT CALLBACK OnSwapChainResized10(
    ID3D10Device* d3dDevice,
	IDXGISwapChain* swapChain,
	const DXGI_SURFACE_DESC* backBufferSurfaceDesc,
	void* userContext )
{


	const UINT w = backBufferSurfaceDesc->Width;
	const UINT h = backBufferSurfaceDesc->Height;
	const float a= float(w) / float(h);

	gViewerCam.SetViewport( w, h );
	gViewerCam.SetPerspective( a, gViewerCam.fovy, gViewerCam.znear, gViewerCam.zfar );

	return S_OK;
}

void CALLBACK OnFrameRender10(
    ID3D10Device* d3dDevice, 
	double time, 
	float elapsedTime, 
	void* userContext )
{
	
	float angle = gLightAngle;

	static float frameCount = 0;
	static float ttime = 0.f;
	if( frameCount >= 50 ) {
		float fps = frameCount / ttime;
		frameCount = 0;
		ttime = 0;
		TCHAR txt[ 256 ];
		StringCchPrintf( txt, 256, L"%s@ %f fps", gRenderPaths[gActiveRenderPath].funcName, fps );
		SetWindowText( DXUTGetHWND(), txt );
	}
	else {
		frameCount++;
		ttime += elapsedTime;
	}
	// render target setup
	d3dDevice->ClearRenderTargetView( DXUTGetD3D10RenderTargetView(), D3DXCOLOR(0,0,0,0) );
	d3dDevice->ClearDepthStencilView( DXUTGetD3D10DepthStencilView(), D3D10_CLEAR_DEPTH | D3D10_CLEAR_STENCIL, 1.f, 0 );
	ID3D10RenderTargetView* rtv = DXUTGetD3D10RenderTargetView();
	d3dDevice->OMSetRenderTargets( 1, &rtv, DXUTGetD3D10DepthStencilView() );


	if( gActiveRenderPath != 3 )
		int stop = 10;
	gRenderPaths[gActiveRenderPath].renderFunc( d3dDevice );

	d3dDevice->VSSetShader( NULL );
	d3dDevice->GSSetShader( NULL );
	d3dDevice->PSSetShader( NULL );
	//GUI.g_HUD.OnRender( elapsedTime );

	TCHAR textStorage[ 128 ];

	gTextHelper->Begin();
	gTextHelper->SetInsertionPos( 2, 2 );
	gTextHelper->SetForegroundColor( D3DXCOLOR( 1, 1, 0, 1 ) );
	gTextHelper->DrawTextLine( DXUTGetFrameStats() );
	gTextHelper->DrawTextLine( DXUTGetDeviceStats() );
	StringCchPrintf( textStorage, LENGTHOF( textStorage ), L"Caster Count: %i", gActiveCasters );
	gTextHelper->DrawTextLine( textStorage );
	gTextHelper->End();
}



void CALLBACK OnSwapChainReleasing10( void* userContext )
{
}

void CALLBACK OnDestroyDevice10( void* userContext )
{
	// free resources
	FreeWoodBoxResources();
	FreeShadowCasterResources();
	FreeEffectResources();

	SAFE_RELEASE( gOcclusionDepthBuffer );
	gItemBuffer->Destroy();
	SAFE_DELETE( gItemBuffer );

	SAFE_DELETE( gTextHelper );
	SAFE_RELEASE( gTextFont );
	SAFE_RELEASE( gTextSprites );


}

int WINAPI WinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd )
{
	DXUTSetCallbackFrameMove( OnFrameMove );
	DXUTSetCallbackKeyboard( OnKeyboard );
	DXUTSetCallbackMouse( OnMouse );
	DXUTSetCallbackMsgProc( OnMsgProc );
	DXUTSetCallbackDeviceChanging( OnModifyDeviceSettings );
	DXUTSetCallbackDeviceRemoved( OnDeviceRemoved );

	DXUTSetCallbackD3D10DeviceCreated( OnCreateDevice10 );
	DXUTSetCallbackD3D10DeviceAcceptable( OnIsDeviceAcceptable10 );
	DXUTSetCallbackD3D10DeviceDestroyed( OnDestroyDevice10 );
	DXUTSetCallbackD3D10FrameRender( OnFrameRender10 );
	DXUTSetCallbackD3D10SwapChainReleasing( OnSwapChainReleasing10 );
	DXUTSetCallbackD3D10SwapChainResized( OnSwapChainResized10 );


	

	DXUTInit( true, true, NULL, true );
	DXUTSetCursorSettings( true, false );
	DXUTCreateDevice( true, 1280, 720 );
	DXUTCreateWindow( TEXT("Hierarchical Item Buffer Demo10"), hInstance );
	DXUTMainLoop( 0 );
	return 0;
}

