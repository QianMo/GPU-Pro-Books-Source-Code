#include "Precompiled.h"

#include "D3D9Device.h"
#include "D3D9EffectStateManagerConfig.h"
#include "D3D9EffectStateManager.h"
#include "D3D9TextureCoordinator.h"

#define MD_NAMESPACE D3D9EffectStateManagerNS
#include "ConfigurableImpl.cpp.h"

namespace Mod
{
	D3D9EffectStateManager::D3D9EffectStateManager( const D3D9EffectStateManagerConfig& cfg ) :
	Parent( cfg ),
	mRefCount( 1 )
	{
		MD_D3DV( cfg.device->GetPixelShader( &mPixelShader) );
		MD_D3DV( cfg.device->GetVertexShader( &mVertexShader ) );
	}

	//------------------------------------------------------------------------

	D3D9EffectStateManager::~D3D9EffectStateManager() 
	{
	}

#pragma warning( push )
#pragma warning( disable: 4297 )

	//------------------------------------------------------------------------
	/*virtual*/

	HRESULT
	D3D9EffectStateManager::QueryInterface(THIS_ REFIID iid, LPVOID *ppv) /*OVERRIDE*/
	{
		iid;
		*ppv = NULL;
		return E_NOINTERFACE;		
	}

	//------------------------------------------------------------------------
	/*virtual*/

	ULONG 
	D3D9EffectStateManager::AddRef(THIS) /*OVERRIDE*/
	{
		return ++mRefCount;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	ULONG 
	D3D9EffectStateManager::Release(THIS) /*OVERRIDE*/
	{
		mRefCount--;

		UINT refCount = mRefCount;

		if(!mRefCount)
			delete this;

		return refCount;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT
	D3D9EffectStateManager::SetTransform(THIS_ D3DTRANSFORMSTATETYPE State, CONST D3DMATRIX *pMatrix) /*OVERRIDE*/
	{
		State, pMatrix;
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT	
	D3D9EffectStateManager::SetMaterial(THIS_ CONST D3DMATERIAL9 *pMaterial) /*OVERRIDE*/
	{
		pMaterial;
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT	
	D3D9EffectStateManager::SetLight(THIS_ DWORD Index, CONST D3DLIGHT9 *pLight) /*OVERRIDE*/
	{
		Index, pLight;
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT
	D3D9EffectStateManager::LightEnable(THIS_ DWORD Index, BOOL Enable) /*OVERRIDE*/
	{
		Index, Enable;
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	HRESULT
	D3D9EffectStateManager::SetRenderState(THIS_ D3DRENDERSTATETYPE State, DWORD Value) /*OVERRIDE*/
	{
		MD_D3DV( d3d9dev()->SetRenderState( State, Value ) );
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT
	D3D9EffectStateManager::SetTexture(THIS_ DWORD Stage, LPDIRECT3DBASETEXTURE9 pTexture) /*OVERRIDE*/
	{
		GetConfig().texCoordinator->SetTexture( Stage, pTexture );
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT
	D3D9EffectStateManager::SetTextureStageState(THIS_ DWORD Stage, D3DTEXTURESTAGESTATETYPE Type, DWORD Value) /*OVERRIDE*/
	{
		Stage, Type, Value;
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	HRESULT
	D3D9EffectStateManager::SetSamplerState(THIS_ DWORD Sampler, D3DSAMPLERSTATETYPE Type, DWORD Value) /*OVERRIDE*/
	{
		MD_D3DV( d3d9dev()->SetSamplerState( Sampler, Type, Value ) );
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT
	D3D9EffectStateManager::SetNPatchMode(THIS_ FLOAT NumSegments) /*OVERRIDE*/
	{
		NumSegments;
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	HRESULT
	D3D9EffectStateManager::SetFVF(THIS_ DWORD FVF) /*OVERRIDE*/
	{
		FVF;
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	HRESULT
	D3D9EffectStateManager::SetVertexShader(THIS_ LPDIRECT3DVERTEXSHADER9 pShader) /*OVERRIDE*/
	{
		if( pShader != mVertexShader )
		{
			MD_D3DV( d3d9dev()->SetVertexShader( pShader ) );
			mVertexShader = pShader;
		}
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	HRESULT
	D3D9EffectStateManager::SetVertexShaderConstantF(THIS_ UINT RegisterIndex, CONST FLOAT *pConstantData, UINT RegisterCount) /*OVERRIDE*/
	{
		MD_D3DV( d3d9dev()->SetVertexShaderConstantF( RegisterIndex, pConstantData, RegisterCount ) );
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT
	D3D9EffectStateManager::SetVertexShaderConstantI(THIS_ UINT RegisterIndex, CONST INT *pConstantData, UINT RegisterCount) /*OVERRIDE*/
	{
		MD_D3DV( d3d9dev()->SetVertexShaderConstantI( RegisterIndex, pConstantData, RegisterCount ) );
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT
	D3D9EffectStateManager::SetVertexShaderConstantB(THIS_ UINT RegisterIndex, CONST BOOL *pConstantData, UINT RegisterCount) /*OVERRIDE*/
	{
		MD_D3DV( d3d9dev()->SetVertexShaderConstantB( RegisterIndex, pConstantData, RegisterCount ) );
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT
	D3D9EffectStateManager::SetPixelShader(THIS_ LPDIRECT3DPIXELSHADER9 pShader) /*OVERRIDE*/
	{
		if( pShader != mPixelShader )
		{
			MD_D3DV( d3d9dev()->SetPixelShader( pShader ) );
			mPixelShader = pShader;
		}
		return S_OK;		
	}

	//------------------------------------------------------------------------
	/*virtual*/

	HRESULT
	D3D9EffectStateManager::SetPixelShaderConstantF(THIS_ UINT RegisterIndex, CONST FLOAT *pConstantData, UINT RegisterCount) /*OVERRIDE*/
	{
		MD_D3DV( d3d9dev()->SetPixelShaderConstantF( RegisterIndex, pConstantData, RegisterCount ) );
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT
	D3D9EffectStateManager::SetPixelShaderConstantI(THIS_ UINT RegisterIndex, CONST INT *pConstantData, UINT RegisterCount) /*OVERRIDE*/
	{
		MD_D3DV( d3d9dev()->SetPixelShaderConstantI( RegisterIndex, pConstantData, RegisterCount ) );
		return S_OK;
	}

	//------------------------------------------------------------------------
	/*virtual*/
	
	HRESULT
	D3D9EffectStateManager::SetPixelShaderConstantB(THIS_ UINT RegisterIndex, CONST BOOL *pConstantData, UINT RegisterCount) /*OVERRIDE*/
	{
		MD_D3DV( d3d9dev()->SetPixelShaderConstantB( RegisterIndex, pConstantData, RegisterCount ) );
		return S_OK;
	}

#pragma warning (pop)

	//------------------------------------------------------------------------

	IDirect3DDevice9*
	D3D9EffectStateManager::d3d9dev() const
	{
		return &*GetConfig().device;
	}

}