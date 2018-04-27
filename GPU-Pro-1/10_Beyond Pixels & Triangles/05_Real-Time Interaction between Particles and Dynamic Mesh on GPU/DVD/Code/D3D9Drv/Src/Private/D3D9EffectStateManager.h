#ifndef D3D9DRV_D3D9EFFECTSTATEMANAGER_H_INCLUDED
#define D3D9DRV_D3D9EFFECTSTATEMANAGER_H_INCLUDED

#include "Forw.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE D3D9EffectStateManagerNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class D3D9EffectStateManager :	public D3D9EffectStateManagerNS::ConfigurableImpl<D3D9EffectStateManagerConfig>,
									public ID3DXEffectStateManager
	{
		// types
	public:

		// constructors / destructors
	public:
		explicit D3D9EffectStateManager( const D3D9EffectStateManagerConfig& cfg );
		~D3D9EffectStateManager();
	
		// manipulation/ access
	public:

		// COM managment stuff
	public:

		STDMETHOD(QueryInterface)(THIS_ REFIID iid, LPVOID *ppv) OVERRIDE;
		STDMETHOD_(ULONG, AddRef)(THIS) OVERRIDE;
		STDMETHOD_(ULONG, Release)(THIS) OVERRIDE;

		// polymorphism
	private:

		STDMETHOD(SetTransform)(THIS_ D3DTRANSFORMSTATETYPE State, CONST D3DMATRIX *pMatrix) OVERRIDE;
		STDMETHOD(SetMaterial)(THIS_ CONST D3DMATERIAL9 *pMaterial) OVERRIDE;
		STDMETHOD(SetLight)(THIS_ DWORD Index, CONST D3DLIGHT9 *pLight) OVERRIDE;
		STDMETHOD(LightEnable)(THIS_ DWORD Index, BOOL Enable) OVERRIDE;
		STDMETHOD(SetRenderState)(THIS_ D3DRENDERSTATETYPE State, DWORD Value) OVERRIDE;
		STDMETHOD(SetTexture)(THIS_ DWORD Stage, LPDIRECT3DBASETEXTURE9 pTexture) OVERRIDE;
		STDMETHOD(SetTextureStageState)(THIS_ DWORD Stage, D3DTEXTURESTAGESTATETYPE Type, DWORD Value) OVERRIDE;
		STDMETHOD(SetSamplerState)(THIS_ DWORD Sampler, D3DSAMPLERSTATETYPE Type, DWORD Value) OVERRIDE;
		STDMETHOD(SetNPatchMode)(THIS_ FLOAT NumSegments) OVERRIDE;
		STDMETHOD(SetFVF)(THIS_ DWORD FVF) OVERRIDE;
		STDMETHOD(SetVertexShader)(THIS_ LPDIRECT3DVERTEXSHADER9 pShader) OVERRIDE;
		STDMETHOD(SetVertexShaderConstantF)(THIS_ UINT RegisterIndex, CONST FLOAT *pConstantData, UINT RegisterCount) OVERRIDE;
		STDMETHOD(SetVertexShaderConstantI)(THIS_ UINT RegisterIndex, CONST INT *pConstantData, UINT RegisterCount) OVERRIDE;
		STDMETHOD(SetVertexShaderConstantB)(THIS_ UINT RegisterIndex, CONST BOOL *pConstantData, UINT RegisterCount) OVERRIDE;
		STDMETHOD(SetPixelShader)(THIS_ LPDIRECT3DPIXELSHADER9 pShader) OVERRIDE;
		STDMETHOD(SetPixelShaderConstantF)(THIS_ UINT RegisterIndex, CONST FLOAT *pConstantData, UINT RegisterCount) OVERRIDE;
		STDMETHOD(SetPixelShaderConstantI)(THIS_ UINT RegisterIndex, CONST INT *pConstantData, UINT RegisterCount) OVERRIDE;
		STDMETHOD(SetPixelShaderConstantB)(THIS_ UINT RegisterIndex, CONST BOOL *pConstantData, UINT RegisterCount) OVERRIDE;

		// helpers
	private:
		IDirect3DDevice9* d3d9dev() const;

		// data
	private:
		UINT mRefCount;
		IDirect3DPixelShader9* mPixelShader;
		IDirect3DVertexShader9* mVertexShader;
	};
}

#endif