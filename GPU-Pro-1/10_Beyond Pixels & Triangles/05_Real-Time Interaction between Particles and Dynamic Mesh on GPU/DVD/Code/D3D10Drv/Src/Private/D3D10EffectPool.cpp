#include "Precompiled.h"

#include "Wrap3D/Src/EffectPoolConfig.h"
#include "Wrap3D/Src/EffectConfig.h"

#include "D3D10EffectPool.h"
#include "D3D10Effect.h"
#include "D3D10Exception.h"

#include "D3D10ImportedFunctions.h"

namespace Mod
{
	D3D10EffectPool::D3D10EffectPool( const EffectPoolConfig& cfg, ID3D10Device* dev ) :
	Parent( cfg )
	{
		UINT HLSLFlags = D3D10_SHADER_ENABLE_STRICTNESS;
		UINT FXFlags = 0;

		// Create effect pool
		{
			ID3D10EffectPool* effPool( NULL );
			ID3D10Blob* errors( NULL );

			HRESULT hr = D3DX10CreateEffectPoolFromMemory(	cfg.code.GetRawPtr(), (SIZE_T)cfg.code.GetSize(), NULL, NULL, NULL, 
															"fx_4_0", HLSLFlags, FXFlags, dev, NULL, &effPool, 
															&errors, NULL );

			if( hr != S_OK )
			{
				if( errors && errors->GetBufferPointer() )
				{
					MD_THROW( L"D3D10EffectPool::D3D10EffectPool: compile error: " + ToString( AnsiString( (char*)errors->GetBufferPointer() ) ) );
				}
				else
					MD_FERROR( L"Unknown error during compilation.");
			}

			mResource.set( effPool );
		}

		// create effect object for base
		{			
			EffectConfig ecfg;
			ecfg.SetCode( cfg.code );

			ID3D10Effect* d3deff = mResource->AsEffect();
			d3deff->AddRef();
			EffectPtr eff( new D3D10Effect( ecfg, D3D10Effect::ResourcePtr( d3deff ) ) );

			SetEffect( eff );
		}
		
	}

	//------------------------------------------------------------------------

	D3D10EffectPool::~D3D10EffectPool()
	{

	}

	//------------------------------------------------------------------------

	D3D10EffectPool::ResourceType
	D3D10EffectPool::GetResource() const
	{
		return mResource;
	}

}