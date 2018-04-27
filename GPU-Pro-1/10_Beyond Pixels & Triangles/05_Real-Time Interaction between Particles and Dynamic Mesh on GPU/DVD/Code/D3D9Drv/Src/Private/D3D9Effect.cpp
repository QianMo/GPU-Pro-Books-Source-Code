#include "Precompiled.h"

#include "Math/Src/Types.h"
#include "Wrap3D/Src/EffectConfig.h"

#include "D3D9EffectVarBindImpl.h"

#include "D3D9EffectPool.h"
#include "D3D9Effect.h"

namespace Mod
{
	using namespace Math;

	//------------------------------------------------------------------------
	D3D9Effect::D3D9Effect( const EffectConfig& cfg, IDirect3DDevice9* dev ) : 
	Parent( cfg )
	{
		ID3DXEffectPool* pool( NULL );

		if( cfg.pool )
		{
			pool = &*static_cast<D3D9EffectPool&>(*cfg.pool).GetResource();
		}

		ID3DXEffect* eff;

		ID3DXBuffer* errors;

		HRESULT hr = D3DXCreateEffect( dev, cfg.code.GetRawPtr(), (UINT32)cfg.code.GetSize(), NULL, NULL, 0, pool, &eff, &errors );

		if( hr != D3D_OK )
		{
			String str;
			if( errors )
			{
				str = ToString( AnsiString( (char*)errors->GetBufferPointer() ) );
			}
			else
			{
				str = L" unknown error";
			}

			MD_THROW( L"D3D9Effect::D3D9Effect: couldnt create: " + str );
		}

		mResource.set( eff );

		SetTechniqueImpl( "main" );

	}

	//------------------------------------------------------------------------
	/*virtual*/
	D3D9Effect::~D3D9Effect()
	{
	}

	//------------------------------------------------------------------------

	void
	D3D9Effect::SetStateManager( ID3DXEffectStateManager* manager )
	{
		MD_D3DV( mResource->SetStateManager ( manager ) );
	}

	//------------------------------------------------------------------------

	D3D9Effect::D3D9Effect( const EffectConfig& cfg, ResourcePtr eff ) :
	Parent( cfg )
	{
		mResource = eff;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	void
	D3D9Effect::SetTechniqueImpl( const AnsiString& name ) /*OVERRIDE*/
	{
		mCurrentTechnique.techHandle	= mResource->GetTechniqueByName( name.c_str() );
		MD_FERROR_ON_FALSE( mCurrentTechnique.techHandle );

		D3DXTECHNIQUE_DESC desc;
		MD_D3DV( mResource->GetTechniqueDesc( mCurrentTechnique.techHandle, &desc ) );

		mCurrentTechnique.numPasses		= desc.Passes;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	bool
	D3D9Effect::BindImpl( UINT32 passNum ) /*OVERRIDE*/
	{
		MD_ASSERT( passNum <= mCurrentTechnique.numPasses );

		if( passNum == mCurrentTechnique.numPasses )
		{
			mResource->EndPass();
			mResource->End();

			return false;
		}

		if( !passNum )
		{
			UINT passCount;
			MD_D3DV( mResource->Begin( &passCount, D3DXFX_DONOTSAVESTATE | D3DXFX_DONOTSAVESAMPLERSTATE | D3DXFX_DONOTSAVESHADERSTATE ) );
		}

		D3DXHANDLE passHandle = mResource->GetPass( mCurrentTechnique.techHandle, passNum );
		MD_ASSERT( passHandle );

		D3DXPASS_DESC pdesc;
		MD_D3DV( mResource->GetPassDesc( passHandle, &pdesc ) );

		bool draw = true;
		// state setting pass, no drawing
		if( !pdesc.pVertexShaderFunction )
			draw = false;
		
		if( passNum )
		{
			MD_D3DV( mResource->EndPass() );
		}

		mResource->BeginPass( passNum );

		if( !draw )
		{
			mResource->EndPass();
			mResource->End();
		}

		return draw;
	}

	//------------------------------------------------------------------------
	/*virtual*/

	namespace
	{
		D3D9EffectVarBindConfig CreateVarBindConfig( D3DXHANDLE bind, const ComPtr<ID3DXEffect>& eff, const AnsiString& name, VarType::Type type )
		{
			D3D9EffectVarBindConfig cfg;

			cfg.bind	= bind;
			cfg.effect	= eff;
			cfg.name	= name;
			cfg.type	= type;

			return cfg;
		}
	}

	EffectVarBindPtr
	D3D9Effect::GetVarBindImpl( const CodeString& name ) /*OVERRIDE*/
	{
		D3DXHANDLE paramHandle = mResource->GetParameterByName( NULL, name.c_str() );

		if( paramHandle )
		{
			D3DXPARAMETER_DESC desc;
			mResource->GetParameterDesc( paramHandle, &desc );

			switch( desc.Class )
			{
			case D3DXPC_SCALAR :
				switch( desc.Type )
				{
				case D3DXPT_FLOAT :
					if( desc.Elements )
						return EffectVarBindPtr( new D3D9EffectFloatVecVarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT_VEC) ) );
					else
						return EffectVarBindPtr( new D3D9EffectFloatVarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT) ) );

				case D3DXPT_INT:
					if( desc.Elements )
						return EffectVarBindPtr( new D3D9EffectIntVecVarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::INT_VEC) ) );
					else
						return EffectVarBindPtr( new D3D9EffectIntVarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::INT) ) );

				default:
					MD_FERROR( L"D3D9Effect::GetVarBindImpl: variable type not supported!");
				}
				break;

			case D3DXPC_VECTOR:
				switch( desc.Type )
				{
				case D3DXPT_FLOAT :
					switch( desc.Columns )
					{
					case 2:
						if( desc.Elements )
							return EffectVarBindPtr( new D3D9EffectFloat2VecVarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT2_VEC ) ) );
						else
							return EffectVarBindPtr( new D3D9EffectFloat2VarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT2 ) ) );
					case 3:
						if( desc.Elements )
							return EffectVarBindPtr( new D3D9EffectFloat3VecVarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT3_VEC ) ) );
						else
							return EffectVarBindPtr( new D3D9EffectFloat3VarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT3 ) ) );
					case 4:
						if( desc.Elements )
							return EffectVarBindPtr( new D3D9EffectFloat4VecVarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT4_VEC ) ) );
						else
							return EffectVarBindPtr( new D3D9EffectFloat4VarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT4 ) ) );
					default:
						MD_FERROR( L"D3D9Effect::GetVarBindImpl: variable type not supported!");
					}
					break;
				case D3DXPT_INT :
					switch( desc.Columns )
					{
					case 2:
						return EffectVarBindPtr( new D3D9EffectInt2VarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::INT2 ) ) );
					case 3:
						return EffectVarBindPtr( new D3D9EffectInt3VarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::INT3 ) ) );
					case 4:
						return EffectVarBindPtr( new D3D9EffectInt4VarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::INT4 ) ) );
					default:
						MD_FERROR( L"D3D9Effect::GetVarBindImpl: variable type not supported!");
					}
				}
				break;
			case D3DXPC_MATRIX_ROWS :
				switch( desc.Type )
				{
				case D3DXPT_FLOAT :
					switch( desc.Columns )
					{
					case 2:
						MD_FERROR_ON_TRUE( desc.Elements ); // arrays not supported yet
						MD_ASSERT( desc.Rows == 2 );
						return EffectVarBindPtr( new D3D9EffectFloat2x2VarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT2x2 ) ) );
						break;
					case 3:
						if( desc.Rows == 3 )
						{
							MD_FERROR_ON_TRUE( desc.Elements ); // arrays not supported yet
							return EffectVarBindPtr( new D3D9EffectFloat3x3VarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT3x3 ) ) );
						}
						else
						if( desc.Rows == 4)
						{
							if( desc.Elements )
								return EffectVarBindPtr( new D3D9EffectFloat3x4VecVarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT3x4_VEC ) ) );
							else
								return EffectVarBindPtr( new D3D9EffectFloat3x4VarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT3x4 ) ) );
						}
						else
							MD_FERROR( L"Row/Column configuration not supported yet!" );
						break;
					case 4:
						MD_ASSERT( desc.Rows == 4 );
						if( desc.Elements )
							return EffectVarBindPtr( new D3D9EffectFloat4x4VecVarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT4x4_VEC ) ) );
						else
							return EffectVarBindPtr( new D3D9EffectFloat4x4VarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::FLOAT4x4 ) ) );
						break;
					}
				}
				break;
			case D3DXPC_OBJECT :
				switch( desc.Type )
				{
				case D3DXPT_TEXTURE:
				case D3DXPT_TEXTURE1D:
				case D3DXPT_TEXTURE2D:
				case D3DXPT_TEXTURE3D:
				case D3DXPT_TEXTURECUBE:
					return EffectVarBindPtr( new D3D9EffectShaderResourceVarBind( CreateVarBindConfig(paramHandle, mResource, name, VarType::SHADER_RESOURCE ) ) );
					break;
				default:
					MD_FERROR( L"D3D9Effect::GetVarBindImpl: object type not supported!");
				}
				break;
			default:
				MD_FERROR( L"D3D9Effect::GetVarBindImpl: variable type not supported!");			
			}
		}

		return EffectVarBindPtr();
	}

	//------------------------------------------------------------------------
	/*virtual*/

	UINT32
	D3D9Effect::GetNumPassesImpl() const
	{
		return mCurrentTechnique.numPasses;
	}

}