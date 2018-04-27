#include "Precompiled.h"

#include "Wrap3D/Src/EffectConfig.h"

#include "D3D10EffectPool.h"
#include "D3D10Effect.h"
#include "D3D10EffectVarBindConfig.h"
#include "D3D10EffectVarBindImpl.h"
#include "D3D10Exception.h"

namespace Mod
{

	namespace
	{
		void ExtractTechnique( D3D10Effect::Technique& oTech, const AnsiString& techName, D3D10Effect::ResourcePtr res )
		{
			ID3D10EffectTechnique* tech = res->GetTechniqueByName( techName.c_str() );

			if( tech->IsValid() )
			{
				D3D10_TECHNIQUE_DESC desc;
				tech->GetDesc( &desc );

				oTech.numPasses	= desc.Passes;
				oTech.tech		= tech;
			}
		}

		void ExtractMainTechnique( D3D10Effect::Technique& oTech, D3D10Effect::ResourcePtr res )
		{
			ExtractTechnique( oTech, "main", res );
		}
	}

	//------------------------------------------------------------------------

	D3D10Effect::D3D10Effect( const EffectConfig& cfg, ID3D10Device* dev ) : 
	Parent( cfg )
	{

		// Create effect
		{

			ID3D10Effect* eff;
			ID3D10Blob* errors;
			HRESULT hr;

			UINT FXFlags = 0;
			ID3D10EffectPool* pool = NULL;

			if( cfg.pool )
			{
				pool	= &*static_cast<D3D10EffectPool*>(cfg.pool.get())->GetResource();
				MD_FERROR_ON_FALSE( pool->AsEffect()->IsValid() && pool->AsEffect()->IsPool() );
				FXFlags	= D3D10_EFFECT_COMPILE_CHILD_EFFECT;
			}

			hr =	D3DX10CreateEffectFromMemory(	cfg.code.GetRawPtr(), (SIZE_T)cfg.code.GetSize(),
													NULL,
													NULL, NULL, 
													"fx_4_0", 
													D3D10_SHADER_ENABLE_STRICTNESS,
													FXFlags, 
													&*dev, pool, NULL, &eff, &errors, &hr );

			if( hr != S_OK )
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

				MD_THROW( L"D3D10Effect::D3D10Effect: couldnt create: " + str );
			}

			mResource.set( eff );
		}

		ExtractMainTechnique( mCurrentTechnique, mResource );
		CheckTechnique();
	}

	//------------------------------------------------------------------------

	D3D10Effect::~D3D10Effect()
	{

	}

	//------------------------------------------------------------------------

	D3D10Effect::D3D10Effect( const EffectConfig& cfg, ResourcePtr eff ) :
	Parent( cfg ),
	mResource( eff )
	{
		MD_FERROR_ON_FALSE( eff->IsValid() );
		ExtractMainTechnique( mCurrentTechnique, mResource );
	}

	//------------------------------------------------------------------------

	void
	D3D10Effect::FillInputSignatureParams( const void*& oInputSignature, UINT32& oByteCodeLength ) const
	{
		D3D10_PASS_DESC passDesc;
		mCurrentTechnique->GetPassByIndex(0)->GetDesc( &passDesc );

		oByteCodeLength = static_cast<UINT32>(passDesc.IAInputSignatureSize);
		oInputSignature = passDesc.pIAInputSignature;
	}

	//------------------------------------------------------------------------

	void
	D3D10Effect::SetTechniqueImpl( const AnsiString& name )
	{
		mCurrentTechnique.tech = NULL;
		ExtractTechnique( mCurrentTechnique, name, mResource );
		CheckTechnique();
	}

	//------------------------------------------------------------------------

	bool
	D3D10Effect::BindImpl( UINT32 passNum )
	{
		Technique &tech = mCurrentTechnique;

		// too much is never good.
		MD_ASSERT( passNum <= tech.numPasses );

		if( tech.tech && passNum < tech.numPasses )
		{						
			ID3D10EffectPass* pass = tech->GetPassByIndex( passNum );
			MD_ASSERT( pass->IsValid() );

			pass->Apply( 0 );

			D3D10_PASS_SHADER_DESC desc;
			D3D10_THROW_IF( pass->GetVertexShaderDesc(&desc) );
			// no valid pass without a vertex shader, hence this is a 'state restore pass' - no drawing
			if( !desc.pShaderVariable->IsValid() )
				return false;
			else
				return true;
		}
		else
			return false;
		
	}

	//------------------------------------------------------------------------

	namespace
	{

		typedef EffectVarBindConfig	VBCfg;
		typedef VarType::Type		VBCfgType;


		template <typename T, typename U>
		typename D3D10EffectVarBindConfig<T>
		CreateVarBindConfig( ID3D10EffectVariable* var, const D3D10Effect::CodeString& name, U ( MD_D3D_CALLING_CONV ID3D10EffectVariable::*convFunc)(), VBCfgType type )
		{
			D3D10EffectVarBindConfig<T> res( (var->*convFunc)() );
			res.name = name;
			res.type = type;
			return res;
		}

		template <typename T>
		typename D3D10EffectVarBindConfig<T>
		CreateScalarVarBindConfig( ID3D10EffectVariable* var, const D3D10Effect::CodeString& name, VBCfgType type )
		{
			return CreateVarBindConfig<T> ( var, name, &ID3D10EffectVariable::AsScalar, type );
		}

		template <typename T>
		typename D3D10EffectVarBindConfig<T>
		CreateVectorVarBindConfig( ID3D10EffectVariable* var, const D3D10Effect::CodeString& name, VBCfgType type )
		{
			return CreateVarBindConfig<T> ( var, name, &ID3D10EffectVariable::AsVector, type );
		}

		template <typename T>
		typename D3D10EffectVarBindConfig<T>
		CreateMatrixVarBindConfig( ID3D10EffectVariable* var, const D3D10Effect::CodeString& name, VBCfgType type )
		{
			return CreateVarBindConfig<T> ( var, name, &ID3D10EffectVariable::AsMatrix, type );
		}

		template <typename T>
		typename D3D10EffectVarBindConfig<T>
		CreateShaderResourceVarBindConfig( ID3D10EffectVariable* var, const D3D10Effect::CodeString& name, VBCfgType type )
		{
			return CreateVarBindConfig<T> ( var, name, &ID3D10EffectVariable::AsShaderResource, type );
		}

		D3D10EffectVarBindConfig<Buffer>
		CreateCBufferVarBindConfig( ID3D10EffectVariable* var, const D3D10Effect::CodeString& name, VBCfgType type )
		{
			return CreateVarBindConfig<Buffer> ( var, name, &ID3D10EffectVariable::AsConstantBuffer, type );
		}

	}

	EffectVarBindPtr
	D3D10Effect::GetVarBindImpl( const CodeString& name )
	{
		using namespace Math;

		ID3D10EffectVariable* var = mResource->GetVariableByName( name.c_str() );

		// if failed, try as a constant buffer
		if( !var->IsValid() )
			var = mResource->GetConstantBufferByName( name.c_str() );

		if( var->IsValid() )
		{
			ID3D10EffectType* etype = var->GetType();

			D3D10_EFFECT_TYPE_DESC desc;
			etype->GetDesc( &desc );

			switch( desc.Class )
			{
			case D3D10_SVC_SCALAR:
				switch( desc.Type )
				{
				case D3D10_SVT_FLOAT:
					if( desc.Elements )
						return EffectVarBindPtr( new D3D10EffectFloatVecVarBind( CreateScalarVarBindConfig<float_vec>(var, name, VarType::FLOAT_VEC) ) );
					else
						return EffectVarBindPtr( new D3D10EffectFloatVarBind( CreateScalarVarBindConfig<float>(var, name, VarType::FLOAT) ) );
				case D3D10_SVT_UINT:
					if( desc.Elements )
						return EffectVarBindPtr( new D3D10EffectUIntVecVarBind( CreateScalarVarBindConfig<uint_vec>(var, name, VarType::UINT_VEC) ) );
					else
						return EffectVarBindPtr( new D3D10EffectUIntVarBind( CreateScalarVarBindConfig<UINT32>(var, name, VarType::UINT) ) );
				case D3D10_SVT_INT:
					if( desc.Elements )
						return EffectVarBindPtr( new D3D10EffectIntVecVarBind( CreateScalarVarBindConfig<int_vec>(var, name, VarType::INT_VEC) ) );
					else
						return EffectVarBindPtr( new D3D10EffectIntVarBind( CreateScalarVarBindConfig<INT32>(var, name, VarType::INT) ) );
				default:
					MD_FERROR( L"D3D10Effect::GetVarBindImpl: variable type not supported!");
				}
				break;
			case D3D10_SVC_VECTOR:
				switch( desc.Type )
				{
				case D3D10_SVT_FLOAT:
					switch( desc.Columns )
					{
					case 2:
						if( desc.Elements )
							return EffectVarBindPtr( new D3D10EffectFloat2VecVarBind( CreateVectorVarBindConfig<float2_vec>(var, name, VarType::FLOAT2_VEC ) ) );
						else
							return EffectVarBindPtr( new D3D10EffectFloat2VarBind( CreateVectorVarBindConfig<float2>(var, name, VarType::FLOAT2 ) ) );
					case 3:
						if( desc.Elements )
							return EffectVarBindPtr( new D3D10EffectFloat3VecVarBind( CreateVectorVarBindConfig<float3_vec>(var, name, VarType::FLOAT3_VEC ) ) );
						else
							return EffectVarBindPtr( new D3D10EffectFloat3VarBind( CreateVectorVarBindConfig<float3>(var, name, VarType::FLOAT3 ) ) );
					case 4:
						if( desc.Elements )
							return EffectVarBindPtr( new D3D10EffectFloat4VecVarBind( CreateVectorVarBindConfig<float4_vec>(var, name, VarType::FLOAT4_VEC ) ) );
						else
							return EffectVarBindPtr( new D3D10EffectFloat4VarBind( CreateVectorVarBindConfig<float4>(var, name, VarType::FLOAT4 ) ) );
					default:
						MD_FERROR( L"D3D10Effect::GetVarBindImpl: variable type not supported!");
					}
					break;
				case D3D10_SVT_INT:
					switch( desc.Columns )
					{
					case 2:
						return EffectVarBindPtr( new D3D10EffectInt2VarBind( CreateVectorVarBindConfig<int2>(var, name, VarType::INT2 ) ) );
					case 3:
						return EffectVarBindPtr( new D3D10EffectInt3VarBind( CreateVectorVarBindConfig<int3>(var, name, VarType::INT3 ) ) );
					case 4:
						return EffectVarBindPtr( new D3D10EffectInt4VarBind( CreateVectorVarBindConfig<int4>(var, name, VarType::INT4 ) ) );
					default:
						MD_FERROR( L"D3D10Effect::GetVarBindImpl: variable type not supported!");
					}

				case D3D10_SVT_UINT:
					switch( desc.Columns )
					{
					case 2:
						return EffectVarBindPtr( new D3D10EffectUInt2VarBind( CreateVectorVarBindConfig<uint2>(var, name, VarType::UINT2 ) ) );
					case 3:
						return EffectVarBindPtr( new D3D10EffectUInt3VarBind( CreateVectorVarBindConfig<uint3>(var, name, VarType::UINT3 ) ) );
					case 4:
						return EffectVarBindPtr( new D3D10EffectUInt4VarBind( CreateVectorVarBindConfig<uint4>(var, name, VarType::UINT4 ) ) );
					default:
						MD_FERROR( L"D3D10Effect::GetVarBindImpl: variable type not supported!");
					}
				}
				break;
			case D3D10_SVC_MATRIX_COLUMNS:
				switch( desc.Type )
				{
				case D3D10_SVT_FLOAT:
					switch( desc.Columns )
					{
					case 2:
						MD_FERROR_ON_TRUE( desc.Elements ); // arrays not supported yet
						MD_FERROR_ON_FALSE( desc.Rows == 2 );
						return EffectVarBindPtr( new D3D10EffectFloat2x2VarBind( CreateMatrixVarBindConfig<float2x2>(var, name, VarType::FLOAT2x2 ) ) );
						break;
					case 3:
						if( desc.Rows == 3 )
						{
							MD_FERROR_ON_TRUE( desc.Elements ); // arrays not supported yet
							return EffectVarBindPtr( new D3D10EffectFloat3x3VarBind( CreateMatrixVarBindConfig<float3x3>(var, name, VarType::FLOAT3x3 ) ) );
						}
						else
						if( desc.Rows == 4)
						{
							if( desc.Elements )
								return EffectVarBindPtr( new D3D10EffectFloat3x4VecVarBind( CreateMatrixVarBindConfig<float3x4_vec>(var, name, VarType::FLOAT3x4_VEC ) ) );
							else
								return EffectVarBindPtr( new D3D10EffectFloat3x4VarBind( CreateMatrixVarBindConfig<float3x4>(var, name, VarType::FLOAT3x4 ) ) );
						}
						else
							MD_FERROR( L"Row/Column configuration not supported yet!" );
						break;
					case 4:
						MD_FERROR_ON_FALSE( desc.Rows == 4 );
						if( desc.Elements )
							return EffectVarBindPtr( new D3D10EffectFloat4x4VecVarBind( CreateMatrixVarBindConfig<float4x4_vec>(var, name, VarType::FLOAT4x4_VEC ) ) );
						else
							return EffectVarBindPtr( new D3D10EffectFloat4x4VarBind( CreateMatrixVarBindConfig<float4x4>(var, name, VarType::FLOAT4x4 ) ) );
						break;
					}
				}
				break;
			case D3D10_SVC_OBJECT:
				switch( desc.Type )
				{
				case D3D10_SVT_BUFFER:
				case D3D10_SVT_TEXTURE:
				case D3D10_SVT_TEXTURE1D:
				case D3D10_SVT_TEXTURE1DARRAY:
				case D3D10_SVT_TEXTURE2D:
				case D3D10_SVT_TEXTURE2DARRAY:
				case D3D10_SVT_TEXTURE2DMS:
				case D3D10_SVT_TEXTURE3D:
				case D3D10_SVT_TEXTURECUBE:
					return EffectVarBindPtr( new D3D10EffectShaderResourceVarBind( CreateShaderResourceVarBindConfig<ShaderResource>(var, name, VarType::SHADER_RESOURCE ) ) );
					break;
				case D3D10_SVT_CBUFFER:
					return EffectVarBindPtr( new D3D10EffectCBufferVarBind( CreateCBufferVarBindConfig(var, name, VarType::CONSTANT_BUFFER ) ) );
					break;
				default:
					MD_FERROR( L"D3D10Effect::GetVarBindImpl: object type not supported!");
				}
				break;
			default:
				MD_FERROR( L"D3D10Effect::GetVarBindImpl: variable type not supported!");
			}
		}

		return EffectVarBindPtr();
	}

	//------------------------------------------------------------------------

	UINT32
	D3D10Effect::GetNumPassesImpl() const
	{
		return mCurrentTechnique.numPasses;
	}

	//------------------------------------------------------------------------

	void
	D3D10Effect::CheckTechnique() const
	{
		MD_FERROR_ON_FALSE( mCurrentTechnique.tech && mCurrentTechnique.tech->IsValid() );
	}

	//------------------------------------------------------------------------

	D3D10Effect::Technique::Technique(  ) :
	numPasses( 0 ),
	tech( NULL )
	{
		
	}

	//------------------------------------------------------------------------

	D3D10Effect::Technique::~Technique()
	{

	}

	//------------------------------------------------------------------------

	ID3D10EffectTechnique*
	D3D10Effect::Technique::operator ->() const
	{
		return tech;
	}

	//------------------------------------------------------------------------	


}

