#include "Precompiled.h"

#include "Providers/Src/EffectDefine.h"
#include "Providers/Src/EffectKey.h"

#include "D3D9EffectIncludes.h"

namespace Mod
{
	bool D3D9CompileEffectImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, UINT effectCompileFlags, Bytes& oCode, String& oErrors )
	{
		ID3DXBuffer* effect;
		ID3DXBuffer* errors;

		D3D9EffectIncludes includes;

		Types< D3DXMACRO > :: Vec macros( defines.empty() ? 0 : defines.size() + 1 );

		if( !defines.empty() )
		{
			size_t ii = 0;
			for( EffectKey::Defines::const_iterator i = defines.begin(), e = defines.end(); i != e; ++i, ++ii )
			{
				macros[ ii ].Name		= (*i).name.c_str();
				macros[ ii ].Definition	= (*i).val.c_str();
			}

			macros[ ii ].Name		= NULL;
			macros[ ii ].Definition	= NULL;
		}

		ID3DXEffectCompiler* compiler;
		HRESULT hr;
		hr = D3DXCreateEffectCompiler( (const char*)&shlangCode[0], (UINT)shlangCode.GetSize(), macros.empty() ?  NULL : &macros[0], &includes, D3DXSHADER_PACKMATRIX_COLUMNMAJOR, &compiler, &errors );

		ComPtr< ID3DXBuffer > errors_keeper( errors );
		ComPtr< ID3DXEffectCompiler > compiler_keeper( compiler );

		if( hr != S_OK )
		{
			if( errors)
				oErrors = ToString( AnsiString( (char*) errors->GetBufferPointer() ) );
			else
				oErrors = L"Unknown errors!";

			return false;
		}
		else
		{
			hr = compiler->CompileEffect( effectCompileFlags, &effect, &errors );
			errors_keeper.set( errors );

			if( hr != S_OK )
			{
				if( errors )
					oErrors = ToString( AnsiString( (char*) errors->GetBufferPointer() ) );
				else
					oErrors = L"Unknown errors!";

				return false;
			}

			SIZE_T size = effect->GetBufferSize();
			oCode.Resize( size );
			std::copy( (UINT8*)effect->GetBufferPointer(), (UINT8*)effect->GetBufferPointer() + size, &oCode[0] );

			return true;
		}


	}
}