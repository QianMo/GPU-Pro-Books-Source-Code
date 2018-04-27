#include "Precompiled.h"

#include "Providers/Src/EffectDefine.h"
#include "Providers/Src/EffectKey.h"

#include "D3D10EffectIncludes.h"

namespace Mod
{
	bool D3D10CompileEffectImpl( const Bytes& shlangCode, const EffectKey::Defines& defines, UINT effectCompileFlags, Bytes& oCode, String& oErrors )
	{

		ID3D10Blob* shader;
		ID3D10Blob* errors;

		HRESULT hr;

		D3D10EffectIncludes includes;

		Types< D3D10_SHADER_MACRO > :: Vec macros( defines.empty() ? 0 : defines.size() + 1 );

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

		hr = D3DX10CompileFromMemory(	(const char*)&shlangCode[0], (SIZE_T)shlangCode.GetSize(), NULL, macros.empty() ?  NULL : &macros[0], &includes, NULL, "fx_4_0", 
										D3D10_SHADER_ENABLE_STRICTNESS, effectCompileFlags, NULL, &shader, &errors, &hr );

		ComPtr<ID3D10Blob> shader_keeper( shader ); shader_keeper;
		ComPtr<ID3D10Blob> errors_keeper( errors ); errors_keeper;

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
			SIZE_T size = shader->GetBufferSize();
			oCode.Resize( size );
			std::copy( (UINT8*)shader->GetBufferPointer(), (UINT8*)shader->GetBufferPointer() + size, &oCode[0] );

			return true;
		}
	}
}