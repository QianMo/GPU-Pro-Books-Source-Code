//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------

#include <set>
#include <vector>
#include <sstream>

class CD3DShaderMacroHelper
{
public:
    CD3DShaderMacroHelper() : m_bIsFinalized(false) {}

    template<typename DefintionType>
	void AddShaderMacro( LPCSTR Name, DefintionType Definition )
	{
        assert( !m_bIsFinalized );
		std::ostringstream ss;
		ss << Definition;
		AddShaderMacro<LPCSTR>( Name, ss.str().c_str() );
	}

	template<>
	void AddShaderMacro( LPCSTR Name, LPCSTR Definition )
	{
        assert( !m_bIsFinalized );
		D3D_SHADER_MACRO NewMacro = 
		{
			Name,
			m_DefinitionsPull.insert(Definition).first->c_str()
		};
		m_Macroes.push_back(NewMacro);
	}

	template<>
	void AddShaderMacro( LPCSTR Name, bool Definition )
	{
        assert( !m_bIsFinalized );
		AddShaderMacro( Name, Definition ? "1" : "0");
	}
	
    void Finalize()
	{
		D3D_SHADER_MACRO LastMacro = {NULL, NULL};
		m_Macroes.push_back(LastMacro);
        m_bIsFinalized = true;
	}

	operator const D3D_SHADER_MACRO* ()
	{
        assert( !m_Macroes.size() || m_bIsFinalized );
        if( m_Macroes.size() && !m_bIsFinalized )
            Finalize();
        return m_Macroes.size() ? &m_Macroes[0] : NULL;
	}
    
private:

	std::vector< D3D_SHADER_MACRO > m_Macroes;
	std::set< std::string > m_DefinitionsPull;
    bool m_bIsFinalized;
};
