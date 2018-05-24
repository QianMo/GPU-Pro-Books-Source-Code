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

#include "stdafx.h"
#include "CloudySky.h"

// Parses the string as bool value
HRESULT ParseParameterBool( LPWSTR Value, bool &BoolValue )
{
    if( wcscmp(L"true", Value) == 0 )
        BoolValue = true;
    else if( wcscmp(L"false", Value) == 0 )
        BoolValue = false;
    else
    {
        CHECK_HR_RET( E_FAIL, L"\"%s\" is not correct boolean value. Only \"true\" and \"false\" are allowed", Value);
    }
    return S_OK;
}

// Parses the string as float value
float ParseParameterFloat( LPWSTR Value )
{
    float FloatVal;
    _stscanf_s( Value, L"%f", &FloatVal );
    return FloatVal;
}

// Parses the string as int value
int ParseParameterInt( LPWSTR Value )
{
    int IntVal;
    _stscanf_s( Value, L"%d", &IntVal);
    return IntVal;
}

// Parses the string as string value
void ParseParameterString( std::wstring &StringValue, FILE *pFile )
{
	StringValue.clear();
    // Read space
    fgetc(pFile);
	while( !feof(pFile) )
	{
		WCHAR CurrSymb = fgetc(pFile);
		if( CurrSymb != L'\n' && CurrSymb != 65535 )
			StringValue += CurrSymb;
		else
			break;
	}
}

void ParseParameterString( LPWSTR StringValue, int MaxLen, FILE *pFile )
{
    // Read space
    fgetc(pFile);
    fgetws(StringValue, MaxLen, pFile);
    size_t len = wcslen(StringValue);
    // Remove \n from the end of the line 
    if( StringValue[len-1] == L'\n' ) 
        StringValue[len-1] = L'\0';
}

// Parses the configuration file
HRESULT CCloudySkySample :: ParseConfigurationFile( LPCWSTR ConfigFilePath )
{
	m_strRawDEMDataFile.clear();
	m_strMtrlMaskFile.clear();

    FILE *pConfigFile = NULL;
    if( _tfopen_s( &pConfigFile, ConfigFilePath, _T("r") ) != 0 )
    {
        CHECK_HR_RET(E_FAIL, L"Failed to open the configuration file (%s)", ConfigFilePath);
    }
    
    while( !feof(pConfigFile) )
    {
        TCHAR Parameter[128];
        TCHAR EqualSign[128];

        _ftscanf_s( pConfigFile, _T("%s"), Parameter, _countof(Parameter));
        _ftscanf_s( pConfigFile, _T("%s"), EqualSign, _countof(EqualSign));
        if( wcscmp(EqualSign, L"=") != 0 )
        {
            LOG_ERROR( L"Equal sign (=) is missing for parameter \"%s\"", Parameter);
            goto ERROR_EXIT;
        }

        if( wcscmp(L"RawDEMDataFile", Parameter) == 0 )
        {
            ParseParameterString(m_strRawDEMDataFile, pConfigFile);
        }
		else if( wcscmp(L"MaterialMaskFile", Parameter) == 0 )
        {
            ParseParameterString(m_strMtrlMaskFile, pConfigFile);
        }
        else if( wcsncmp(L"TileTexture", Parameter, _tcslen(L"TileTexture")) == 0 )
        {
            int iTileNum = _tstoi(Parameter + _tcslen(L"TileTexture") );
			if( iTileNum >= 0 && iTileNum < CEarthHemsiphere::NUM_TILE_TEXTURES )
            {
                ParseParameterString(m_strTileTexPaths[iTileNum], pConfigFile);  
            }
        }
        else if( wcsncmp(L"TileNormalMap", Parameter, _tcslen(L"TileNormalMap")) == 0 )
        {
            int iTileNum = _tstoi(Parameter + _tcslen(L"TileNormalMap") );
			if( iTileNum >= 0 && iTileNum < CEarthHemsiphere::NUM_TILE_TEXTURES )
            {
                ParseParameterString(m_strNormalMapTexPaths[iTileNum], pConfigFile);  
            }
        }
        else if( wcscmp(L"TexturingMode", Parameter) == 0 )
        {
            TCHAR Value[128];
            ParseParameterString(Value, _countof(Value), pConfigFile);
            if( wcscmp(L"HeightBased", Value) == 0 )
                m_TerrainRenderParams.m_TexturingMode = SRenderingParams::TM_HEIGHT_BASED;
            else if( wcscmp(L"MaterialMask", Value) == 0 )
                m_TerrainRenderParams.m_TexturingMode = SRenderingParams::TM_MATERIAL_MASK;
            else if( wcscmp(L"MaterialMaskNM", Value) == 0 )
                m_TerrainRenderParams.m_TexturingMode = SRenderingParams::TM_MATERIAL_MASK_NM;
            else 
                LOG_ERROR( L"Unknown texturing mode (%s)\n"
                           L"Only the following modes are recognized:\n"
                           L"HeightBased\n", Value);
        }
        else
        {
            TCHAR Value[128];
            _ftscanf_s( pConfigFile, _T("%s"), Value, _countof(Value));

            // Parameters
            if( wcscmp(L"ElevationSamplingInterval", Parameter) == 0 )
            {
                m_TerrainRenderParams.m_TerrainAttribs.m_fElevationSamplingInterval = ParseParameterFloat( Value );
            }
            else if( wcscmp(L"ElevationScale", Parameter) == 0 )
            {
                m_TerrainRenderParams.m_TerrainAttribs.m_fElevationScale = ParseParameterFloat( Value );
            }
            else if(wcsncmp(L"TilingScale", Parameter, _tcslen(L"TilingScale")) == 0 )
            {
                int iTileNum = _tstoi(Parameter + _tcslen(L"TilingScale") );
                if( iTileNum == 0 )
                {
                    m_TerrainRenderParams.m_TerrainAttribs.m_fBaseMtrlTilingScale = ParseParameterFloat( Value );
                }
			    else if( iTileNum >= 1 && iTileNum < CEarthHemsiphere::NUM_TILE_TEXTURES )
                {
                    ((float*)&m_TerrainRenderParams.m_TerrainAttribs.m_f4TilingScale)[iTileNum-1] = ParseParameterFloat( Value );
                }
            }
            else if( wcscmp(L"RingDimension", Parameter) == 0 )
            {
                m_TerrainRenderParams.m_iRingDimension = ParseParameterInt( Value );
            }
            else if( wcscmp(L"NumRings", Parameter) == 0 )
            {
                m_TerrainRenderParams.m_iNumRings = ParseParameterInt( Value );
            }
            else if( wcscmp(L"ColOffset", Parameter) == 0 )
            {
                m_TerrainRenderParams.m_iColOffset = ParseParameterInt( Value );
            } 
            else if( wcscmp(L"RowOffset", Parameter) == 0 )
            {
                m_TerrainRenderParams.m_iRowOffset = ParseParameterInt( Value );
            } 
            else if( wcscmp(L"AnimateSun", Parameter) == 0 )
            {
                ParseParameterBool( Value, m_bAnimateSun );
            } 
        }
    }

    fclose(pConfigFile);

    return S_OK;

ERROR_EXIT:

    fclose(pConfigFile);

    return E_FAIL;
}
