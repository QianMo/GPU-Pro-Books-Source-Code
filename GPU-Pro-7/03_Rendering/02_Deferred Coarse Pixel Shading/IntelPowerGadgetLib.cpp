/////////////////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or imlied.
// See the License for the specific language governing permissions and
// limitations under the License.
/////////////////////////////////////////////////////////////////////////////////////////////

#include "IntelPowerGadgetLib.h"
#include <Windows.h>
#include <string>
#include <vector>

using namespace std;

wstring g_lastError;
HMODULE g_hModule = NULL;


static bool split(const wstring& s, wstring &path)
{
	bool bResult = false;
	vector<wstring> output;

	wstring::size_type prev_pos = 0, pos = 0;

	while((pos = s.find(L';', pos)) != wstring::npos)
	{
		wstring substring( s.substr(prev_pos, pos-prev_pos) );
		if (substring.find(L"Power Gadget 2.") != wstring::npos)
		{
			path = substring;
			bResult = true;
			break;
		}
		prev_pos = ++pos;
	}

	if (!bResult)
	{
		wstring substring(s.substr(prev_pos, pos-prev_pos));

		if (substring.find(L"Power Gadget 2.") != wstring::npos)
		{
			path = substring;
			bResult = true;
		}
	}	

	if (bResult)
	{
		basic_string <char>::size_type pos = path.rfind(L" ");
		wstring version = path.substr(pos+1, path.length());
		double fVer = _wtof(version.c_str());
		if (fVer > 2.6)
			bResult = true;
	}
	
	return bResult;
}

static bool GetLibraryLocation(wstring& strLocation)
{
	WCHAR *pszPath = _wgetenv(L"IPG_Dir");
	if (pszPath == NULL || wcslen(pszPath) == 0)
		return false;

	WCHAR *pszVersion = _wgetenv(L"IPG_Ver");
	if (pszVersion == NULL || wcslen(pszVersion) == 0)
		return false;

	int version = (int)(_wtof(pszVersion) * 100);
	if (version >= 270)
	{
#if _M_X64
		strLocation = wstring(pszPath) + L"\\EnergyLib64.dll";	
#else
		strLocation = wstring(pszPath) + L"\\EnergyLib32.dll";
#endif
		return true;
	}
	else
		return false;
}

CIntelPowerGadgetLib::CIntelPowerGadgetLib(void) :
	pInitialize(NULL),
	pGetNumNodes(NULL),
	pGetMsrName(NULL),
	pGetMsrFunc(NULL),
	pGetIAFrequency(NULL),
	pGetTDP(NULL),
	pGetMaxTemperature(NULL),
	pGetTemperature(NULL),
	pReadSample(NULL),
	pGetBaseFrequency(NULL),
	pGetPowerData(NULL),
	pGetNumMsrs(NULL),
	pReadMSR(NULL)
{
	LastInstructions=0,LastUnhalted=0, LastRef=0;

	wstring strLocation;
	if (GetLibraryLocation(strLocation) == false)
	{
		g_lastError = L"Intel Power Gadget 2.7 or higher not found. If unsure, check if the path is in the user's path environment variable";
		return;
	}

	g_hModule = LoadLibrary(strLocation.c_str());
	if (g_hModule == NULL)
	{
		g_lastError = L"LoadLibrary failed on " + strLocation; 
		return;
	}
	
	pInitialize = (IPGInitialize) GetProcAddress(g_hModule, "IntelEnergyLibInitialize");
    pGetNumNodes = (IPGGetNumNodes) GetProcAddress(g_hModule, "GetNumNodes");
    pGetMsrName = (IPGGetMsrName) GetProcAddress(g_hModule, "GetMsrName");
    pGetMsrFunc = (IPGGetMsrFunc) GetProcAddress(g_hModule, "GetMsrFunc");
    pGetIAFrequency = (IPGGetIAFrequency) GetProcAddress(g_hModule, "GetIAFrequency");
    pGetTDP = (IPGGetTDP) GetProcAddress(g_hModule, "GetTDP");
    pGetMaxTemperature = (IPGGetMaxTemperature) GetProcAddress(g_hModule, "GetMaxTemperature");
    pGetTemperature = (IPGGetTemperature) GetProcAddress(g_hModule, "GetTemperature");
    pReadSample = (IPGReadSample) GetProcAddress(g_hModule, "ReadSample");
    pGetBaseFrequency = (IPGGetBaseFrequency) GetProcAddress(g_hModule, "GetBaseFrequency");
    pGetPowerData = (IPGGetPowerData) GetProcAddress(g_hModule, "GetPowerData");
    pGetNumMsrs = (IPGGetNumMsrs) GetProcAddress(g_hModule, "GetNumMsrs");
}




CIntelPowerGadgetLib::~CIntelPowerGadgetLib(void)
{
	if (g_hModule != NULL)
		FreeLibrary(g_hModule);
}



wstring CIntelPowerGadgetLib::GetLastError()
{
	return g_lastError;
}

bool CIntelPowerGadgetLib::IntelEnergyLibInitialize(void)
{
	if (pInitialize == NULL)
		return false;
	
	bool bSuccess = pInitialize();
	if (!bSuccess)
	{
		g_lastError = L"Initializing the energy library failed";
		return false;
	}
	
	return true;
}


bool CIntelPowerGadgetLib::GetNumNodes(int * nNodes)
{
	if(pGetNumNodes)
		return pGetNumNodes(nNodes);

	return false;
}

bool CIntelPowerGadgetLib::GetNumMsrs(int * nMsrs)
{
	if(pGetNumMsrs)
		return pGetNumMsrs(nMsrs);

	return false;
}

bool CIntelPowerGadgetLib::GetMsrName(int iMsr, wchar_t *pszName)
{
	if(pGetMsrName)
		return pGetMsrName(iMsr, pszName);

	return false;
}

bool CIntelPowerGadgetLib::GetMsrFunc(int iMsr, int *funcID)
{
	if(pGetMsrFunc)
		return  pGetMsrFunc(iMsr, funcID);

	return false;
}

bool CIntelPowerGadgetLib::GetIAFrequency(int iNode, int *freqInMHz)
{
	if(pGetIAFrequency)
		return pGetIAFrequency(iNode, freqInMHz);

	return false;
}


bool CIntelPowerGadgetLib::GetTDP(int iNode, double *TDP)
{
	if(pGetTDP)
		return  pGetTDP(iNode, TDP);

	return false;
}

bool CIntelPowerGadgetLib::GetMaxTemperature(int iNode, int *degreeC)
{
	if(pGetMaxTemperature)
		return  pGetMaxTemperature(iNode, degreeC);

	return false;
}

bool CIntelPowerGadgetLib::GetTemperature(int iNode, int *degreeC)
{
	if(pGetTemperature)
		return pGetTemperature(iNode, degreeC);

	return false;
}

bool CIntelPowerGadgetLib::ReadSample()
{
	bool bSuccess = pReadSample();
	if (bSuccess == false)
		g_lastError = L"MSR overflowed. You can safely discard this sample";
	return bSuccess;
}


bool CIntelPowerGadgetLib::GetBaseFrequency(int iNode, double *baseFrequency)
{
	if(pGetBaseFrequency)
		return pGetBaseFrequency(iNode, baseFrequency);
	return false;
}

bool CIntelPowerGadgetLib::GetPowerData(int iNode, int iMSR, double *results, int *nResult)
{
	if(pGetPowerData)
		return pGetPowerData(iNode, iMSR, results, nResult);
	return false;
}


