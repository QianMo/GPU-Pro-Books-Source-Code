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

#pragma once
#include <Windows.h>
#include <string>

using namespace std;

typedef bool (*IPGInitialize) ();
typedef bool (*IPGGetNumNodes) (int *nNodes);
typedef bool (*IPGGetNumMsrs) (int *nMsr);
typedef bool (*IPGGetMsrName) (int iMsr, wchar_t *szName);
typedef bool (*IPGGetMsrFunc) (int iMsr, int *funcID);
typedef bool (*IPGGetIAFrequency) (int iNode, int *freqInMHz);
typedef bool (*IPGGetTDP) (int iNode, double *TDP);
typedef bool (*IPGGetMaxTemperature) (int iNode, int *degreeC);
typedef bool (*IPGGetTemperature) (int iNode, int *degreeC);
typedef bool (*IPGReadSample) ();
typedef bool (*IPGGetBaseFrequency) (int iNode, double *baseFrequency);
typedef bool (*IPGGetPowerData) (int iNode, int iMSR, double *result, int *nResult);
typedef bool (*ReadMSRFunc)(void*, void*, void* );

class CIntelPowerGadgetLib
{
public:
	CIntelPowerGadgetLib(void);
	~CIntelPowerGadgetLib(void);
	
	bool IntelEnergyLibInitialize(void);
	bool GetNumNodes(int * nNodes);
	bool GetNumMsrs(int *nMsrs);
	bool GetMsrName(int iMsr, wchar_t *szName);
	bool GetMsrFunc(int iMsr, int *funcID);
	bool GetIAFrequency(int iNode, int *freqInMHz);
	bool GetTDP(int iNode, double *TDP);
	bool GetMaxTemperature(int iNode, int *degreeC);
	bool GetTemperature(int iNode, int *degreeC);
	bool ReadSample();
	bool GetBaseFrequency(int iNode, double *baseFrequency);
	bool GetPowerData(int iNode, int iMSR, double *results, int *nResult);
	wstring GetLastError();

private:
	IPGInitialize pInitialize;
	IPGGetNumNodes pGetNumNodes;
	IPGGetNumMsrs pGetNumMsrs;
	IPGGetMsrName pGetMsrName;
	IPGGetMsrFunc pGetMsrFunc;
	IPGGetIAFrequency pGetIAFrequency;
	IPGGetTDP pGetTDP;
	IPGGetMaxTemperature pGetMaxTemperature;
	IPGGetTemperature pGetTemperature;
	IPGReadSample pReadSample; 
	IPGGetBaseFrequency pGetBaseFrequency;
	IPGGetPowerData pGetPowerData;
	ReadMSRFunc pReadMSR;

	unsigned __int64  nominal_frequency; 
	unsigned __int64 LastInstructions,LastUnhalted, LastRef;
};