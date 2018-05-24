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
#ifndef __CPUTPARSELIBRARY_H__
#define __CPUTPARSELIBRARY_H__



#include "CPUT.h"

#include <algorithm> // for std::transform

#if !defined(UNICODE) && !defined(_UNICODE)
#define fgetws      fgets
#define swscanf_s   sscanf_s
#define wcstok_s    strtok_s
#define wcsncmp     strncmp
#define _wtoi       atoi
#define _wtol       atol
#endif

typedef UINT UINT;

class CPUTConfigEntry
{
private:
    cString szName;
    cString szValue;

    friend class CPUTConfigBlock;
    friend class CPUTConfigFile;

public:
    CPUTConfigEntry() {}
    CPUTConfigEntry(const cString &name, const cString &value): szName(name), szValue(value){};

    static CPUTConfigEntry  &sNullConfigValue;

    const cString & NameAsString(void){ return  szName;};
    const cString & ValueAsString(void){ return szValue; }
	bool IsValid(void){ return !szName.empty(); }
    float ValueAsFloat(void)
    {
        float fValue=0;
        int retVal;
        retVal=swscanf_s(szValue.c_str(), _L("%g"), &fValue ); // float (regular float, or E exponentially notated float)
        ASSERT(0!=retVal, _L("ValueAsFloat - value specified is not a float"));
        return fValue;
    }
    int ValueAsInt(void)
    {
        int nValue=0;
        int retVal;
        retVal=swscanf_s(szValue.c_str(), _L("%d"), &nValue ); // signed int (NON-hex)
        ASSERT(0!=retVal, _L("ValueAsInt - value specified is not a signed int"));
        return nValue;
    }
    UINT ValueAsUint(void)
    {
        UINT nValue=0;
        int retVal;
        retVal=swscanf_s(szValue.c_str(), _L("%u"), &nValue ); // unsigned int
        ASSERT(0!=retVal, _L("ValueAsUint - value specified is not a UINT"));
        return nValue;
    }
    bool ValueAsBool(void)
    {
        return  (szValue.compare(_L("true")) == 0) || 
                (szValue.compare(_L("1")) == 0) || 
                (szValue.compare(_L("t")) == 0);
    }

    void ValueAsFloatArray(float *pFloats, int count);
};

class CPUTConfigBlock
{
public:
    CPUTConfigBlock();
    ~CPUTConfigBlock();

    CPUTConfigEntry *AddValue(const cString &szName, const cString &szValue);
    CPUTConfigEntry *GetValue(int nValueIndex);
    CPUTConfigEntry *GetValueByName(const cString &szName);
    const cString &GetName(void);
    int GetNameValue(void);
    int ValueCount(void);
    bool IsValid() { return mnValueCount > 0; }
private:
    CPUTConfigEntry mpValues[64];
    CPUTConfigEntry mName;
    cString         mszName;
    int             mnValueCount;

    friend class CPUTConfigFile;
};

class CPUTConfigFile
{
public:
    CPUTConfigFile();
    ~CPUTConfigFile();

    CPUTResult LoadFile(const cString &szFilename);

    CPUTConfigBlock *GetBlock(int nBlockIndex);
    CPUTConfigBlock *GetBlockByName(const cString &szBlockName);
    int BlockCount(void);
private:
    CPUTConfigBlock    *mpBlocks;
    int                 mnBlockCount;
};

#endif //#ifndef __CPUTPARSELIBRARY_H__