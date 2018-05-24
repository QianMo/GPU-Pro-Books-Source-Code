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
#include "CPUTConfigBlock.h"
#include "CPUTOSServicesWin.h"

CPUTConfigEntry  &CPUTConfigEntry::sNullConfigValue = CPUTConfigEntry(_L(""), _L(""));

//----------------------------------------------------------------
static bool iswhite(char ch)
{
	return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n';
}

template<typename T>
static void RemoveWhitespace(T &start, T &end)
{
	while (start < end && iswhite(*start))
    {
		++start;
    }

	while (end > start && iswhite(*(end - 1)))
    {
		--end;
    }
}

//----------------------------------------------------------------
static bool ReadLine(const char **ppStart, const char **ppEnd, const char **ppCur)
{
	const char *pCur = *ppCur;
	if (!*pCur) // check for EOF
    {
		return false;
    }

	// We're at the start of a line now, skip leading whitespace
	while (*pCur == ' ' || *pCur == '\t')
    {
		++pCur;
    }

	*ppStart = pCur;

	// Forward to the end of the line and keep track of last non-whitespace char
	const char *pEnd = pCur;
	for (;;)
	{
		char ch = *pCur++;
		if (!ch)
		{
			--pCur; // terminating NUL isn't consumed
			break;
		}
		else if (ch == '\n')
        {
			break;
        }
		else if (!iswhite(ch))
        {
			pEnd = pCur;
        }
	}

	*ppEnd = pEnd;
	*ppCur = pCur;
	return true;
}

//----------------------------------------------------------------
static const char *FindFirst(const char *start, const char *end, char ch)
{
	const char *p = start;
	while (p < end && *p != ch)
    {
		++p;
    }
	return p;
}

static const char *FindLast(const char *start, const char *end, char ch)
{
	const char *p = end;
	while (--p >= start && *p != ch)
    {
    }

	return p;
}

static void AssignStr(cString &dest, const char *start, const char *end, _locale_t locale)
{
	dest.clear();
	if (end <= start)
    {
		return;
    }

	static const int NBUF = 64;
	wchar_t buf[NBUF];
	int nb = 0;

	size_t len = end - start;
	size_t initial = len + 1; // assume most characters are 1-byte
	dest.reserve(initial);

	const char *p = start;
	while (p < end)
	{
		int len = _mbtowc_l(&buf[nb++], p, end - p, locale);
		if (len < 1)
        {
			break;
        }

		p += len;
		if (p >= end || nb >= NBUF)
		{
			dest.append(buf, nb);
			nb = 0;
		}
	}
}

//----------------------------------------------------------------
void CPUTConfigEntry::ValueAsFloatArray(float *pFloats, int count)
{
    cString valueCopy = szValue;
    TCHAR *szOrigValue = (TCHAR*)valueCopy.c_str();

    TCHAR *szNewValue = NULL;
    TCHAR *szCurrValue = wcstok_s(szOrigValue, _L(" "), &szNewValue);
	for(int clear = 0; clear < count; clear++)
	{
		pFloats[clear] = 0.0f;
	}
    for(int ii=0;ii<count;++ii)
    {
		if(szCurrValue == NULL)
        {
            return;
        }
		pFloats[ii] = (float) _wtof(szCurrValue);
        szCurrValue = wcstok_s(NULL, _L(" "), &szNewValue);

    }
}
//----------------------------------------------------------------
CPUTConfigBlock::CPUTConfigBlock()
    : mnValueCount(0)
{
}
//----------------------------------------------------------------
CPUTConfigBlock::~CPUTConfigBlock()
{
}
//----------------------------------------------------------------
const cString &CPUTConfigBlock::GetName(void)
{
    return mszName;
}
//----------------------------------------------------------------
int CPUTConfigBlock::GetNameValue(void)
{
    return mName.ValueAsInt();
}
//----------------------------------------------------------------
CPUTConfigEntry *CPUTConfigBlock::GetValue(int nValueIndex)
{
    if(nValueIndex < 0 || nValueIndex >= mnValueCount)
    {
        return NULL;
    }
    return &mpValues[nValueIndex];
}
//----------------------------------------------------------------
CPUTConfigEntry *CPUTConfigBlock::AddValue(const cString &szName, const cString &szValue )
{
    cString szNameLower = szName;
    std::transform(szNameLower.begin(), szNameLower.end(), szNameLower.begin(), ::tolower);

    cString szValueLower = szValue;
    std::transform(szValueLower.begin(), szValueLower.end(), szValueLower.begin(), ::tolower);

    // TODO: What should we do if it already exists?
    CPUTConfigEntry *pEntry = &mpValues[mnValueCount++];
    pEntry->szName  = szNameLower;
    pEntry->szValue = szValueLower;
    return pEntry;
}
//----------------------------------------------------------------
CPUTConfigEntry *CPUTConfigBlock::GetValueByName(const cString &szName)
{
    for(int ii=0; ii<mnValueCount; ++ii)
    {
		const cString &valName = mpValues[ii].szName;
		if(valName.size() != szName.size())
        {
			continue;
        }

		size_t j = 0;
		while (j < valName.size() && tolower(szName[j]) == valName[j])
        {
			++j;
        }

		if (j == valName.size()) // match
		{
            return &mpValues[ii];
        }
    }

    // not found - return an 'empty' object to avoid crashes/extra error checking
    return &CPUTConfigEntry::sNullConfigValue;
}
//----------------------------------------------------------------
int CPUTConfigBlock::ValueCount(void)
{
    return mnValueCount;
}
//----------------------------------------------------------------
CPUTConfigFile::CPUTConfigFile()
    : mnBlockCount(0)
    , mpBlocks(NULL)
{
}
//----------------------------------------------------------------
CPUTConfigFile::~CPUTConfigFile()
{
    if(mpBlocks)
    {
        delete [] mpBlocks;
        mpBlocks = 0;
    }
    mnBlockCount = 0;
}
//----------------------------------------------------------------
CPUTResult CPUTConfigFile::LoadFile(const cString &szFilename)
{
    // Load the file
    cString             szCurrLine;
    CPUTConfigBlock    *pCurrBlock = NULL;
    FILE               *pFile = NULL;
    int                 nCurrBlock = 0;
    CPUTResult result = CPUTOSServices::GetOSServices()->OpenFile(szFilename, &pFile);
    if(CPUTFAILED(result))
    {
        return result;
    }

	_locale_t locale = _get_current_locale();

	/* Determine file size */
	fseek(pFile, 0, SEEK_END);
	int nBytes = ftell(pFile); // for text files, this is an overestimate
	fseek(pFile, 0, SEEK_SET);

	/* Read the whole thing */
	char *pFileContents = new char[nBytes + 1];
	nBytes = (int)fread(pFileContents, 1, nBytes, pFile);
	fclose(pFile);

	pFileContents[nBytes] = 0; // add 0-terminator

	/* Count the number of blocks */
	const char *pCur = pFileContents;
	const char *pStart, *pEnd;

	while(ReadLine(&pStart, &pEnd, &pCur))
	{
		const char *pOpen = FindFirst(pStart, pEnd, '[');
		const char *pClose = FindLast(pOpen + 1, pEnd, ']');
		if (pOpen < pClose)
		{
			// This line is a valid block header
			mnBlockCount++;
		}
	}

    // For files that don't have any blocks, just add the entire file to one block
    if(mnBlockCount == 0)
    {
        mnBlockCount   = 1;
    }

	pCur = pFileContents;
    mpBlocks = new CPUTConfigBlock[mnBlockCount];
    pCurrBlock = mpBlocks;

	/* Find the first block first */
	while(ReadLine(&pStart, &pEnd, &pCur))
	{
		const char *pOpen = FindFirst(pStart, pEnd, '[');
		const char *pClose = FindLast(pOpen + 1, pEnd, ']');
		if (pOpen < pClose)
		{
			// This line is a valid block header
            pCurrBlock = mpBlocks + nCurrBlock++;
			AssignStr(pCurrBlock->mszName, pOpen + 1, pClose, locale);
            std::transform(pCurrBlock->mszName.begin(), pCurrBlock->mszName.end(), pCurrBlock->mszName.begin(), ::tolower);
		}
		else if (pStart < pEnd)
		{
			// It's a value
			if (pCurrBlock == NULL)
            {
				continue;
            }

			const char *pEquals = FindFirst(pStart, pEnd, '=');
			if (pEquals == pEnd)
			{
                // No value, just a key, save it anyway
				// Optimistically, we assume it's new
				cString &name = pCurrBlock->mpValues[pCurrBlock->mnValueCount].szName;
				AssignStr(name, pStart, pEnd, locale);

                bool dup = false;
                for(int ii=0;ii<pCurrBlock->mnValueCount;++ii)
                {
                    if(!pCurrBlock->mpValues[ii].szName.compare(name))
                    {
                        dup = true;
                        break;
                    }
                }
                if(!dup)
                {
                    pCurrBlock->mnValueCount++;
                }
			}
			else
			{
				const char *pNameStart = pStart;
				const char *pNameEnd = pEquals;
				const char *pValStart = pEquals + 1;
				const char *pValEnd = pEnd;

				RemoveWhitespace(pNameStart, pNameEnd);
				RemoveWhitespace(pValStart, pValEnd);

				// Optimistically assume the name is new
				cString &name = pCurrBlock->mpValues[pCurrBlock->mnValueCount].szName;
				AssignStr(name, pNameStart, pNameEnd, locale);
				std::transform(name.begin(), name.end(), name.begin(), ::tolower);

                bool dup = false;
                for(int ii=0;ii<pCurrBlock->mnValueCount;++ii)
                {
                    if(!pCurrBlock->mpValues[ii].szName.compare(name))
                    {
                        dup = true;
                        break;
                    }
                }
                if(!dup)
                {
                    AssignStr(pCurrBlock->mpValues[pCurrBlock->mnValueCount].szValue, pValStart, pValEnd, locale);
                    pCurrBlock->mnValueCount++;
                }
			}
		}
	}

	delete[] pFileContents;
    return CPUT_SUCCESS;
}

//----------------------------------------------------------------
CPUTConfigBlock *CPUTConfigFile::GetBlock(int nBlockIndex)
{
    if(nBlockIndex >= mnBlockCount || nBlockIndex < 0)
    {
        return NULL;
    }

    return &mpBlocks[nBlockIndex];
}

//----------------------------------------------------------------
CPUTConfigBlock *CPUTConfigFile::GetBlockByName(const cString &szBlockName)
{
    cString szString = szBlockName;
    std::transform(szString.begin(), szString.end(), szString.begin(), ::tolower);

    for(int ii=0; ii<mnBlockCount; ++ii)
    {
        if(mpBlocks[ii].mszName.compare(szString) == 0)
        {
            return &mpBlocks[ii];
        }
    }
    return NULL;
}

//----------------------------------------------------------------
int CPUTConfigFile::BlockCount(void)
{
    return mnBlockCount;
}
