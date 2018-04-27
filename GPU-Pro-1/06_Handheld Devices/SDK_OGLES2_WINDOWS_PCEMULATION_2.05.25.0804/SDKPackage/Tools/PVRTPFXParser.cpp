/******************************************************************************

 @File         PVRTPFXParser.cpp

 @Title        PVRTPFXParser

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Windows + Linux

 @Description  PFX file parser.

******************************************************************************/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "PVRTGlobal.h"
#include "PVRTContext.h"
#include "PVRTMatrix.h"
#include "PVRTFixedPoint.h"
#include "PVRTMisc.h"
#include "PVRTPFXParser.h"
#include "PVRTResourceFile.h"
#include "PVRTString.h"

/****************************************************************************
** CPVRTPFXParserReadContext Class
****************************************************************************/
class CPVRTPFXParserReadContext
{
public:
	char			**ppszEffectFile;
	int				*pnFileLineNumber;
	unsigned int	nNumLines, nMaxLines;

public:
	CPVRTPFXParserReadContext();
	~CPVRTPFXParserReadContext();
};

/*!***************************************************************************
 @Function			CPVRTPFXParserReadContext
 @Description		Initialises values.
*****************************************************************************/
CPVRTPFXParserReadContext::CPVRTPFXParserReadContext()
{
	nMaxLines = 5000;
	nNumLines = 0;
	ppszEffectFile		= new char*[nMaxLines];
	pnFileLineNumber	= new int[nMaxLines];
}

/*!***************************************************************************
 @Function			~CPVRTPFXParserReadContext
 @Description		Frees allocated memory
*****************************************************************************/
CPVRTPFXParserReadContext::~CPVRTPFXParserReadContext()
{
	// free effect file
	for(unsigned int i = 0; i < nNumLines; i++)
	{
		FREE(ppszEffectFile[i]);
	}
	delete [] ppszEffectFile;
	delete [] pnFileLineNumber;
}


/*!***************************************************************************
 @Function			IgnoreWhitespace
 @Input				pszString
 @Output			pszString
 @Description
*****************************************************************************/
void IgnoreWhitespace(char **pszString)
{
	while(	*pszString[0] == '\t' ||
			*pszString[0] == '\n' ||
			*pszString[0] == '\r' ||
			*pszString[0] == ' ' )
	{
		(*pszString)++;
	}
}

/*!***************************************************************************
 @Function			GetSemanticDataFromString
 @Input				pszArgumentString
 @Input				eType
 @Output			pError				error message
 @Return			true if successful
 @Description
*****************************************************************************/
bool GetSemanticDataFromString(SPVRTSemanticDefaultData *pDataItem, const char * const pszArgumentString, ESemanticDefaultDataType eType, CPVRTString *pError)
{
	char *pszString = (char *)pszArgumentString;
	char *pszTmp;

	IgnoreWhitespace(&pszString);

	if(pszString[0] != '(')
	{
		*pError = CPVRTString("Missing '(' after ") + c_psSemanticDefaultDataTypeInfo[eType].pszName;
		return false;
	}
	pszString++;

	IgnoreWhitespace(&pszString);

	if(!strlen(pszString))
	{
		*pError = c_psSemanticDefaultDataTypeInfo[eType].pszName + CPVRTString(" missing arguments");
		return false;
	}

	pszTmp = pszString;
	switch(c_psSemanticDefaultDataTypeInfo[eType].eInternalType)
	{
		case eFloating:
			pDataItem->pfData[0] = (float)strtod(pszString, &pszTmp);
			break;
		case eInteger:
			pDataItem->pnData[0] = (int)strtol(pszString, &pszTmp, 10);
			break;
		case eBoolean:
			if(strncmp(pszString, "true", 4) == 0)
			{
				pDataItem->pbData[0] = true;
				pszTmp = &pszString[4];
			}
			else if(strncmp(pszString, "false", 5) == 0)
			{
				pDataItem->pbData[0] = false;
				pszTmp = &pszString[5];
			}
			break;
	}

	if(pszString == pszTmp)
	{
		size_t n = strcspn(pszString, ",\t ");
		char *pszError = (char *)malloc(n + 1);
		strcpy(pszError, "");
		strncat(pszError, pszString, n);
		*pError = CPVRTString("'") + pszError + "' unexpected for " + c_psSemanticDefaultDataTypeInfo[eType].pszName;
		FREE(pszError);
		return false;
	}
	pszString = pszTmp;

	IgnoreWhitespace(&pszString);

	for(unsigned int i = 1; i < c_psSemanticDefaultDataTypeInfo[eType].nNumberDataItems; i++)
	{
		if(!strlen(pszString))
		{
			*pError = c_psSemanticDefaultDataTypeInfo[eType].pszName + CPVRTString(" missing arguments");
			return false;
		}

		if(pszString[0] != ',')
		{
			size_t n = strcspn(pszString, ",\t ");
			char *pszError = (char *)malloc(n + 1);
			strcpy(pszError, "");
			strncat(pszError, pszString, n);
			*pError = CPVRTString("'") + pszError + "' unexpected for " + c_psSemanticDefaultDataTypeInfo[eType].pszName;
			FREE(pszError);
			return false;
		}
		pszString++;

		IgnoreWhitespace(&pszString);

		if(!strlen(pszString))
		{
			*pError = c_psSemanticDefaultDataTypeInfo[eType].pszName + CPVRTString(" missing arguments");
			return false;
		}

		pszTmp = pszString;
		switch(c_psSemanticDefaultDataTypeInfo[eType].eInternalType)
		{
			case eFloating:
				pDataItem->pfData[i] = (float)strtod(pszString, &pszTmp);
				break;
			case eInteger:
				pDataItem->pnData[i] = (int)strtol(pszString, &pszTmp, 10);
				break;
			case eBoolean:
				if(strncmp(pszString, "true", 4) == 0)
				{
					pDataItem->pbData[i] = true;
					pszTmp = &pszString[4];
				}
				else if(strncmp(pszString, "false", 5) == 0)
				{
					pDataItem->pbData[i] = false;
					pszTmp = &pszString[5];
				}
				break;
		}

		if(pszString == pszTmp)
		{
			size_t n = strcspn(pszString, ",\t ");
			char *pszError = (char *)malloc(n + 1);
			strcpy(pszError, "");
			strncat(pszError, pszString, n);
			*pError = CPVRTString("'") + pszError + "' unexpected for " + c_psSemanticDefaultDataTypeInfo[eType].pszName;
			FREE(pszError);
			return false;
		}
		pszString = pszTmp;

		IgnoreWhitespace(&pszString);
	}

	if(pszString[0] != ')')
	{
		size_t n = strcspn(pszString, "\t )");
		char *pszError = (char *)malloc(n + 1);
		strcpy(pszError, "");
		strncat(pszError, pszString, n);
		*pError = CPVRTString("'") + pszError + "' found when expecting ')' for " + c_psSemanticDefaultDataTypeInfo[eType].pszName;
		FREE(pszError);
		return false;
	}
	pszString++;

	IgnoreWhitespace(&pszString);

	if(strlen(pszString))
	{
		*pError = CPVRTString("'") + pszString + "' unexpected after ')'";
		return false;
	}

	return true;
}

/*!***************************************************************************
 @Function			ConcatenateLinesUntil
 @Output			pszOut		output text
 @Output			nLine		end line number
 @Input				nLine		start line number
 @Input				ppszLines	input text - one array element per line
 @Input				nLimit		number of lines input
 @Input				pszEnd		end string
 @Return			true if successful
 @Description		Outputs a block of text starting from nLine and ending
					when the string pszEnd is found.
*****************************************************************************/
static bool ConcatenateLinesUntil(char *&pszOut, int &nLine, const char * const * const ppszLines, const unsigned int nLimit, const char * const pszEnd)
{
	unsigned int	i, j;
	size_t			nLen;

	nLen = 0;
	for(i = nLine; i < nLimit; ++i)
	{
		if(strcmp(ppszLines[i], pszEnd) == 0)
			break;
		nLen += strlen(ppszLines[i]) + 1;
	}
	if(i == nLimit)
	{
		return false;
	}

	if(nLen)
	{
		++nLen;

		pszOut = (char*)malloc(nLen * sizeof(*pszOut));
		*pszOut = 0;

		for(j = nLine; j < i; ++j)
		{
			strcat(pszOut, ppszLines[j]);
			strcat(pszOut, "\n");
		}
	}
	else
	{
		pszOut = 0;
	}

	nLine = i;
	return true;
}


static char errorMsg[2048];

/*!***************************************************************************
 @Function			CPVRTPFXParser
 @Description		Sets initial values.
*****************************************************************************/
CPVRTPFXParser::CPVRTPFXParser()
{
	m_sHeader.pszVersion = NULL;
	m_sHeader.pszDescription = NULL;
	m_sHeader.pszCopyright = NULL;

	m_nMaxTextures = 10;
	m_nNumTextures = 0;
	m_psTexture = new SPVRTPFXParserTexture[m_nMaxTextures];

	m_nMaxVertShaders = 10;
	m_nNumVertShaders = 0;
	m_psVertexShader = new SPVRTPFXParserShader[m_nMaxVertShaders];

	m_nMaxFragShaders = 10;
	m_nNumFragShaders = 0;
	m_psFragmentShader = new SPVRTPFXParserShader[m_nMaxFragShaders];

	m_nMaxEffects = 10;
	m_nNumEffects = 0;
	m_psEffect = new SPVRTPFXParserEffect[m_nMaxEffects];
}

/*!***************************************************************************
 @Function			~CPVRTPFXParser
 @Description		Frees memory used.
*****************************************************************************/
CPVRTPFXParser::~CPVRTPFXParser()
{
	unsigned int i;

	// FREE header strings
	FREE(m_sHeader.pszVersion);
	FREE(m_sHeader.pszDescription);
	FREE(m_sHeader.pszCopyright);

	// free texture info
	for(i = 0; i < m_nNumTextures; ++i)
	{
		FREE(m_psTexture[i].pszName);
		FREE(m_psTexture[i].pszFile);
	}
	delete [] m_psTexture;

	// free shader strings
	for(i = 0; i < m_nNumFragShaders; ++i)
	{
		FREE(m_psFragmentShader[i].pszName);
		FREE(m_psFragmentShader[i].pszGLSLfile);
		FREE(m_psFragmentShader[i].pszGLSLcode);
		FREE(m_psFragmentShader[i].pszGLSLBinaryFile);
		FREE(m_psFragmentShader[i].pbGLSLBinary);
	}
	delete [] m_psFragmentShader;

	for(i = 0; i < m_nNumVertShaders; ++i)
	{
		FREE(m_psVertexShader[i].pszName);
		FREE(m_psVertexShader[i].pszGLSLfile);
		FREE(m_psVertexShader[i].pszGLSLcode);
		FREE(m_psVertexShader[i].pszGLSLBinaryFile);
		FREE(m_psVertexShader[i].pbGLSLBinary);
	}
	delete [] m_psVertexShader;

	for(unsigned int nEffect = 0; nEffect < m_nNumEffects; ++nEffect)
	{
		// free uniform strings
		for(i=0; i < m_psEffect[nEffect].nNumUniforms; ++i)
		{
			FREE(m_psEffect[nEffect].psUniform[i].pszName);
			FREE(m_psEffect[nEffect].psUniform[i].pszValue);
		}
		delete [] m_psEffect[nEffect].psUniform;

		// free uniform strings
		for(i=0; i < m_psEffect[nEffect].nNumAttributes; ++i)
		{
			FREE(m_psEffect[nEffect].psAttribute[i].pszName);
			FREE(m_psEffect[nEffect].psAttribute[i].pszValue);
		}
		delete [] m_psEffect[nEffect].psAttribute;

		for(i=0; i < m_psEffect[nEffect].nNumTextures; ++i)
		{
			FREE(m_psEffect[nEffect].psTextures[i].pszName);
		}
		delete [] m_psEffect[nEffect].psTextures;

		FREE(m_psEffect[nEffect].pszFragmentShaderName);
		FREE(m_psEffect[nEffect].pszVertexShaderName);

		FREE(m_psEffect[nEffect].pszAnnotation);
		FREE(m_psEffect[nEffect].pszName);
	}
	delete [] m_psEffect;
}

/*!***************************************************************************
 @Function			Parse
 @Output			pReturnError	error string
 @Return			bool			true for success parsing file
 @Description		Parses a loaded PFX file.
*****************************************************************************/
bool CPVRTPFXParser::Parse(CPVRTString * const pReturnError)
{
	int nEndLine = 0;
	int nHeaderCounter = 0, nTexturesCounter = 0;
	m_nNumVertShaders	= 0;
	m_nNumFragShaders	= 0;
	m_nNumEffects		= 0;

	// Loop through the file
	for(unsigned int nLine=0; nLine < m_psContext->nNumLines; nLine++)
	{
		// Skip blank lines
		if(!*m_psContext->ppszEffectFile[nLine])
			continue;

		if(strcmp("[HEADER]", m_psContext->ppszEffectFile[nLine]) == 0)
		{
			if(nHeaderCounter>0)
			{
				sprintf(errorMsg, "[HEADER] redefined on line %d\n", m_psContext->pnFileLineNumber[nLine]);
				*pReturnError = errorMsg;
				return false;
			}
			if(GetEndTag("HEADER", nLine, &nEndLine))
			{
				if(ParseHeader(nLine, nEndLine, pReturnError))
					nHeaderCounter++;
				else
					return false;
			}
			else
			{
				sprintf(errorMsg, "Missing [/HEADER] tag after [HEADER] on line %d\n", m_psContext->pnFileLineNumber[nLine]);
				*pReturnError = errorMsg;
				return false;
			}
			nLine = nEndLine;
		}
		else if(strcmp("[TEXTURES]", m_psContext->ppszEffectFile[nLine]) == 0)
		{
			if(nTexturesCounter>0)
			{
				sprintf(errorMsg, "[TEXTURES] redefined on line %d\n", m_psContext->pnFileLineNumber[nLine]);
				*pReturnError = errorMsg;
				return false;
			}
			if(GetEndTag("TEXTURES", nLine, &nEndLine))
			{
				if(ParseTextures(nLine, nEndLine, pReturnError))
					nTexturesCounter++;
				else
					return false;
			}
			else
			{
				sprintf(errorMsg, "Missing [/TEXTURES] tag after [TEXTURES] on line %d\n", m_psContext->pnFileLineNumber[nLine]);
				*pReturnError = errorMsg;
				return false;
			}
			nLine = nEndLine;
		}
		else if(strcmp("[VERTEXSHADER]", m_psContext->ppszEffectFile[nLine]) == 0)
		{
			if(GetEndTag("VERTEXSHADER", nLine, &nEndLine))
			{
				if(m_nNumVertShaders >= m_nMaxVertShaders)
				{
					sprintf(errorMsg, "%d vertex shaders read, can't store any more, on line %d\n", m_nNumVertShaders, m_psContext->pnFileLineNumber[nLine]);
					*pReturnError = errorMsg;
					return false;
				}
				if(ParseShader(nLine, nEndLine, pReturnError, m_psVertexShader[m_nNumVertShaders], "VERTEXSHADER"))
					m_nNumVertShaders++;
				else
				{
					FREE(m_psVertexShader[m_nNumVertShaders].pszName);
					FREE(m_psVertexShader[m_nNumVertShaders].pszGLSLfile);
					FREE(m_psVertexShader[m_nNumVertShaders].pszGLSLcode);
					FREE(m_psVertexShader[m_nNumVertShaders].pszGLSLBinaryFile);
					return false;
				}
			}
			else
			{
				sprintf(errorMsg, "Missing [/VERTEXSHADER] tag after [VERTEXSHADER] on line %d\n", m_psContext->pnFileLineNumber[nLine]);
				*pReturnError = errorMsg;
				return false;
			}
			nLine = nEndLine;
		}
		else if(strcmp("[FRAGMENTSHADER]", m_psContext->ppszEffectFile[nLine]) == 0)
		{
			if(GetEndTag("FRAGMENTSHADER", nLine, &nEndLine))
			{
				if(m_nNumFragShaders >= m_nMaxFragShaders)
				{
					sprintf(errorMsg, "%d fragment shaders read, can't store any more, on line %d\n", m_nNumFragShaders, m_psContext->pnFileLineNumber[nLine]);
					*pReturnError = errorMsg;
					return false;
				}
				if(ParseShader(nLine, nEndLine, pReturnError, m_psFragmentShader[m_nNumFragShaders], "FRAGMENTSHADER"))
					m_nNumFragShaders++;
				else
				{
					FREE(m_psFragmentShader[m_nNumFragShaders].pszName);
					FREE(m_psFragmentShader[m_nNumFragShaders].pszGLSLfile);
					FREE(m_psFragmentShader[m_nNumFragShaders].pszGLSLcode);
					FREE(m_psFragmentShader[m_nNumFragShaders].pszGLSLBinaryFile);
					return false;
				}
			}
			else
			{
				sprintf(errorMsg, "Missing [/FRAGMENTSHADER] tag after [FRAGMENTSHADER] on line %d\n", m_psContext->pnFileLineNumber[nLine]);
				*pReturnError = errorMsg;
				return false;
			}
			nLine = nEndLine;
		}
		else if(strcmp("[EFFECT]", m_psContext->ppszEffectFile[nLine]) == 0)
		{
			if(GetEndTag("EFFECT", nLine, &nEndLine))
			{
				if(m_nNumEffects >= m_nMaxEffects)
				{
					sprintf(errorMsg, "%d effects read, can't store any more, on line %d\n", m_nNumEffects, m_psContext->pnFileLineNumber[nLine]);
					*pReturnError = errorMsg;
					return false;
				}
				if(ParseEffect(m_psEffect[m_nNumEffects], nLine, nEndLine, pReturnError))
					m_nNumEffects++;
				else
				{
					unsigned int i;
					for(i=0; i < m_psEffect[m_nNumEffects].nNumUniforms; ++i)
					{
						FREE(m_psEffect[m_nNumEffects].psUniform[i].pszName);
						FREE(m_psEffect[m_nNumEffects].psUniform[i].pszValue);
					}
					delete [] m_psEffect[m_nNumEffects].psUniform;

					for(i=0; i < m_psEffect[m_nNumEffects].nNumAttributes; ++i)
					{
						FREE(m_psEffect[m_nNumEffects].psAttribute[i].pszName);
						FREE(m_psEffect[m_nNumEffects].psAttribute[i].pszValue);
					}
					delete [] m_psEffect[m_nNumEffects].psAttribute;

					for(i=0; i < m_psEffect[m_nNumEffects].nNumTextures; ++i)
					{
						FREE(m_psEffect[m_nNumEffects].psTextures[i].pszName);
					}
					delete [] m_psEffect[m_nNumEffects].psTextures;

					FREE(m_psEffect[m_nNumEffects].pszFragmentShaderName);
					FREE(m_psEffect[m_nNumEffects].pszVertexShaderName);

					FREE(m_psEffect[m_nNumEffects].pszAnnotation);
					FREE(m_psEffect[m_nNumEffects].pszName);

					return false;
				}
			}
			else
			{
				sprintf(errorMsg, "Missing [/EFFECT] tag after [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[nLine]);
				*pReturnError = errorMsg;
				return false;
			}
			nLine = nEndLine;
		}
		else
		{
			sprintf(errorMsg, "'%s' unexpected on line %d\n", m_psContext->ppszEffectFile[nLine], m_psContext->pnFileLineNumber[nLine]);
			*pReturnError = errorMsg;
			return false;
		}
	}

	if(m_nNumEffects < 1)
	{
		sprintf(errorMsg, "No [EFFECT] found. PFX file must have at least one defined.\n");
		*pReturnError = errorMsg;
		return false;
	}

	if(m_nNumFragShaders < 1)
	{
		sprintf(errorMsg, "No [FRAGMENTSHADER] found. PFX file must have at least one defined.\n");
		*pReturnError = errorMsg;
		return false;
	}

	if(m_nNumVertShaders < 1)
	{
		sprintf(errorMsg, "No [VERTEXSHADER] found. PFX file must have at least one defined.\n");
		*pReturnError = errorMsg;
		return false;
	}

	return true;
}

/*!***************************************************************************
 @Function			ParseFromMemory
 @Input				pszScript		PFX script
 @Output			pReturnError	error string
 @Return			EPVRTError		PVR_SUCCESS for success parsing file
									PVR_FAIL if file doesn't exist or is invalid
 @Description		Parses a PFX script from memory.
*****************************************************************************/
EPVRTError CPVRTPFXParser::ParseFromMemory(const char * const pszScript, CPVRTString * const pReturnError)
{
	CPVRTPFXParserReadContext	context;
	char			pszLine[512];
	const char		*pszEnd, *pszCurr;
	int				nLineCounter;
	unsigned int	nLen;
	unsigned int	nReduce;
	bool			bDone;

	if(!pszScript)
		return PVR_FAIL;

	m_psContext = &context;

	// Find & process each line
	nLineCounter	= 0;
	bDone			= false;
	pszCurr			= pszScript;
	while(!bDone)
	{
		nLineCounter++;

		while(*pszCurr == '\r')
			++pszCurr;

		// Find length of line
		pszEnd = strchr(pszCurr, '\n');
		if(pszEnd)
		{
			nLen = (unsigned int)(pszEnd - pszCurr);
		}
		else
		{
			nLen = (unsigned int)strlen(pszCurr);
			bDone = true;
		}

		nReduce = 0; // Tells how far to go back because of '\r'.
		while(nLen - nReduce > 0 && pszCurr[nLen - 1 - nReduce] == '\r')
			nReduce++;

		// Ensure pszLine will not be not overrun
		if(nLen+1-nReduce > sizeof(pszLine) / sizeof(*pszLine))
			nLen = sizeof(pszLine) / sizeof(*pszLine) - 1 + nReduce;

		// Copy line into pszLine
		strncpy(pszLine, pszCurr, nLen - nReduce);
		pszLine[nLen - nReduce] = 0;
		pszCurr += nLen + 1;

		_ASSERT(strchr(pszLine, '\r') == 0);
		_ASSERT(strchr(pszLine, '\n') == 0);

		// Ignore comments
		char *tmp = strstr(pszLine, "//");
		if(tmp != NULL)	*tmp = '\0';

		// Reduce whitespace to one character.
		ReduceWhitespace(pszLine);

		// Store the line, even if blank lines (to get correct errors from GLSL compiler).
		if(m_psContext->nNumLines < m_psContext->nMaxLines)
		{
			m_psContext->pnFileLineNumber[m_psContext->nNumLines] = nLineCounter;
			m_psContext->ppszEffectFile[m_psContext->nNumLines] = (char *)malloc((strlen(pszLine) + 1) * sizeof(char));
			strcpy(m_psContext->ppszEffectFile[m_psContext->nNumLines], pszLine);
			m_psContext->nNumLines++;
		}
		else
		{
			sprintf(errorMsg, "Too many lines of text in file (maximum is %d)\n", m_psContext->nMaxLines);
			*pReturnError = errorMsg;
			return PVR_FAIL;
		}
	}

	return Parse(pReturnError) ? PVR_SUCCESS : PVR_FAIL;
}

/*!***************************************************************************
 @Function			ParseFromFile
 @Input				pszFileName		PFX file name
 @Output			pReturnError	error string
 @Return			EPVRTError		PVR_SUCCESS for success parsing file
									PVR_FAIL if file doesn't exist or is invalid
 @Description		Reads the PFX file and calls the parser.
*****************************************************************************/
EPVRTError CPVRTPFXParser::ParseFromFile(const char * const pszFileName, CPVRTString * const pReturnError)
{
	CPVRTResourceFile PfxFile(pszFileName);
	if (!PfxFile.IsOpen())
	{
		*pReturnError = CPVRTString("Unable to open file ") + pszFileName;
		return PVR_FAIL;
	}
	return ParseFromMemory(PfxFile.StringPtr(), pReturnError);
}

/*!***************************************************************************
 @Function			GetEndTag
 @Input				pszTagName		tag name
 @Input				nStartLine		start line
 @Output			pnEndLine		line end tag found
 @Return			true if tag found
 @Description		Searches for end tag pszTagName from line nStartLine.
					Returns true and outputs the line number of the end tag if
					found, otherwise returning false.
*****************************************************************************/
bool CPVRTPFXParser::GetEndTag(const char *pszTagName, int nStartLine, int *pnEndLine)
{
	char pszEndTag[100];
	strcpy(pszEndTag, "[/");
	strcat(pszEndTag, pszTagName);
	strcat(pszEndTag, "]");

	for(unsigned int i = nStartLine; i < m_psContext->nNumLines; i++)
	{
		if(strcmp(pszEndTag, m_psContext->ppszEffectFile[i]) == 0)
		{
			*pnEndLine = i;
			return true;
		}
	}

	return false;
}

/*!***************************************************************************
 @Function			ReduceWhitespace
 @Output			line		output text
 @Input				line		input text
 @Description		Resuces all white space characters in the string to one
					blank space.
*****************************************************************************/
void CPVRTPFXParser::ReduceWhitespace(char *line)
{
	// TO DO: tidy up to remove code duplication

	// convert tabs and newlines to ' '
	char *tmp = strpbrk (line, "\t\n");
	while(tmp != NULL)
	{
		*tmp = ' ';
		tmp = strpbrk (line, "\t\n");
	}

	// remove all whitespace at start
	while(line[0] == ' ')
	{
		// move chars along to omit whitespace
		int counter = 0;
		do{
			line[counter] = line[counter+1];
			counter++;
		}while(line[counter] != '\0');
	}

	// step through chars of line remove multiple whitespace
	for(int i=0; i < (int)strlen(line); i++)
	{
		// whitespace found
		if(line[i] == ' ')
		{
			// count number of whitespace chars
			int numWhiteChars = 0;
			while(line[i+1+numWhiteChars] == ' ')
			{
				numWhiteChars++;
			}

			// multiple whitespace chars found
			if(numWhiteChars>0)
			{
				// move chars along to omit whitespace
				int counter=1;
				while(line[i+counter] != '\0')
				{
					line[i+counter] = line[i+numWhiteChars+counter];
					counter++;
				}
			}
		}
	}

	// remove all whitespace from end
	while(line[strlen(line)-1] == ' ')
	{
		// move chars along to omit whitespace
		line[strlen(line)-1] = '\0';
	}
}

/*!***************************************************************************
 @Function			ParseHeader
 @Input				nStartLine		start line number
 @Input				nEndLine		end line number
 @Output			pReturnError	error string
 @Return			bool			true if parse is successful
 @Description		Parses the HEADER section of the PFX file.
*****************************************************************************/
bool CPVRTPFXParser::ParseHeader(int nStartLine, int nEndLine, CPVRTString * const pReturnError)
{
	for(int i = nStartLine+1; i < nEndLine; i++)
	{
		// Skip blank lines
		if(!*m_psContext->ppszEffectFile[i])
			continue;

		char *str = strtok (m_psContext->ppszEffectFile[i]," ");
		if(str != NULL)
		{
			if(strcmp(str, "VERSION") == 0)
			{
				str += (strlen(str)+1);
				m_sHeader.pszVersion = (char *)malloc((strlen(str) + 1) * sizeof(char));
				strcpy(m_sHeader.pszVersion, str);
			}
			else if(strcmp(str, "DESCRIPTION") == 0)
			{
				str += (strlen(str)+1);
				m_sHeader.pszDescription = (char *)malloc((strlen(str) + 1) * sizeof(char));
				strcpy(m_sHeader.pszDescription, str);
			}
			else if(strcmp(str, "COPYRIGHT") == 0)
			{
				str += (strlen(str)+1);
				m_sHeader.pszCopyright = (char *)malloc((strlen(str) + 1) * sizeof(char));
				strcpy(m_sHeader.pszCopyright, str);
			}
			else
			{
				sprintf(errorMsg, "Unknown keyword '%s' in [HEADER] on line %d\n", str, m_psContext->pnFileLineNumber[i]);
				*pReturnError = errorMsg;
				return false;
			}
		}
		else
		{
			sprintf(errorMsg, "Missing arguments in [HEADER] on line %d : %s\n", m_psContext->pnFileLineNumber[i],  m_psContext->ppszEffectFile[i]);
			*pReturnError = errorMsg;
			return false;
		}
	}

	// initialise empty strings
	if(m_sHeader.pszVersion == NULL)
	{
		m_sHeader.pszVersion = (char *)malloc(sizeof(char));
		strcpy(m_sHeader.pszVersion, "");
	}
	if(m_sHeader.pszDescription == NULL)
	{
		m_sHeader.pszDescription = (char *)malloc(sizeof(char));
		strcpy(m_sHeader.pszDescription, "");
	}
	if(m_sHeader.pszCopyright == NULL)
	{
		m_sHeader.pszCopyright = (char *)malloc(sizeof(char));
		strcpy(m_sHeader.pszCopyright, "");
	}

	return true;
}

/*!***************************************************************************
 @Function			ParseTextures
 @Input				nStartLine		start line number
 @Input				nEndLine		end line number
 @Output			pReturnError	error string
 @Return			bool			true if parse is successful
 @Description		Parses the TEXTURE section of the PFX file.
*****************************************************************************/
bool CPVRTPFXParser::ParseTextures(int nStartLine, int nEndLine, CPVRTString * const pReturnError)
{
	m_nNumTextures = 0;

	for(int i = nStartLine+1; i < nEndLine; i++)
	{
		// Skip blank lines
		if(!*m_psContext->ppszEffectFile[i])
			continue;

		char *str = strtok (m_psContext->ppszEffectFile[i]," ");
		if(str != NULL)
		{
			if(strcmp(str, "FILE") == 0)
			{
				char			*pszName, *pszFile;
				unsigned int	nMin, nMag, nMIP;
				unsigned int	nWrapS, nWrapT, nWrapR;

				str = strtok (NULL, " ");
				if(str != NULL)
				{
					pszName = (char *)malloc( (strlen(str)+1) * sizeof(char));
					strcpy(pszName, str);
				}
				else
				{
					sprintf(errorMsg, "Texture name missing in [TEXTURES] on line %d: %s\n", m_psContext->pnFileLineNumber[i], m_psContext->ppszEffectFile[i]);
					*pReturnError = errorMsg;
					return false;
				}

				str = strtok (NULL, " ");
				if(str != NULL)
				{
					pszFile = (char *)malloc( (strlen(str)+1) * sizeof(char));
					strcpy(pszFile, str);
				}
				else
				{
					sprintf(errorMsg, "Texture name missing in [TEXTURES] on line %d: %s\n", m_psContext->pnFileLineNumber[i], m_psContext->ppszEffectFile[i]);
					*pReturnError = errorMsg;
					FREE(pszName);
					return false;
				}

				nMin = 0;
				nMag = 0;
				nMIP = 0;

				str = strtok (NULL, " ");
				if(str != NULL)
				{
					size_t	nLen;
					char	*pszMin, *pszMag, *pszMip, *pszBreak;

					nLen = strlen(str)+1;
					pszMin = (char*)malloc(nLen * sizeof(char));
					pszMag = (char*)malloc(nLen * sizeof(char));
					pszMip = (char*)malloc(nLen * sizeof(char));

					strcpy(pszMin, str);
					pszBreak = strchr(pszMin, '-');
					*pszBreak = 0;
					strcpy(pszMag, pszBreak + 1);
					pszBreak = strchr(pszMag, '-');
					*pszBreak = 0;
					strcpy(pszMip, pszBreak + 1);

					if(strcmp(pszMin, "LINEAR") == 0)
					{
						nMin = 1;
					}
					else
					{
						nMin = 0;
					}

					if(strcmp(pszMag, "LINEAR") == 0)
					{
						nMag = 1;
					}
					else
					{
						nMag = 0;
					}

					if(strcmp(pszMip, "LINEAR") == 0)
					{
						nMIP = 2;
					}
					else if(strcmp(pszMip, "NEAREST") == 0)
					{
						nMIP = 1;
					}
					else
					{
						nMIP = 0;
					}

					FREE(pszMin);
					FREE(pszMag);
					FREE(pszMip);
				}

				nWrapS = 1;
				nWrapT = 1;
				nWrapR = 1;

				str = strtok (NULL, " ");
				if(str != NULL)
				{
					if(strncmp(str, "CLAMP", 5) == 0)
					{
						nWrapS = 0;
						str += 5;
					}
					else if (strncmp(str, "REPEAT", 6) == 0)
					{
						nWrapS = 1;
						str += 6;
					}

					if(*str)
						++str;

					if(strncmp(str, "CLAMP", 5) == 0)
					{
						nWrapT = 0;
						str += 5;
					}
					else if (strncmp(str, "REPEAT", 6) == 0)
					{
						nWrapT = 1;
						str += 6;
					}

					if(*str)
						++str;

					if(strncmp(str, "CLAMP", 5) == 0)
					{
						nWrapR = 0;
						str += 5;
					}
					else if (strncmp(str, "REPEAT", 6) == 0)
					{
						nWrapR = 1;
						str += 6;
					}
				}

				if(m_nNumTextures >= m_nMaxTextures)
				{
					sprintf(errorMsg, "Too many textures in [TEXTURES] on line %d\n", m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					FREE(pszName);
					FREE(pszFile);
					return false;
				}
				m_psTexture[m_nNumTextures].pszName	= pszName;
				m_psTexture[m_nNumTextures].pszFile	= pszFile;
				m_psTexture[m_nNumTextures].nMin	= nMin;
				m_psTexture[m_nNumTextures].nMag	= nMag;
				m_psTexture[m_nNumTextures].nMIP	= nMIP;
				m_psTexture[m_nNumTextures].nWrapS	= nWrapS;
				m_psTexture[m_nNumTextures].nWrapT	= nWrapT;
				m_psTexture[m_nNumTextures].nWrapR	= nWrapR;
				++m_nNumTextures;
			}
			else
			{
				sprintf(errorMsg, "Unknown keyword '%s' in [TEXTURES] on line %d\n", str, m_psContext->pnFileLineNumber[i]);
				*pReturnError = errorMsg;
				return false;
			}

			str = strtok (NULL, " ");
			if(str != NULL)
			{
				sprintf(errorMsg, "unexpected data in [TEXTURES] on line %d: '%s'\n", m_psContext->pnFileLineNumber[i], str);
				*pReturnError = errorMsg;
				return false;
			}
		}
		else
		{
			sprintf(errorMsg, "Missing arguments in [TEXTURES] on line %d: %s\n", m_psContext->pnFileLineNumber[i],  m_psContext->ppszEffectFile[i]);
			*pReturnError = errorMsg;
			return false;
		}
	}

	return true;
}

/*!***************************************************************************
 @Function			ParseShader
 @Input				nStartLine		start line number
 @Input				nEndLine		end line number
 @Output			pReturnError	error string
 @Output			shader			shader data object
 @Input				pszBlockName	name of block in PFX file
 @Return			bool			true if parse is successful
 @Description		Parses the VERTEXSHADER or FRAGMENTSHADER section of the
					PFX file.
*****************************************************************************/
bool CPVRTPFXParser::ParseShader(int nStartLine, int nEndLine, CPVRTString * const pReturnError, SPVRTPFXParserShader &shader, const char * const pszBlockName)
{
	bool glslcode=0, glslfile=0, bName=0;

	shader.pszName			= NULL;
	shader.bUseFileName		= false;
	shader.pszGLSLfile		= NULL;
	shader.pszGLSLcode		= NULL;
	shader.pszGLSLBinaryFile= NULL;
	shader.pbGLSLBinary		= NULL;
	shader.nFirstLineNumber	= 0;

	for(int i = nStartLine+1; i < nEndLine; i++)
	{
		// Skip blank lines
		if(!*m_psContext->ppszEffectFile[i])
			continue;

		char *str = strtok (m_psContext->ppszEffectFile[i]," ");
		if(str != NULL)
		{
			// Check for [GLSL_CODE] tags first and remove those lines form loop.
			if(strcmp(str, "[GLSL_CODE]") == 0)
			{
				if(glslcode)
				{
					sprintf(errorMsg, "[GLSL_CODE] redefined in [%s] on line %d\n", pszBlockName, m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}
				if(glslfile && shader.pbGLSLBinary==NULL )
				{
					sprintf(errorMsg, "[GLSL_CODE] not allowed with FILE in [%s] on line %d\n", pszBlockName, m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}

				shader.nFirstLineNumber = m_psContext->pnFileLineNumber[i];

				// Skip the block-start
				i++;

				if(!ConcatenateLinesUntil(
					shader.pszGLSLcode,
					i,
					m_psContext->ppszEffectFile,
					m_psContext->nNumLines,
					"[/GLSL_CODE]"))
				{
					return false;
				}

				shader.bUseFileName = false;
				glslcode = 1;
			}
			else if(strcmp(str, "NAME") == 0)
			{
				if(bName)
				{
					sprintf(errorMsg, "NAME redefined in [%s] on line %d\n", pszBlockName, m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}

				str = strtok (NULL, " ");
				if(str == NULL)
				{
					sprintf(errorMsg, "NAME missing value in [%s] on line %d\n", pszBlockName, m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}

				shader.pszName = (char*)malloc((strlen(str)+1) * sizeof(char));
				strcpy(shader.pszName, str);
				bName = true;
			}
			else if(strcmp(str, "FILE") == 0)
			{
				if(glslfile)
				{
					sprintf(errorMsg, "FILE redefined in [%s] on line %d\n", pszBlockName, m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}
				if(glslcode)
				{
					sprintf(errorMsg, "FILE not allowed with [GLSL_CODE] in [%s] on line %d\n", pszBlockName, m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}

				str = strtok (NULL, " ");
				if(str == NULL)
				{
					sprintf(errorMsg, "FILE missing value in [%s] on line %d\n", pszBlockName, m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}

				shader.pszGLSLfile = (char*)malloc((strlen(str)+1) * sizeof(char));
				strcpy(shader.pszGLSLfile, str);

				CPVRTResourceFile GLSLFile(str);

				if(!GLSLFile.IsOpen())
				{
					sprintf(errorMsg, "Error loading file '%s' in [%s] on line %d\n", str, pszBlockName, m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}
				shader.pszGLSLcode = new char[GLSLFile.Size() + 1];
				strcpy(shader.pszGLSLcode, GLSLFile.StringPtr());

				shader.bUseFileName = true;
				glslfile = 1;
			}
			else if(strcmp(str, "BINARYFILE") == 0)
			{
				str = strtok (NULL, " ");
				if(str == NULL)
				{
					sprintf(errorMsg, "BINARYFILE missing value in [%s] on line %d\n", pszBlockName, m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}

				shader.pszGLSLBinaryFile = (char*)malloc((strlen(str)+1) * sizeof(char));
				strcpy(shader.pszGLSLBinaryFile, str);

				CPVRTResourceFile GLSLFile(str);

				if(!GLSLFile.IsOpen())
				{
					sprintf(errorMsg, "Error loading file '%s' in [%s] on line %d\n", str, pszBlockName, m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}
				shader.pbGLSLBinary = new char[GLSLFile.Size() + 1];
				shader.nGLSLBinarySize = (unsigned int)GLSLFile.Size();
				memcpy(shader.pbGLSLBinary, GLSLFile.StringPtr(), GLSLFile.Size());

				shader.bUseFileName = true;
				glslfile = 1;
			}
			else
			{
				sprintf(errorMsg, "Unknown keyword '%s' in [%s] on line %d\n", str, pszBlockName, m_psContext->pnFileLineNumber[i]);
				*pReturnError = errorMsg;
				return false;
			}

			str = strtok (NULL, " ");
			if(str != NULL)
			{
				sprintf(errorMsg, "Unexpected data in [%s] on line %d: '%s'\n", pszBlockName, m_psContext->pnFileLineNumber[i], str);
				*pReturnError = errorMsg;
				return false;
			}
		}
		else
		{
			sprintf(errorMsg, "Missing arguments in [%s] on line %d: %s\n", pszBlockName, m_psContext->pnFileLineNumber[i], m_psContext->ppszEffectFile[i]);
			*pReturnError = errorMsg;
			return false;
		}
	}

	if(!bName)
	{
		sprintf(errorMsg, "NAME not found in [%s] on line %d.\n", pszBlockName, m_psContext->pnFileLineNumber[nStartLine]);
		*pReturnError = errorMsg;
		return false;
	}

	if(!glslfile && !glslcode)
	{
		sprintf(errorMsg, "No Shader File or Shader Code specified in [%s] on line %d\n", pszBlockName, m_psContext->pnFileLineNumber[nStartLine]);
		*pReturnError = errorMsg;
		return false;
	}

	return true;
}

/*!***************************************************************************
 @Function			ParseSemantic
 @Output			semantic		semantic data object
 @Input				nStartLine		start line number
 @Input				nEndLine		end line number
 @Output			pReturnError	error string
 @Return			bool			true if parse is successful
 @Description		Parses a semantic.
*****************************************************************************/
bool CPVRTPFXParser::ParseSemantic(SPVRTPFXParserSemantic &semantic, const int nStartLine, const int nEndLine, CPVRTString * const pReturnError)
{
	char *str;

	semantic.pszName = 0;
	semantic.pszValue = 0;
	semantic.sDefaultValue.eType = eDataTypeNone;
	semantic.nIdx = 0;

	str = strtok (NULL, " ");
	if(str == NULL)
	{
		sprintf(errorMsg, "UNIFORM missing name in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[nStartLine]);
		*pReturnError = errorMsg;
		return false;
	}
	semantic.pszName = (char*)malloc((strlen(str)+1) * sizeof(char));
	strcpy(semantic.pszName, str);

	str = strtok (NULL, " ");
	if(str == NULL)
	{
		sprintf(errorMsg, "UNIFORM missing value in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[nStartLine]);
		*pReturnError = errorMsg;

		FREE(semantic.pszName);
		return false;
	}

	/*
		If the final digits of the semantic are a number they are
		stripped off and used as the index, with the remainder
		used as the semantic.
	*/
	{
		size_t idx, len;
		len = strlen(str);

		idx = len;
		while(idx)
		{
			--idx;
			if(strcspn(&str[idx], "0123456789") != 0)
			{
				break;
			}
		}
		if(idx == 0)
		{
			sprintf(errorMsg, "Semantic contains only numbers in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[nStartLine]);
			*pReturnError = errorMsg;

			FREE(semantic.pszName);
			return false;
		}

		++idx;
		// Store the semantic index
		if(len == idx)
		{
			semantic.nIdx = 0;
		}
		else
		{
			semantic.nIdx = atoi(&str[idx]);
		}

		// Chop off the index from the string containing the semantic
		str[idx] = 0;
	}

	// Store a copy of the semantic name
	semantic.pszValue = (char*)malloc((strlen(str)+1) * sizeof(char));
	strcpy(semantic.pszValue, str);

	/*
		Optional default semantic value
	*/
	char pszString[2048];
	strcpy(pszString,"");
	str = strtok (NULL, " ");
	if(str != NULL)
	{
		// Get all ramainning arguments
		while(str != NULL)
		{
			strcat(pszString, str);
			strcat(pszString, " ");
			str = strtok (NULL, " ");
		}

		// default value
		int i;
		for(i = 0; i < eNumDefaultDataTypes; i++)
		{
			if(strncmp(pszString, c_psSemanticDefaultDataTypeInfo[i].pszName, strlen(c_psSemanticDefaultDataTypeInfo[i].pszName)) == 0)
			{
				if(!GetSemanticDataFromString(	&semantic.sDefaultValue,
												&pszString[strlen(c_psSemanticDefaultDataTypeInfo[i].pszName)],
												c_psSemanticDefaultDataTypeInfo[i].eType,
												pReturnError
												))
				{
					sprintf(errorMsg, " on line %d.\n", m_psContext->pnFileLineNumber[nStartLine]);
					*pReturnError = *pReturnError + errorMsg;

					FREE(semantic.pszValue);
					FREE(semantic.pszName);
					return false;
				}

				semantic.sDefaultValue.eType = c_psSemanticDefaultDataTypeInfo[i].eType;
				break;
			}
		}

		// invalid data type
		if(i == eNumDefaultDataTypes)
		{
			sprintf(errorMsg, "'%s' unknown on line %d.\n", pszString, m_psContext->pnFileLineNumber[nStartLine]);
			*pReturnError = CPVRTString(errorMsg);

			FREE(semantic.pszValue);
			FREE(semantic.pszName);
			return false;
		}

	}

	return true;
}

/*!***************************************************************************
 @Function			ParseEffect
 @Output			effect			effect data object
 @Input				nStartLine		start line number
 @Input				nEndLine		end line number
 @Output			pReturnError	error string
 @Return			bool			true if parse is successful
 @Description		Parses the EFFECT section of the PFX file.
*****************************************************************************/
bool CPVRTPFXParser::ParseEffect(SPVRTPFXParserEffect &effect, const int nStartLine, const int nEndLine, CPVRTString * const pReturnError)
{
	bool bName = false;
	bool bVertShader = false;
	bool bFragShader = false;

	effect.pszName					= NULL;
	effect.pszAnnotation			= NULL;
	effect.pszVertexShaderName		= NULL;
	effect.pszFragmentShaderName	= NULL;

	effect.nMaxTextures				= 100;
	effect.nNumTextures				= 0;
	effect.psTextures				= new SPVRTPFXParserEffectTexture[effect.nMaxTextures];

	effect.nMaxUniforms				= 100;
	effect.nNumUniforms				= 0;
	effect.psUniform				= new SPVRTPFXParserSemantic[effect.nMaxUniforms];

	effect.nMaxAttributes			= 100;
	effect.nNumAttributes			= 0;
	effect.psAttribute				= new SPVRTPFXParserSemantic[effect.nMaxAttributes];

	for(int i = nStartLine+1; i < nEndLine; i++)
	{
		// Skip blank lines
		if(!*m_psContext->ppszEffectFile[i])
			continue;

		char *str = strtok (m_psContext->ppszEffectFile[i]," ");
		if(str != NULL)
		{
			if(strcmp(str, "[ANNOTATION]") == 0)
			{
				if(effect.pszAnnotation)
				{
					sprintf(errorMsg, "ANNOTATION redefined in [EFFECT] on line %d: \n", m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}

				i++;		// Skip the block-start
				if(!ConcatenateLinesUntil(
					effect.pszAnnotation,
					i,
					m_psContext->ppszEffectFile,
					m_psContext->nNumLines,
					"[/ANNOTATION]"))
				{
					return false;
				}
			}
			else if(strcmp(str, "VERTEXSHADER") == 0)
			{
				if(bVertShader)
				{
					sprintf(errorMsg, "VERTEXSHADER redefined in [EFFECT] on line %d: \n", m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}

				str = strtok(NULL, " ");
				if(str == NULL)
				{
					sprintf(errorMsg, "VERTEXSHADER missing value in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}
				effect.pszVertexShaderName = (char*)malloc((strlen(str)+1) * sizeof(char));
				strcpy(effect.pszVertexShaderName, str);

				bVertShader = true;
			}
			else if(strcmp(str, "FRAGMENTSHADER") == 0)
			{
				if(bFragShader)
				{
					sprintf(errorMsg, "FRAGMENTSHADER redefined in [EFFECT] on line %d: \n", m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}

				str = strtok(NULL, " ");
				if(str == NULL)
				{
					sprintf(errorMsg, "FRAGMENTSHADER missing value in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}
				effect.pszFragmentShaderName = (char*)malloc((strlen(str)+1) * sizeof(char));
				strcpy(effect.pszFragmentShaderName, str);

				bFragShader = true;
			}
			else if(strcmp(str, "TEXTURE") == 0)
			{
				if(effect.nNumTextures < effect.nMaxTextures)
				{
					// texture number
					str = strtok(NULL, " ");
					if(str != NULL)
						effect.psTextures[effect.nNumTextures].nNumber = atoi(str);
					else
					{
						sprintf(errorMsg, "TEXTURE missing value in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[i]);
						*pReturnError = errorMsg;
						return false;
					}

					// texture name
					str = strtok(NULL, " ");
					if(str != NULL)
					{
						effect.psTextures[effect.nNumTextures].pszName = (char*)malloc(strlen(str) + 1);
						strcpy(effect.psTextures[effect.nNumTextures].pszName, str);
					}
					else
					{
						sprintf(errorMsg, "TEXTURE missing value in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[i]);
						*pReturnError = errorMsg;
						return false;
					}

					++effect.nNumTextures;
				}
				else
				{
					sprintf(errorMsg, "Too many textures in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}
			}
			else if(strcmp(str, "UNIFORM") == 0)
			{
				if(effect.nNumUniforms < effect.nMaxUniforms)
				{
					if(!ParseSemantic(effect.psUniform[effect.nNumUniforms], i, nEndLine, pReturnError))
						return false;
					effect.nNumUniforms++;
				}
				else
				{
					sprintf(errorMsg, "Too many uniforms in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}
			}
			else if(strcmp(str, "ATTRIBUTE") == 0)
			{
				if(effect.nNumAttributes < effect.nMaxAttributes)
				{
					if(!ParseSemantic(effect.psAttribute[effect.nNumAttributes], i, nEndLine, pReturnError))
						return false;
					effect.nNumAttributes++;
				}
				else
				{
					sprintf(errorMsg, "Too many attributes in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[i]);
					*pReturnError = errorMsg;
					return false;
				}
			}
			else if(strcmp(str, "NAME") == 0)
			{
				if(bName)
				{
					sprintf(errorMsg, "NAME redefined in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[nStartLine]);
					*pReturnError = errorMsg;
					return false;
				}

				str = strtok (NULL, " ");
				if(str == NULL)
				{
					sprintf(errorMsg, "NAME missing value in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[nStartLine]);
					*pReturnError = errorMsg;
					return false;
				}

				effect.pszName = (char *)malloc((strlen(str)+1) * sizeof(char));
				strcpy(effect.pszName, str);
				bName = true;
			}
			else
			{
				sprintf(errorMsg, "Unknown keyword '%s' in [EFFECT] on line %d\n", str, m_psContext->pnFileLineNumber[i]);
				*pReturnError = errorMsg;
				return false;
			}
		}
		else
		{
			sprintf(errorMsg, "Missing arguments in [EFFECT] on line %d: %s\n", m_psContext->pnFileLineNumber[i], m_psContext->ppszEffectFile[i]);
			*pReturnError = errorMsg;
			return false;
		}

	}

	if(!bName)
	{
		sprintf(errorMsg, "No 'NAME' found in [EFFECT] on line %d\n", m_psContext->pnFileLineNumber[nStartLine]);
		*pReturnError = errorMsg;
		return false;
	}
	if(!bVertShader)
	{
		sprintf(errorMsg, "No 'VERTEXSHADER' defined in [EFFECT] starting on line %d: \n", m_psContext->pnFileLineNumber[nStartLine-1]);
		*pReturnError = errorMsg;
		return false;
	}
	if(!bFragShader)
	{
		sprintf(errorMsg, "No 'FRAGMENTSHADER' defined in [EFFECT] starting on line %d: \n", m_psContext->pnFileLineNumber[nStartLine-1]);
		*pReturnError = errorMsg;
		return false;
	}

	return true;
}

/*!***************************************************************************
 @Function			DebugDump
 @Description		Debug output.
*****************************************************************************/
void CPVRTPFXParser::DebugDump() const
{
	unsigned int i;

	printf("[HEADER]\n");
	printf("VERSION		%s\n", m_sHeader.pszVersion);
	printf("DESCRIPTION		%s\n", m_sHeader.pszDescription);
	printf("COPYRIGHT		%s\n", m_sHeader.pszCopyright);
	printf("[/HEADER]\n\n");

	printf("[TEXTURES]\n");
	for(i = 0; i < m_nNumTextures; ++i)
	{
		printf("FILE		%s		%s\n", m_psTexture[i].pszName, m_psTexture[i].pszFile);
	}
	printf("[/TEXTURES]\n\n");

	printf("[VERTEXSHADER]\n");
	printf("NAME		%s\n",	m_psVertexShader[0].pszName);
	printf("GLSLFILE		%s\n",	m_psVertexShader[0].pszGLSLfile);
	printf("[GLSL_CODE]\n");
	printf("%s", m_psVertexShader[0].pszGLSLcode);
	printf("[/GLSL_CODE]\n");
	printf("[/VERTEXSHADER]\n\n");

	printf("[FRAGMENTSHADER]\n");
	printf("NAME		%s\n",	m_psFragmentShader[0].pszName);
	printf("GLSLFILE		%s\n",	m_psFragmentShader[0].pszGLSLfile);
	printf("[GLSL_CODE]\n");
	printf("%s", m_psFragmentShader[0].pszGLSLcode);
	printf("[/GLSL_CODE]\n");
	printf("[/FRAGMENTSHADER]\n\n");

	for(unsigned int nEffect = 0; nEffect < m_nNumEffects; ++nEffect)
	{
		printf("[EFFECT]\n");

		printf("NAME		%s\n",	m_psEffect[nEffect].pszName);
		printf("[ANNOTATION]\n%s[/ANNOTATION]\n",	m_psEffect[nEffect].pszAnnotation);
		printf("FRAGMENTSHADER		%s\n",	m_psEffect[nEffect].pszFragmentShaderName);
		printf("VERTEXSHADER		%s\n",	m_psEffect[nEffect].pszVertexShaderName);

		for(i=0; i<m_psEffect[nEffect].nNumTextures; i++)
		{
			printf("TEXTURE		%d		%s\n", m_psEffect[nEffect].psTextures[i].nNumber, m_psEffect[nEffect].psTextures[i].pszName);
		}

		for(i=0; i<m_psEffect[nEffect].nNumUniforms; i++)
		{
			printf("UNIFORM		%s		%s%d\n", m_psEffect[nEffect].psUniform[i].pszName, m_psEffect[nEffect].psUniform[i].pszValue, m_psEffect[nEffect].psUniform[i].nIdx);
		}

		for(i=0; i<m_psEffect[nEffect].nNumAttributes; i++)
		{
			printf("ATTRIBUTE		%s		%s%d\n", m_psEffect[nEffect].psAttribute[i].pszName, m_psEffect[nEffect].psAttribute[i].pszValue, m_psEffect[nEffect].psAttribute[i].nIdx);
		}

		printf("[/EFFECT]\n\n");
	}
}

/*****************************************************************************
 End of file (PVRTPFXParser.cpp)
*****************************************************************************/
