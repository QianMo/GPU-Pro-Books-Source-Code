/****************************************************************************

  GPU Pro 5 : Quaternions revisited - sample code
  All sample code written from scratch by Sergey Makeev specially for article.

  Copyright (c) 2013, Sergey Makeev

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.

  2. If you use this software in a non-commercial product, an acknowledgment
     in the product documentation would be appreciated but is not required.

  3. If you use this software in a commercial product you requried to credit
     the author.

  4. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

  5. This notice may not be removed or altered from any source distribution.
    

   Please let me know if you use the code in your products or have any questions or suggestions.

   e-mail: sergeymakeev@inbox.ru
   http://linkedin.com/in/sergeymakeev/

****************************************************************************/
#include <stdio.h>
#include <vector>
#include "shader.h"




HRESULT STDMETHODCALLTYPE Shader::ShaderIncluder::Open (D3DXINCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID *ppData, UINT *pBytes)
{
	UNREFERENCED_PARAMETER(pParentData);
	UNREFERENCED_PARAMETER(IncludeType);

	FILE* file = NULL;
	
	errno_t openResult = fopen_s(&file, pFileName, "rb");

	if (openResult != 0 || file == NULL)
	{
		return  D3DERR_NOTFOUND;
	}

	fseek(file, 0, SEEK_END);
	long  fileSize = ftell(file);
	fseek(file, 0, SEEK_SET);


	void* buffer = malloc(fileSize);

	size_t readedBytes = fread(buffer, 1, fileSize, file);
	if (readedBytes != (size_t)fileSize)
	{
		free(buffer);
		buffer = NULL;

		fclose(file);
		file = NULL;

		return D3DERR_INVALIDCALL;
	}


	fclose(file);
	file = NULL;

	*ppData = buffer;
	*pBytes = fileSize;

	return S_OK;
}

HRESULT STDMETHODCALLTYPE Shader::ShaderIncluder::Close (LPCVOID pData)
{
	if (pData)
	{
		free((void*)pData);
	}

	return S_OK;
}



//////////////////////////////////////////////////////////////////////////
Shader::Shader()
{
	constantTable = NULL;
}

Shader::~Shader()
{
	Destroy();
}


void Shader::Destroy()
{
	SAFE_RELEASE(constantTable);
}

ID3DXBuffer* Shader::CreateFromFile(const char* fileName, const char* profile, const char* entryPoint, const char* macroDefinition)
{
	std::vector<D3DXMACRO> macro;

	if (macroDefinition)
	{
		static char macroDefBuffer[16384];
		strcpy_s(macroDefBuffer, macroDefinition);

		const char * currentMacro = &macroDefBuffer[0];
		char * currentChar = &macroDefBuffer[0];
		while(*currentChar)
		{
			if (*currentChar == ';')
			{
				*currentChar = '\0';
				D3DXMACRO macroDef;
				macroDef.Name = currentMacro;
				macroDef.Definition = "1";
				macro.push_back(macroDef);
				currentMacro = (currentChar + 1);
			}
			currentChar++;
		}

		if (currentMacro != currentChar)
		{
			D3DXMACRO macroDef;
			macroDef.Name = currentMacro;
			macroDef.Definition = "1";
			macro.push_back(macroDef);
		}
	}

	D3DXMACRO macroEndMarker;
	macroEndMarker.Name = NULL;
	macroEndMarker.Definition = NULL;
	macro.push_back(macroEndMarker);

	ID3DXBuffer* shaderCode = NULL;
	ID3DXBuffer* compileErrors = NULL;

	DWORD compileFlag = 0;

#ifdef _DEBUG
	compileFlag |= D3DXSHADER_SKIPOPTIMIZATION | D3DXSHADER_DEBUG;
#endif

	static ShaderIncluder includeHandler;
	HRESULT hr = D3DXCompileShaderFromFileA(fileName, macroDefinition ? &macro[0] : NULL, &includeHandler, entryPoint, profile, compileFlag, &shaderCode, &compileErrors, &constantTable);

	if (hr == D3D_OK)
	{
		if (compileErrors)
		{
			const char* warningsMessage = (const char *)compileErrors->GetBufferPointer();
			if (warningsMessage)
			{
				OutputDebugStringA("Shader compile warnings: ");
				OutputDebugStringA(fileName);
				OutputDebugStringA("\n");
				if (macroDefinition)
				{
					OutputDebugStringA("Defines:");
					OutputDebugStringA(macroDefinition);
					OutputDebugStringA("\n");
				}
				OutputDebugStringA(warningsMessage);
			}
		}

		return shaderCode;
	}

	OutputDebugStringA("Shader compile errors: ");
	OutputDebugStringA(fileName);
	OutputDebugStringA("\n");
	if (macroDefinition)
	{
		OutputDebugStringA("Defines:");
		OutputDebugStringA(macroDefinition);
		OutputDebugStringA("\n");
	}
	if (compileErrors)
	{
		const char* errorsMessage = (const char *)compileErrors->GetBufferPointer();
		OutputDebugStringA(errorsMessage);
	} else
	{
		OutputDebugStringA("Unknown error\n");
	}
	OutputDebugStringA("\n---------------------------------------------\n");
	

	return NULL;
}


void Shader::SetFloat4x4(IDirect3DDevice9* device, const char* name, const D3DXMATRIX & v)
{
	if (!constantTable)
	{
		return;
	}

	D3DXHANDLE h = constantTable->GetConstantByName(NULL, name);
	if (!h)
	{
		return;
	}

	constantTable->SetMatrix(device, h, &v);
}

void Shader::SetFloat4(IDirect3DDevice9* device, const char* name, const D3DXVECTOR4 & v)
{
	if (!constantTable)
	{
		return;
	}

	D3DXHANDLE h = constantTable->GetConstantByName(NULL, name);
	if (!h)
	{
		return;
	}

	constantTable->SetVector(device, h, &v);
}

