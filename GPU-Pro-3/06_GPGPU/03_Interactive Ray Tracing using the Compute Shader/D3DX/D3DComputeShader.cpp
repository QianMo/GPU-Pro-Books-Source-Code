// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

#include "D3DComputeShader.h"

D3DComputeShader::D3DComputeShader(void)
{
	m_pShader = NULL;
	m_X = m_Y = m_Z = 1;
}

D3DComputeShader::D3DComputeShader(UINT x, UINT y, UINT z)
{
	m_pShader = NULL;
	SetDimensiones(x,y,z);
}

D3DComputeShader::~D3DComputeShader(void)
{
	m_pShader->Release();
}

//-----------------------------------------------------------------------------------------
// Load and compile the shader
//-----------------------------------------------------------------------------------------
HRESULT D3DComputeShader::Load( WCHAR* pSrcFile, LPCSTR pFunctionName, D3D11Object* m_d3dUtil, 
							   std::vector<std::pair<string, int>> a_Macros )
{
	m_Macros = a_Macros;

	HRESULT hr = S_OK;

	ID3DBlob* pBlob = NULL;         // used to store the compiled compute shader
	ID3DBlob* pErrorBlob = NULL;    // used to store any compilation errors

	D3D10_SHADER_MACRO Shader_Macros[10] = { "BLOCK_SIZE_X", "2", 
											"BLOCK_SIZE_Y", "1", 
											"BLOCK_SIZE_Z","1", "N", 
											"0" ,
											NULL, NULL,
											NULL, NULL,
											NULL, NULL,
											NULL, NULL,
											NULL, NULL,
											NULL, NULL,
	};

	// Use shader macros to dynamically define variables on the shader
	std::vector<string> str(m_Macros.size());
	for(unsigned int i = 0; i < m_Macros.size(); ++i)
	{		
		std::stringstream ss;
		ss << m_Macros[i].second;
		str[i] = ss.str();

		Shader_Macros[i].Name = m_Macros[i].first.data();
		Shader_Macros[i].Definition = str[i].data();	
	}

	hr = D3DX11CompileFromFile(
		pSrcFile,                   // use the code in this file
		Shader_Macros,              // use additional defines
		NULL,                       // don't use additional includes
		pFunctionName,              // compile this function
		"cs_5_0",                   // use compute shader 5.0
		NULL,                       // no compile flags
		NULL,                       // no effect flags
		NULL,                       // don't use a thread pump
		&pBlob,                     // store the compiled shader here
		&pErrorBlob,                // store any errors here
		NULL );                     // no thread pump is used, so no asynchronous HRESULT is needed

	 // if there were any errors, display them
	if( pErrorBlob )
	{
		printf("%s\n", (char*)pErrorBlob->GetBufferPointer());
	}

	if( FAILED(hr) )
		return hr;

	// if the compute shader was compiled successfully, create it on the GPU
	m_d3dUtil->GetDevice()->CreateComputeShader(
		pBlob->GetBufferPointer(),  // use the compute shader that was compiled here
		pBlob->GetBufferSize(),     // with this size
		NULL,                       // don't use any dynamic linkage
		&m_pShader );      // store the reference to the compute shader here

	return hr;
}

//-----------------------------------------------------------------------------------------
// Execute the shader
//-----------------------------------------------------------------------------------------
void D3DComputeShader::Dispatch( D3D11Object* m_d3dUtil )
{
	m_d3dUtil->GetDeviceContext()->CSSetShader( m_pShader, NULL, 0 );
	m_d3dUtil->GetDeviceContext()->Dispatch( m_X, m_Y, m_Z );
}

//-----------------------------------------------------------------------------------------
// Change the number of threads to execute per group/block
//-----------------------------------------------------------------------------------------
void D3DComputeShader::SetDimensiones(UINT x, UINT y, UINT z)
{
	m_X = x;
	m_Y = y;
	m_Z = z;
}