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

#ifndef __D3DCOMPUTESHADER_H__
#define __D3DCOMPUTESHADER_H__

#include "D3D11Object.h"

// Helper class to manage the compute shaders
class D3DComputeShader
{
public:
	D3DComputeShader(void);
	D3DComputeShader(UINT x, UINT y, UINT z);
	~D3DComputeShader(void);

	ID3D11ComputeShader*				GetShader() { return m_pShader; }
	std::vector<std::pair<string, int>>	GetMacros() { return m_Macros; }
	
	HRESULT								Load( WCHAR* pSrcFile, LPCSTR pFunctionName, 
												D3D11Object* m_d3dUtil, 
												std::vector<std::pair<string, int>> a_Macros );
	void								Dispatch( D3D11Object* m_d3dUtil );
	void								Release() { SAFE_RELEASE( m_pShader ); }
	void								SetDimensiones( UINT x, UINT y, UINT z );
private:
	ID3D11ComputeShader*				m_pShader; // pointer to the shader to be executed
	std::vector<std::pair<string, int>> m_Macros; // its content is sent as macros on the hlsl file
	UINT								m_X; // x-dimension on the group/block execution
	UINT								m_Y; // y-dimension on the group/block execution
	UINT								m_Z; // z-dimension on the group/block execution
};

#endif