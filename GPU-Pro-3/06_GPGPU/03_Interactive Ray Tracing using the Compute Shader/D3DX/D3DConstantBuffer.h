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

#ifndef __D3DCONSTANTBUFFER_H__
#define __D3DCONSTANTBUFFER_H__

#include "D3D11Object.h"

class D3D11ConstantBuffer
{
private:
	ID3D11Buffer*		m_pBuffer;
	char				m_cName[64];

	ID3D11Buffer*		GetBuffer() { return m_pBuffer; }
	char*				GetName()	{ return m_cName; }
	
	HRESULT				Initialize();
public:
	D3D11ConstantBuffer( void ); 
	D3D11ConstantBuffer( char* a_Name ); 
	D3D11ConstantBuffer( D3D11ConstantBuffer *const );
	~D3D11ConstantBuffer( void );
};

#endif