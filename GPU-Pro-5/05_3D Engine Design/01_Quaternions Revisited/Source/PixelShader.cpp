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
#include "PixelShader.h"

PixelShader::PixelShader()
{
	pixelShader = NULL;
}

PixelShader::~PixelShader()
{
	Destroy();
}

void PixelShader::Destroy()
{
	Shader::Destroy();
	SAFE_RELEASE(pixelShader);
}

void PixelShader::Create(IDirect3DDevice9* device, const char* fileName, const char* entryPoint, const char* macroDefinition)
{
	Destroy();

	ID3DXBuffer* shaderCode = Shader::CreateFromFile(fileName, "ps_3_0", entryPoint, macroDefinition);
	if (shaderCode)
	{
		device->CreatePixelShader( ( DWORD* )shaderCode->GetBufferPointer(), &pixelShader );
		shaderCode->Release();
	}
}

void PixelShader::Set(IDirect3DDevice9* device) const
{
	device->SetPixelShader( pixelShader );
}

void PixelShader::SetSampler(IDirect3DDevice9* device, const char* name, IDirect3DBaseTexture9* texture) const
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

	UINT samplerIndex = constantTable->GetSamplerIndex(h);

	device->SetTexture(samplerIndex, texture);
	device->SetSamplerState(samplerIndex, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
	device->SetSamplerState(samplerIndex, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
	device->SetSamplerState(samplerIndex, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
	device->SetSamplerState(samplerIndex, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
	device->SetSamplerState(samplerIndex, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
	device->SetSamplerState(samplerIndex, D3DSAMP_ADDRESSW, D3DTADDRESS_WRAP);
}

 

