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
#include "D3DFloor.h"
#include "Assert.h"


D3DFloor::D3DFloor()
{
	dxDevice = NULL;
	vertexDeclaration = NULL;
	textureFloor = NULL;
}

D3DFloor::~D3DFloor()
{
	Destroy();
}

void D3DFloor::Create(IDirect3DDevice9* _dxDevice)
{
	dxDevice = _dxDevice;
	CreateVertexDeclaration();

	const char* shadersFile = ".\\Shaders\\Shaders.hlsl";
	vertexShader.Create(dxDevice, shadersFile, "FloorVS", NULL);
	pixelShader.Create(dxDevice, shadersFile, "FloorPS", NULL);

	HRESULT hr = D3DXCreateTextureFromFileEx(dxDevice, L".\\data\\Plane.dds", D3DX_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT, 0, D3DFMT_UNKNOWN, D3DPOOL_MANAGED, D3DX_DEFAULT, D3DX_DEFAULT, 0, NULL, NULL, &textureFloor);
	ASSERT(hr == D3D_OK, "Can't create 'data/Plane.dds' texture");
}

void D3DFloor::Destroy()
{
	SAFE_RELEASE(vertexDeclaration);
	SAFE_RELEASE(textureFloor);

	vertexShader.Destroy();
	pixelShader.Destroy();
}

void D3DFloor::CreateVertexDeclaration()
{
	D3DVERTEXELEMENT9 vertexElements[] =
	{
		{ 0,  0, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		D3DDECL_END()
	};

	HRESULT hr = dxDevice->CreateVertexDeclaration(vertexElements, &vertexDeclaration);
	ASSERT(hr == D3D_OK, "Can't create vertex declaration");
}

void D3DFloor::Draw(const D3DXMATRIX & viewProj)
{
	dxDevice->SetVertexDeclaration(vertexDeclaration);

	vertexShader.SetFloat4x4(dxDevice, "viewProj", viewProj);
	pixelShader.SetSampler(dxDevice, "floorMap", textureFloor);

	vertexShader.Set(dxDevice);
	pixelShader.Set(dxDevice);

	D3DXVECTOR3 vertices[6];
	vertices[0] = D3DXVECTOR3( 1.0f, 0.0f, -1.0f);
	vertices[1] = D3DXVECTOR3(-1.0f, 0.0f,  1.0f);
	vertices[2] = D3DXVECTOR3( 1.0f, 0.0f,  1.0f);

	vertices[3] = D3DXVECTOR3( 1.0f, 0.0f, -1.0f);
	vertices[4] = D3DXVECTOR3(-1.0f, 0.0f, -1.0f);
	vertices[5] = D3DXVECTOR3(-1.0f, 0.0f,  1.0f);


	dxDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
	dxDevice->DrawPrimitiveUP(D3DPT_TRIANGLELIST, 2, vertices, sizeof(D3DXVECTOR3));
	dxDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
}
