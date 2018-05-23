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
#include "D3DMesh.h"
#include "D3DAnimation.h"
#include "Convertors.h"

extern bool g_DisableAlbedo;
extern bool g_PauseAnimation;
extern bool g_ShowNormals;
extern int g_TechniqueIndex;
extern bool g_DisableSkining;

D3DMesh::D3DMesh()
{
	dxDevice = NULL;
	animation = NULL;

	vertexCount = 0;
	indexCount = 0;

	ambientCube = NULL;
	middleGray = NULL;

	instanceVertexBuffer = NULL;
	indexBuffer = NULL;

	for (int i = 0; i < VFMT_MAX; i++)
	{
		vertexDeclarations[i] = NULL;
		vertexBuffers[i] = NULL;
	}

	verticesSizeof[0] = sizeof(VertexUnpackedTBN76);
	verticesSizeof[1] = sizeof(VertexPackedTBN40);
	verticesSizeof[2] = sizeof(VertexUnpackedQuat44);
	verticesSizeof[3] = sizeof(VertexPackedQuat32);
	verticesSizeof[4] = sizeof(VertexNoSkinUnpackedTBN56);
	verticesSizeof[5] = sizeof(VertexNoSkinPackedTBN28);
	verticesSizeof[6] = sizeof(VertexNoSkinUnpackedQuat32);
	verticesSizeof[7] = sizeof(VertexNoSkinPackedQuat20);

	int columnsCount = 16;
	float columnsSpace = 1.25f;
	float columnOffset = (columnsCount-1) * columnsSpace * 0.5f;

	int rowsCount = 16;
	float rowSpace = 1.25f;
	float rowOffset = (rowsCount-1) * rowSpace * 0.5f;

	for (int x = 0; x < columnsCount; x++)
	{
		for (int z = 0; z < rowsCount; z++)
		{
			meshInstances.push_back(MeshInstance(x * columnsSpace - columnOffset, z  * rowSpace - rowOffset, Utils::FRand(0.9f, 1.039f)));
		}
	}

}

D3DMesh::~D3DMesh()
{
	Destroy();
}

void D3DMesh::Draw(const D3DXMATRIX & mtxViewProj, const D3DXVECTOR3& cameraDir, float deltaTime)
{
	void* instanceVertexData = NULL;
	instanceVertexBuffer->Lock(0, sizeof(InstanceData) * meshInstances.size(), &instanceVertexData, D3DLOCK_DISCARD);
	InstanceData* instanceData = (InstanceData*)instanceVertexData;
	for (int instanceID = 0; instanceID < (int)meshInstances.size(); instanceID++)
	{
		const MeshInstance & instance = meshInstances[instanceID];

		instanceData[instanceID].fp16Quaternion[0] = Convert::Fp32ToFp16(instance.q.x);
		instanceData[instanceID].fp16Quaternion[1] = Convert::Fp32ToFp16(instance.q.y);
		instanceData[instanceID].fp16Quaternion[2] = Convert::Fp32ToFp16(instance.q.z);
		instanceData[instanceID].fp16Quaternion[3] = Convert::Fp32ToFp16(instance.q.w);

		instanceData[instanceID].fp16Position[0] = Convert::Fp32ToFp16(instance.p.x);
		instanceData[instanceID].fp16Position[1] = Convert::Fp32ToFp16(instance.p.y);
		instanceData[instanceID].fp16Position[2] = Convert::Fp32ToFp16(instance.p.z);

		instanceData[instanceID].fp16Scale[0] = Convert::Fp32ToFp16(instance.s.x);
		instanceData[instanceID].fp16Scale[1] = Convert::Fp32ToFp16(instance.s.y);
		instanceData[instanceID].fp16Scale[2] = Convert::Fp32ToFp16(instance.s.z);

		instanceData[instanceID].rgbColor[0] = 0xFF;
		instanceData[instanceID].rgbColor[1] = 0xFF;
		instanceData[instanceID].rgbColor[2] = 0xFF;

		instanceData[instanceID].instanceID = (byte)instanceID;
	}
	instanceVertexBuffer->Unlock();


	if (!g_PauseAnimation && !g_DisableSkining)
	{
		animation->BeginUpdate();
		for (int instanceID = 0; instanceID < (int)meshInstances.size(); instanceID++)
		{
			EvaluateAnimation(deltaTime, instanceID);
		}
		animation->EndUpdate();
	}

	dxDevice->SetRenderState( D3DRS_ALPHABLENDENABLE, TRUE );
	dxDevice->SetRenderState( D3DRS_SRCBLEND, D3DBLEND_SRCALPHA );
	dxDevice->SetRenderState( D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA );
	dxDevice->SetRenderState( D3DRS_BLENDOP, D3DBLENDOP_ADD );
	dxDevice->SetRenderState( D3DRS_ALPHATESTENABLE, FALSE );


	IDirect3DTexture9* animationTexture = animation->GetAsTexture();
	D3DSURFACE_DESC animationTextureDesc;
	animationTexture->GetLevelDesc(0, &animationTextureDesc);

	D3DXVECTOR4 animationDataParameters = D3DXVECTOR4((0.5f / animationTextureDesc.Width), (0.5f / animationTextureDesc.Height), 1.0f / animationTextureDesc.Width, 1.0f / animationTextureDesc.Height);

	
	int techniqueIndex = g_TechniqueIndex;
	int vertexFormatIndex = techniqueIndex;

	if (g_DisableSkining)
	{
		vertexFormatIndex += 4;
		techniqueIndex += 4;
	}
	
	if (g_ShowNormals)
		techniqueIndex += 8;


	ASSERT(techniqueIndex >= 0 && techniqueIndex < TECHNIQUES_COUNT, "Invalid technique index");
	VertexShader* currentVS = &techniques[techniqueIndex].vertexShader;
	PixelShader* currentPS = &techniques[techniqueIndex].pixelShader;

	D3DXMATRIX selectorConstant;
	D3DXMatrixIdentity(&selectorConstant);
	currentVS->SetFloat4x4(dxDevice, "selectorConstant", selectorConstant);
	currentVS->SetFloat4x4(dxDevice, "viewProj", mtxViewProj);
	currentVS->SetFloat4(dxDevice, "animationDataParameters", animationDataParameters);
	currentVS->SetSampler(dxDevice, "animationData", animationTexture);
	
	currentPS->SetSampler(dxDevice, "ambientCube", ambientCube);


	D3DXVECTOR4 defaultLightDirection = D3DXVECTOR4(0.23576348f, -0.76889056f, -0.59424841f, 0.0f);
	D3DXVECTOR4 lightDirection;
	D3DXVec4Normalize(&lightDirection, &defaultLightDirection);
	currentPS->SetFloat4(dxDevice, "lightDirection", lightDirection);

	currentVS->Set(dxDevice);
	currentPS->Set(dxDevice);

	unsigned long numInstancesToDraw = (unsigned long)meshInstances.size();
	dxDevice->SetStreamSource(0, vertexBuffers[vertexFormatIndex], 0, verticesSizeof[vertexFormatIndex]);
	dxDevice->SetStreamSourceFreq(0, (D3DSTREAMSOURCE_INDEXEDDATA | numInstancesToDraw));

	dxDevice->SetStreamSource(1, instanceVertexBuffer, 0, sizeof(D3DMesh::InstanceData));
	dxDevice->SetStreamSourceFreq(1, (D3DSTREAMSOURCE_INSTANCEDATA | 1ul));

	dxDevice->SetIndices(indexBuffer);
	dxDevice->SetVertexDeclaration(vertexDeclarations[vertexFormatIndex]);

	for (int materialIndex = 0; materialIndex < (int)materials.size(); materialIndex++)
	{
		const Material & material = materials[materialIndex];

		if (g_DisableAlbedo)
		{
			currentPS->SetSampler(dxDevice, "albedo", middleGray);
		} else
		{
			if (material.texAlbedoIndex >= 0)
				currentPS->SetSampler(dxDevice, "albedo", textures[material.texAlbedoIndex]);
		}

		if (material.texNormalsIndex >= 0)
			currentPS->SetSampler(dxDevice, "normalMap", textures[material.texNormalsIndex]);

		dxDevice->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, vertexCount, material.startIndex, material.trianglesCount);
	}

	dxDevice->SetStreamSourceFreq(0, 1);
	dxDevice->SetStreamSourceFreq(1, 1);
}

void D3DMesh::Create(IDirect3DDevice9* _dxDevice)
{
	dxDevice = _dxDevice;
	Destroy();
	CreateVertexDeclaration();

	const char* shadersFile = ".\\Shaders\\Shaders.hlsl";

	const char * techniqueNames[TECHNIQUES_COUNT] = {
		"UNPACKED_TBN",
		"PACKED_TBN",
		"UNPACKED_QUATERNIONS",
		"PACKED_QUATERNIONS",

		"UNPACKED_TBN;DISABLE_SKINING",
		"PACKED_TBN;DISABLE_SKINING",
		"UNPACKED_QUATERNIONS;DISABLE_SKINING",
		"PACKED_QUATERNIONS;DISABLE_SKINING",

		"UNPACKED_TBN;SHOW_NORMALS",
		"PACKED_TBN;SHOW_NORMALS",
		"UNPACKED_QUATERNIONS;SHOW_NORMALS",
		"PACKED_QUATERNIONS;SHOW_NORMALS",

		"UNPACKED_TBN;DISABLE_SKINING;SHOW_NORMALS",
		"PACKED_TBN;DISABLE_SKINING;SHOW_NORMALS",
		"UNPACKED_QUATERNIONS;DISABLE_SKINING;SHOW_NORMALS",
		"PACKED_QUATERNIONS;DISABLE_SKINING;SHOW_NORMALS"
	};
	
	for (int i = 0; i < TECHNIQUES_COUNT; i++)
	{
		OutputDebugStringA(Utils::StringFormat("[%d] - %s\n", i, techniqueNames[i]));
		techniques[i].vertexShader.Create(dxDevice, shadersFile, "MeshVS", techniqueNames[i]);
		techniques[i].pixelShader.Create(dxDevice, shadersFile, "MeshPS", techniqueNames[i]);
	}

	HRESULT hr = D3D_OK;

	hr = D3DXCreateCubeTextureFromFileA(dxDevice, ".\\data\\AmbientCube.dds", &ambientCube);
	ASSERT(hr == D3D_OK, "Can't load ambient cubemap 'data/AmbientCube.dds'");

	hr = D3DXCreateTextureFromFileExA(dxDevice, ".\\data\\Gray.dds", D3DX_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT, 0, D3DFMT_UNKNOWN, D3DPOOL_MANAGED, D3DX_DEFAULT, D3DX_DEFAULT, 0, NULL, NULL, &middleGray);
	ASSERT(hr == D3D_OK, "Can't load gray texture 'data/Gray.dds'");
}

void D3DMesh::Destroy()
{
	for (int i = 0; i < TECHNIQUES_COUNT; i++)
	{
		techniques[i].vertexShader.Destroy();
		techniques[i].pixelShader.Destroy();
	}

	for (int i = 0; i < VFMT_MAX; i++)
	{
		SAFE_RELEASE(vertexDeclarations[i]);
		SAFE_RELEASE(vertexBuffers[i]);
	}

	SAFE_RELEASE(ambientCube);
	SAFE_RELEASE(middleGray);

	SAFE_RELEASE(indexBuffer);
	
	SAFE_RELEASE(instanceVertexBuffer);

	for (int i = 0; i < (int)textures.size(); i++)
	{
		SAFE_RELEASE(textures[i]);
	}

	textures.clear();
}

void D3DMesh::CreateInstanceBuffer()
{
	SAFE_RELEASE(instanceVertexBuffer);
	dxDevice->CreateVertexBuffer( D3DAnimation::MAX_INSTANCES * sizeof(InstanceData), D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &instanceVertexBuffer, 0 );
}

void D3DMesh::DeviceLost()
{
	SAFE_RELEASE(instanceVertexBuffer);
}

void D3DMesh::DeviceReset()
{
	CreateInstanceBuffer();
}

void D3DMesh::CreateVertexDeclaration()
{

	D3DVERTEXELEMENT9 declarationVertexUnpacked76[] =
	{
		{ 0, offsetof(VertexUnpackedTBN76, pos), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		{ 0, offsetof(VertexUnpackedTBN76, uv), D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		{ 0, offsetof(VertexUnpackedTBN76, normal), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0 },
		{ 0, offsetof(VertexUnpackedTBN76, tangent), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TANGENT, 0 },
		{ 0, offsetof(VertexUnpackedTBN76, binormal), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BINORMAL, 0 },
		{ 0, offsetof(VertexUnpackedTBN76, blendWeights), D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDWEIGHT, 0 },
		{ 0, offsetof(VertexUnpackedTBN76, blendIndices), D3DDECLTYPE_UBYTE4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDINDICES, 0 },
		
		{ 1, 0, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1 },
		{ 1, 8, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2 },
		{ 1, 16, D3DDECLTYPE_FLOAT16_2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 3 },
		{ 1, 20, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
		D3DDECL_END()
	};

	

	D3DVERTEXELEMENT9 declarationVertexPackedTBN40[] =
	{
		{ 0, offsetof(VertexPackedTBN40, pos), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		{ 0, offsetof(VertexPackedTBN40, uv), D3DDECLTYPE_SHORT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		{ 0, offsetof(VertexPackedTBN40, packedNormal), D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0 },
		{ 0, offsetof(VertexPackedTBN40, packedTangent), D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TANGENT, 0 },
		{ 0, offsetof(VertexPackedTBN40, packedBinormal), D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BINORMAL, 0 },
		{ 0, offsetof(VertexPackedTBN40, blendWeights), D3DDECLTYPE_SHORT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDWEIGHT, 0 },
		{ 0, offsetof(VertexPackedTBN40, blendIndices), D3DDECLTYPE_UBYTE4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDINDICES, 0 },
		
		{ 1, 0, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1 },
		{ 1, 8, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2 },
		{ 1, 16, D3DDECLTYPE_FLOAT16_2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 3 },
		{ 1, 20, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
		D3DDECL_END()
	};

	D3DVERTEXELEMENT9 declarationVertexUnpackedQuat44[] =
	{
		{ 0, offsetof(VertexUnpackedQuat44, pos), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		{ 0, offsetof(VertexUnpackedQuat44, uv), D3DDECLTYPE_SHORT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		{ 0, offsetof(VertexUnpackedQuat44, quaternionTBN), D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 4 },
		{ 0, offsetof(VertexUnpackedQuat44, blendWeights), D3DDECLTYPE_SHORT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDWEIGHT, 0 },
		{ 0, offsetof(VertexUnpackedQuat44, blendIndices), D3DDECLTYPE_UBYTE4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDINDICES, 0 },

		{ 1, 0, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1 },
		{ 1, 8, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2 },
		{ 1, 16, D3DDECLTYPE_FLOAT16_2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 3 },
		{ 1, 20, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
		D3DDECL_END()
	};

	D3DVERTEXELEMENT9 declarationVertexPackedQuat32[] =
	{
		{ 0, offsetof(VertexPackedQuat32, pos), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		{ 0, offsetof(VertexPackedQuat32, uv), D3DDECLTYPE_SHORT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		{ 0, offsetof(VertexPackedQuat32, packedQuaternionTBN), D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 4 },
		{ 0, offsetof(VertexPackedQuat32, blendWeights), D3DDECLTYPE_SHORT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDWEIGHT, 0 },
		{ 0, offsetof(VertexPackedQuat32, blendIndices), D3DDECLTYPE_UBYTE4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDINDICES, 0 },

		{ 1, 0, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1 },
		{ 1, 8, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2 },
		{ 1, 16, D3DDECLTYPE_FLOAT16_2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 3 },
		{ 1, 20, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
		D3DDECL_END()
	};


	D3DVERTEXELEMENT9 declarationVertexNoSkinUnpackedTBN56[] =
	{
		{ 0, offsetof(VertexUnpackedTBN76, pos), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		{ 0, offsetof(VertexUnpackedTBN76, uv), D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		{ 0, offsetof(VertexUnpackedTBN76, normal), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0 },
		{ 0, offsetof(VertexUnpackedTBN76, tangent), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TANGENT, 0 },
		{ 0, offsetof(VertexUnpackedTBN76, binormal), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BINORMAL, 0 },

		{ 1, 0, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1 },
		{ 1, 8, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2 },
		{ 1, 16, D3DDECLTYPE_FLOAT16_2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 3 },
		{ 1, 20, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
		D3DDECL_END()
	};



	D3DVERTEXELEMENT9 declarationVertexNoSkinPackedTBN28[] =
	{
		{ 0, offsetof(VertexPackedTBN40, pos), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		{ 0, offsetof(VertexPackedTBN40, uv), D3DDECLTYPE_SHORT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		{ 0, offsetof(VertexPackedTBN40, packedNormal), D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0 },
		{ 0, offsetof(VertexPackedTBN40, packedTangent), D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TANGENT, 0 },
		{ 0, offsetof(VertexPackedTBN40, packedBinormal), D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BINORMAL, 0 },

		{ 1, 0, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1 },
		{ 1, 8, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2 },
		{ 1, 16, D3DDECLTYPE_FLOAT16_2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 3 },
		{ 1, 20, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
		D3DDECL_END()
	};

	D3DVERTEXELEMENT9 declarationVertexNoSkinUnpackedQuat32[] =
	{
		{ 0, offsetof(VertexUnpackedQuat44, pos), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		{ 0, offsetof(VertexUnpackedQuat44, uv), D3DDECLTYPE_SHORT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		{ 0, offsetof(VertexUnpackedQuat44, quaternionTBN), D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 4 },

		{ 1, 0, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1 },
		{ 1, 8, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2 },
		{ 1, 16, D3DDECLTYPE_FLOAT16_2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 3 },
		{ 1, 20, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
		D3DDECL_END()
	};

	D3DVERTEXELEMENT9 declarationVertexNoSkinPackedQuat20[] =
	{
		{ 0, offsetof(VertexPackedQuat32, pos), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		{ 0, offsetof(VertexPackedQuat32, uv), D3DDECLTYPE_SHORT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		{ 0, offsetof(VertexPackedQuat32, packedQuaternionTBN), D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 4 },

		{ 1, 0, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1 },
		{ 1, 8, D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2 },
		{ 1, 16, D3DDECLTYPE_FLOAT16_2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 3 },
		{ 1, 20, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
		D3DDECL_END()
	};


	HRESULT hr = D3D_OK;
	hr = dxDevice->CreateVertexDeclaration(declarationVertexUnpacked76, &vertexDeclarations[0]);
	ASSERT(hr == D3D_OK, "Can't create vertex declaration for VertexUnpacked76");

	hr = dxDevice->CreateVertexDeclaration(declarationVertexPackedTBN40, &vertexDeclarations[1]);
	ASSERT(hr == D3D_OK, "Can't create vertex declaration for VertexPackedTBN40");

	hr = dxDevice->CreateVertexDeclaration(declarationVertexUnpackedQuat44, &vertexDeclarations[2]);
	ASSERT(hr == D3D_OK, "Can't create vertex declaration for VertexUnpackedQuat44");

	hr = dxDevice->CreateVertexDeclaration(declarationVertexPackedQuat32, &vertexDeclarations[3]);
	ASSERT(hr == D3D_OK, "Can't create vertex declaration for VertexPackedQuat32");

	hr = dxDevice->CreateVertexDeclaration(declarationVertexNoSkinUnpackedTBN56, &vertexDeclarations[4]);
	ASSERT(hr == D3D_OK, "Can't create vertex declaration for VertexUnpacked76");

	hr = dxDevice->CreateVertexDeclaration(declarationVertexNoSkinPackedTBN28, &vertexDeclarations[5]);
	ASSERT(hr == D3D_OK, "Can't create vertex declaration for VertexPackedTBN40");

	hr = dxDevice->CreateVertexDeclaration(declarationVertexNoSkinUnpackedQuat32, &vertexDeclarations[6]);
	ASSERT(hr == D3D_OK, "Can't create vertex declaration for VertexUnpackedQuat44");

	hr = dxDevice->CreateVertexDeclaration(declarationVertexNoSkinPackedQuat20, &vertexDeclarations[7]);
	ASSERT(hr == D3D_OK, "Can't create vertex declaration for VertexPackedQuat32");
}


void D3DMesh::CreateBones(int bonesCount)
{
	skinBones.resize(bonesCount);
}

D3DMesh::SkinBone* D3DMesh::MapBones()
{
	return &skinBones[0];
}

void D3DMesh::UnmapBones()
{
}

void D3DMesh::CreateMaterials(int materialsCount)
{
	materials.resize(materialsCount);
}

D3DMesh::Material* D3DMesh::MapMaterials()
{
	return &materials[0];
}

void D3DMesh::UnmapMaterials()
{
}

void D3DMesh::CreateVertexBuffer(int verticesCount)
{
	vertexCount = verticesCount;

	for (int i = 0; i < VFMT_MAX; i++)
	{
		SAFE_RELEASE(vertexBuffers[i]);

		HRESULT hr = dxDevice->CreateVertexBuffer( verticesCount * verticesSizeof[i], D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED, &vertexBuffers[i], 0);
		ASSERT(hr == D3D_OK, "Can't create vertex buffer");
	}

}

void* D3DMesh::MapVertexBuffer(D3DMesh::VertexFormat fmt)
{
	int formatIndex = (int)fmt;
	ASSERT(vertexBuffers[formatIndex], "Vertex buffer can't be NULL");

	void* pData = NULL;
	HRESULT hr = vertexBuffers[formatIndex]->Lock(0, 0, &pData, 0);
	ASSERT(hr == D3D_OK, "Can't lock vertex buffer");
	return pData;
}

void D3DMesh::UnmapVertexBuffer(D3DMesh::VertexFormat fmt)
{
	int formatIndex = (int)fmt;
	ASSERT(vertexBuffers[formatIndex], "Vertex buffer can't be NULL");
	vertexBuffers[formatIndex]->Unlock();
}

void D3DMesh::CreateIndexBuffer(int indicesCount)
{
	indexCount = indicesCount;
	SAFE_RELEASE(indexBuffer);

	HRESULT hr = dxDevice->CreateIndexBuffer(indicesCount * sizeof(WORD), D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED, &indexBuffer, 0);
	ASSERT(hr == D3D_OK, "Can't create index buffer");
}

WORD* D3DMesh::MapIndexBuffer()
{
	ASSERT(indexBuffer, "Vertex index can't be NULL");

	void* pData = NULL;
	indexBuffer->Lock(0, 0, &pData, 0);
	return (WORD *)pData;
}

void D3DMesh::UnmapIndexBuffer()
{
	ASSERT(indexBuffer, "Vertex index can't be NULL");
	indexBuffer->Unlock();
}


void D3DMesh::CreateTextures()
{
	std::string dataFolder = ".\\data\\";
	std::string texturePostfix = ".dds";
	std::string normalmapPostfix = "_normalmap.dds";

	IDirect3DTexture9* texture = NULL;
	std::string textureFileName;
	for (int materialIndex = 0; materialIndex < (int)materials.size(); materialIndex++)
	{
		Material & material = materials[materialIndex];

		if (material.fileName.empty())
		{
			material.texAlbedoIndex = -1;
			material.texNormalsIndex = -1;
			OutputDebugStringA("WARNING: Material without textures\n");
			continue;
		}

		textureFileName = dataFolder + material.fileName + texturePostfix;
		texture = NULL;
		D3DXCreateTextureFromFileExA(dxDevice, textureFileName.c_str(), D3DX_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT, 0, D3DFMT_UNKNOWN, D3DPOOL_MANAGED, D3DX_DEFAULT, D3DX_DEFAULT, 0, NULL, NULL, &texture);
		if (texture)
		{
			material.texAlbedoIndex = (int)textures.size();
			textures.push_back(texture);
		}

		textureFileName = dataFolder + material.fileName + normalmapPostfix;
		texture = NULL;
		D3DXCreateTextureFromFileExA(dxDevice, textureFileName.c_str(), D3DX_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT, 0, D3DFMT_UNKNOWN, D3DPOOL_MANAGED, D3DX_DEFAULT, D3DX_DEFAULT, 0, NULL, NULL, &texture);
		if (texture)
		{
			material.texNormalsIndex = (int)textures.size();
			textures.push_back(texture);
		}
	}
}


void D3DMesh::SetAnimation(D3DAnimation* anim)
{
	animation = anim;
	for (int i = 0; i < (int)skinBones.size(); i++)
	{
		skinBones[i].indexInAnimation = animation->GetBoneIndex(skinBones[i].name.c_str());
	}

	for (int i = 0; i < (int)skinBones.size(); i++)
	{
		OutputDebugStringA(Utils::StringFormat("[%d->%d].Skin: %s\n", i, skinBones[i].indexInAnimation, skinBones[i].name.c_str()));
	}
	OutputDebugStringA("\n\n");

}

void D3DMesh::EvaluateAnimation(float deltaTime, int instanceID)
{
	ASSERT(animation, "Mesh don't have attached animation, can't evaluate animation");

	MeshInstance & meshInstance = meshInstances[instanceID];

	for (int boneIndex = 0; boneIndex < (int)skinBones.size(); boneIndex++)
	{
		const SkinBone & skinBone = skinBones[boneIndex];
		animation->UpdateBone(meshInstance.animationTime, boneIndex, skinBone.indexInAnimation, skinBone.invBindPoseTranslate, skinBone.invBindPoseRotate, instanceID);
	}

	meshInstance.animationTime += deltaTime * animation->GetFrameRate() * meshInstance.animationSpeed;
}

