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
#pragma once

#include <vector>
#include "DXUT.h"
#include "VertexShader.h"
#include "PixelShader.h"
#include "Math.h"


class D3DAnimation;

class D3DMesh
{
public:

	static const int TECHNIQUES_COUNT = 16;

	struct VertexUnpackedTBN76
	{
		Vector3 pos;
		Vector2 uv;
		Vector3 normal;
		Vector3 tangent;
		Vector3 binormal;
		float blendWeights[4];
		unsigned char blendIndices[4];
	};

	struct VertexPackedTBN40
	{
		Vector3 pos;
		short uv[2];
		DWORD packedNormal;
		DWORD packedTangent;
		DWORD packedBinormal;
		short blendWeights[4];
		unsigned char blendIndices[4];
	};

	struct VertexUnpackedQuat44
	{
		Vector3 pos;
		short uv[2];
		float quaternionTBN[4];
		short blendWeights[4];
		unsigned char blendIndices[4];
	};

	struct VertexPackedQuat32
	{
		Vector3 pos;
		short uv[2];
		DWORD packedQuaternionTBN;
		short blendWeights[4];
		unsigned char blendIndices[4];
	};

	struct VertexNoSkinUnpackedTBN56
	{
		Vector3 pos;
		Vector2 uv;
		Vector3 normal;
		Vector3 tangent;
		Vector3 binormal;
	};

	struct VertexNoSkinPackedTBN28
	{
		Vector3 pos;
		short uv[2];
		DWORD packedNormal;
		DWORD packedTangent;
		DWORD packedBinormal;
	};

	struct VertexNoSkinUnpackedQuat32
	{
		Vector3 pos;
		short uv[2];
		float quaternionTBN[4];
	};

	struct VertexNoSkinPackedQuat20
	{
		Vector3 pos;
		short uv[2];
		DWORD packedQuaternionTBN;
	};


	enum VertexFormat
	{
		VFMT_UNPACKED_TBN = 0,
		VFMT_PACKED_TBN = 1,
		VFMT_UNPACKED_QUAT = 2,
		VFMT_PACKED_QUAT = 3,

		VFMT_UNPACKED_TBN_NOSKIN = 4,
		VFMT_PACKED_TBN_NOSKIN = 5,
		VFMT_UNPACKED_QUAT_NOSKIN = 6,
		VFMT_PACKED_QUAT_NOSKIN = 7,

		VFMT_MAX,
		VFMT_FORCE_DWORD = 0x7fffffff
	};

	struct InstanceData
	{
		unsigned short fp16Quaternion[4];
		unsigned short fp16Position[3];
		unsigned short fp16Scale[3];
		
		unsigned char rgbColor[3];
		unsigned char instanceID;
	};


	struct SkinBone
	{
		std::string name;
		int indexInAnimation;
		Vector3 invBindPoseTranslate;
		Quaternion invBindPoseRotate;
	};

	struct Material
	{
		std::string fileName;
		unsigned long startIndex;
		unsigned long trianglesCount;
		int texAlbedoIndex;
		int texNormalsIndex;
	};

private:

	struct MeshInstance
	{
		Quaternion q;
		Vector3 p;
		Vector3 s;

		float animationTime;
		float animationSpeed;

		MeshInstance()
		{
			q = Quaternion::Identity();
			p = Vector3(0.0f, 2.0f, 0.0f);
			s = Vector3(1.0f);
			animationTime = 0.0f;
			animationSpeed = 1.0f;
		}

		MeshInstance(float x, float z, float heightScale)
		{
			q = Quaternion::Identity();
			p = Vector3(x, 0.0f, z);
			s = Vector3(1.0f, heightScale, 1.0f);
			animationTime = Utils::FRand() * 20.0f;
			animationSpeed = Utils::FRand(0.75f, 1.13f);
		}
	};


	D3DAnimation* animation;
	
	std::vector<SkinBone> skinBones;
	std::vector<Material> materials;
	std::vector<IDirect3DTexture9*> textures;

	IDirect3DCubeTexture9* ambientCube;
	IDirect3DTexture9* middleGray;

	UINT vertexCount;
	UINT indexCount;

	IDirect3DVertexBuffer9* instanceVertexBuffer;
	IDirect3DVertexBuffer9* vertexBuffers[VFMT_MAX];
	IDirect3DIndexBuffer9* indexBuffer;
	IDirect3DDevice9* dxDevice;
	unsigned long verticesSizeof[VFMT_MAX];

	IDirect3DVertexDeclaration9* vertexDeclarations[VFMT_MAX];

	struct Technique
	{
		VertexShader vertexShader;
		PixelShader pixelShader;
	};

	Technique techniques[TECHNIQUES_COUNT];

	void CreateVertexDeclaration();
	void CreateInstanceBuffer();

	std::vector<MeshInstance> meshInstances;

	void EvaluateAnimation(float deltaTime, int instanceID);

public:

	D3DMesh();
	~D3DMesh();

	void Create(IDirect3DDevice9* dxDevice);
	void Destroy();

	void SetAnimation(D3DAnimation* anim);
	void Draw(const D3DXMATRIX & mtxViewProj, const D3DXVECTOR3& cameraDir, float deltaTime);


	void CreateMaterials(int materialsCount);
	D3DMesh::Material* MapMaterials();
	void UnmapMaterials();

	void CreateBones(int bonesCount);
	D3DMesh::SkinBone* MapBones();
	void UnmapBones();

	void CreateVertexBuffer(int verticesCount);

	void* MapVertexBuffer(D3DMesh::VertexFormat fmt);
	void UnmapVertexBuffer(D3DMesh::VertexFormat fmt);


	void CreateIndexBuffer(int indicesCount);
	WORD* MapIndexBuffer();
	void UnmapIndexBuffer();

	void CreateTextures();

	void DeviceLost();
	void DeviceReset();

};
