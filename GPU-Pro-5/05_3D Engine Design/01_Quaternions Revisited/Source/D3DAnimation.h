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

#include <string>
#include <vector>
#include <hash_map>

#include "DXUT.h"
#include "Math.h"


class D3DAnimation
{

public:

	static const int BONE_INVALID_INDEX = -1;

	struct AnimationKey
	{
		Vector3 translate;
		Quaternion rotate;
	};

	static const int MAX_INSTANCES = 512;

private:

	IDirect3DDevice9* dxDevice;
	IDirect3DTexture9* cpuTexture;
	IDirect3DTexture9* gpuTexture;

	struct Bone
	{
		int parentIndex;
		int firstIndexOnTrack;
	};

	int framesCount;

	std::vector<AnimationKey> animationTracks;

	std::vector<Bone> bones;
	stdext::hash_map<std::string, int> boneNameToIndex;

	float frameRate;

	D3DLOCKED_RECT lrCpuTexture;
	int maxInstanceID;
	int maxBoneIndex;

public:

	D3DAnimation();
	~D3DAnimation();

	void Create(IDirect3DDevice9* dxDevice);
	void Destroy();

	void SetDuration(int framesCount);
	void SetFrameRate(float _frameRate);
	

	AnimationKey* CreateBone(const char* boneName, const char* boneParentName);

	void DeviceReset();
	void DeviceLost();

	void BeginUpdate();
	void EndUpdate();

	IDirect3DTexture9* GetAsTexture() const;

	int GetDuration() const;
	int GetBoneIndex(const char* boneName) const;
	int GetBonesCount() const ;
	const AnimationKey* GetBoneAnimationTrack(int boneIndex) const;
	AnimationKey* GetBoneAnimationTrack(int boneIndex);
	int GetBoneParentIndex(int boneIndex) const;

	void UpdateBone(float frame, int skinBoneIndex, int animationBoneIndex, const Vector3 & invBindPoseTranslate, const Quaternion & invBindPoseRotate, int meshInstanceID);

	float GetFrameRate();
};
