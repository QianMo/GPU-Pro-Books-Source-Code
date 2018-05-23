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
#include "D3DAnimation.h"
#include "Convertors.h"

//FP32 vs FP16 format selector
#define HALF_PRECISION_ANIMATION (1)


D3DAnimation::D3DAnimation()
{
	frameRate = 1.0f;
	maxBoneIndex = 0;
	maxInstanceID = 0;
	dxDevice = NULL;
	framesCount = 0;
	cpuTexture = NULL;
	gpuTexture = NULL;
}

D3DAnimation::~D3DAnimation()
{
	Destroy();
}

void D3DAnimation::Create(IDirect3DDevice9* _dxDevice)
{
	Destroy();
	dxDevice = _dxDevice;

#ifdef HALF_PRECISION_ANIMATION
	D3DFORMAT animationTextureFormat = D3DFMT_A16B16G16R16F;
#else
	D3DFORMAT animationTextureFormat = D3DFMT_A32B32G32R32F;
#endif

	HRESULT hr = D3D_OK;
	hr = dxDevice->CreateTexture(512, MAX_INSTANCES, 1, D3DUSAGE_DYNAMIC, animationTextureFormat, D3DPOOL_SYSTEMMEM, &cpuTexture, 0);
	ASSERT(hr == D3D_OK, "D3DAnimation: Can't create system memory texture");
}

void D3DAnimation::Destroy()
{
	animationTracks.clear();
	bones.clear();
	boneNameToIndex.clear();

	SAFE_RELEASE(cpuTexture);
	SAFE_RELEASE(gpuTexture);
}

void D3DAnimation::DeviceReset()
{
#ifdef HALF_PRECISION_ANIMATION
	D3DFORMAT animationTextureFormat = D3DFMT_A16B16G16R16F;
#else
	D3DFORMAT animationTextureFormat = D3DFMT_A32B32G32R32F;
#endif

	ASSERT(!gpuTexture, "Animation texture already exist, can't create animation texture twice");
	HRESULT hr = dxDevice->CreateTexture(512, MAX_INSTANCES, 1, 0, animationTextureFormat, D3DPOOL_DEFAULT, &gpuTexture, 0);
	ASSERT(hr == D3D_OK, "D3DAnimation: Can't create video memory texture");
}

void D3DAnimation::DeviceLost()
{
	SAFE_RELEASE(gpuTexture);
}


int D3DAnimation::GetBoneIndex(const char* boneName) const
{
	auto it = boneNameToIndex.find(boneName);
	if (it == boneNameToIndex.end())
	{
		return BONE_INVALID_INDEX;
	}
	return it->second;
}

void D3DAnimation::SetDuration(int _framesCount)
{
	framesCount = _framesCount;
}

int D3DAnimation::GetDuration() const
{
	return framesCount;
}

void D3DAnimation::SetFrameRate(float _frameRate)
{
	frameRate = _frameRate;
}

D3DAnimation::AnimationKey* D3DAnimation::CreateBone(const char* boneName, const char* boneParentName)
{
	ASSERT(framesCount > 0, "You must call D3DAnimation::SetDuration before calling CreateBone");
	ASSERT(boneNameToIndex.find(boneName) == boneNameToIndex.end(), "Same bone already exist");

	boneNameToIndex[boneName] = (int)bones.size();

	Bone bone;
	bone.firstIndexOnTrack = (int)animationTracks.size();

	if (boneParentName)
	{
		bone.parentIndex = GetBoneIndex(boneParentName);
		ASSERT(bone.parentIndex >= 0, "Can't find parent bone index");
	} else
	{
		bone.parentIndex = BONE_INVALID_INDEX;
	}
	bones.push_back(bone);

	animationTracks.resize(animationTracks.size() + framesCount);

	D3DAnimation::AnimationKey* boneTrackData = &animationTracks[bone.firstIndexOnTrack];
	return boneTrackData;
}

const D3DAnimation::AnimationKey* D3DAnimation::GetBoneAnimationTrack(int boneIndex) const
{
	return &animationTracks[ bones[boneIndex].firstIndexOnTrack ];
}

D3DAnimation::AnimationKey* D3DAnimation::GetBoneAnimationTrack(int boneIndex)
{
	return &animationTracks[ bones[boneIndex].firstIndexOnTrack ];
}

int D3DAnimation::GetBoneParentIndex(int boneIndex) const
{
	return bones[boneIndex].parentIndex;
}

int D3DAnimation::GetBonesCount() const
{
	return (int)bones.size();
}

void D3DAnimation::UpdateBone(float frame, int skinBoneIndex, int animationBoneIndex, const Vector3 & invBindPoseTranslate, const Quaternion & invBindPoseRotate, int meshInstanceID)
{
	ASSERT(meshInstanceID < MAX_INSTANCES, "meshInstanceID is too big");

	maxBoneIndex = Utils::Max(maxBoneIndex, skinBoneIndex);

	int frameA = ((int)frame) % framesCount;
	int frameB = (frameA + 1) % framesCount;
	float blendK = frame - floor(frame);

	maxInstanceID = meshInstanceID;

	int firstIndexOnTrack = bones[animationBoneIndex].firstIndexOnTrack;

	const AnimationKey & keyA = animationTracks[firstIndexOnTrack + frameA];
	const AnimationKey & keyB = animationTracks[firstIndexOnTrack + frameB];

	// Linear interpolate position between frames
	Vector3 boneTranslate = lerp(keyA.translate, keyB.translate, blendK);

	// Linear interpolate rotation between frames
	Quaternion boneRotate = lerp(keyA.rotate, keyB.rotate, blendK);

	// Spherical linear interpolate rotation between frames
	//Quaternion q = slerp(keyA.rotate, keyB.rotate, blendK);

	boneRotate.Normalize();

	Vector3 translate = boneTranslate + boneRotate * invBindPoseTranslate;
	Quaternion rotate = boneRotate * invBindPoseRotate;

#ifdef HALF_PRECISION_ANIMATION

	unsigned short* animationData = (unsigned short*)((unsigned char*)lrCpuTexture.pBits + lrCpuTexture.Pitch * meshInstanceID);
	int boneDataOffset = skinBoneIndex * 4 * 2;

	animationData[boneDataOffset + 0] = Convert::Fp32ToFp16(translate.x);
	animationData[boneDataOffset + 1] = Convert::Fp32ToFp16(translate.y);
	animationData[boneDataOffset + 2] = Convert::Fp32ToFp16(translate.z);
	//animationData[boneDataOffset + 3] = Convert::Fp32ToFp16(0.0f);
	animationData[boneDataOffset + 4] = Convert::Fp32ToFp16(rotate.x);
	animationData[boneDataOffset + 5] = Convert::Fp32ToFp16(rotate.y);
	animationData[boneDataOffset + 6] = Convert::Fp32ToFp16(rotate.z);
	animationData[boneDataOffset + 7] = Convert::Fp32ToFp16(rotate.w);
#else

	float* animationData = (float*)((unsigned char*)lrCpuTexture.pBits + lrCpuTexture.Pitch * meshInstanceID);
	int boneDataOffset = skinBoneIndex * 4 * 2;

	animationData[boneDataOffset + 0] = translate.x;
	animationData[boneDataOffset + 1] = translate.y;
	animationData[boneDataOffset + 2] = translate.z;
	//animationData[boneDataOffset + 3] = 0.0f;
	animationData[boneDataOffset + 4] = rotate.x;
	animationData[boneDataOffset + 5] = rotate.y;
	animationData[boneDataOffset + 6] = rotate.z;
	animationData[boneDataOffset + 7] = rotate.w;
#endif
}

void D3DAnimation::BeginUpdate()
{
	if (!cpuTexture)
		return;

	HRESULT hr = cpuTexture->LockRect(0, &lrCpuTexture, NULL, D3DLOCK_DISCARD | D3DLOCK_NO_DIRTY_UPDATE);
	ASSERT(hr == D3D_OK, "Can't lock animation texture");

	maxInstanceID = -1;
	maxBoneIndex = 0;
}

void D3DAnimation::EndUpdate()
{
	if (!cpuTexture)
		return;

	RECT dirtyRect;
	dirtyRect.left = 0;
	dirtyRect.right = 512;
	dirtyRect.top = 0;
	dirtyRect.bottom = maxInstanceID+1;

	HRESULT hr = D3D_OK;

	hr = cpuTexture->AddDirtyRect( &dirtyRect );
	ASSERT(hr == D3D_OK, "Can't add dirty rect");

	hr = cpuTexture->UnlockRect(0);
	ASSERT(hr == D3D_OK, "Can't unlock animation texture");

	if (maxInstanceID < 0)
		return;

	hr = dxDevice->UpdateTexture( cpuTexture, gpuTexture );
	ASSERT(hr == D3D_OK, "Can't update animation texture");
}

IDirect3DTexture9* D3DAnimation::GetAsTexture() const
{
	return gpuTexture;
}

float D3DAnimation::GetFrameRate()
{
	return frameRate;
}

