#ifndef PROVIDERS_FBXHELPERS_H_INCLUDED
#define PROVIDERS_FBXHELPERS_H_INCLUDED

#include "Math/Src/Forw.h"

#include "Forw.h"

#include "RawSkeletonData.h"

namespace fbxsdk_200901
{
	class KFbxImporter;
	class KFbxSdkManager;
	class KFbxMatrix;
	class KFbxVector2;
	class KFbxNode;
	class KFbxSkin;
	class KFbxScene;
}

namespace FBX_SDK_NAMESPACE = fbxsdk_200901;

namespace Mod
{
	template <typename T>
	class FBXWrapper : AntiValue
	{
	public:
		explicit FBXWrapper( T* a_ptr );
		FBXWrapper();
		~FBXWrapper();

		// construction/ destruction
	public:

		void Release();
		void Reset( T* a_ptr );

		T* operator->() const;
		T& operator* () const;

		T* Get() const;
		bool IsNull() const;

		// data
	private:
		T* ptr;
	};

	namespace FBX
	{
		Math::float4x4		AsFloat4x4( const FBX_SDK_NAMESPACE::KFbxMatrix& m);
		Math::float3		AsFloat3( const FBX_SDK_NAMESPACE::KFbxVector2& v );
		void				UndoMaxMatrix( Math::float3x4& mat );
		AnimationTrack*		ExtractAnimationFromNode(FBX_SDK_NAMESPACE::KFbxNode *node, bool useZeroKeysIfNoAnimation = false );

		struct FBXBoneNodeProxy : BoneNode
		{
			FBXBoneNodeProxy(const AnimationTrack &track, FBX_SDK_NAMESPACE::KFbxNode *proxy, const Math::float3x4& bindTransform, bool bindSet );
			FBX_SDK_NAMESPACE::KFbxNode *proxy;
		};

		typedef Types2< FBX_SDK_NAMESPACE::KFbxNode*, Math::float3x4 > :: Map NodeMatrixMap;
		typedef Types < FBXBoneNodeProxy > :: SharedPtr FBXBoneNodeProxyPtr;

		Math::float3x4 ExtractInvTransform( FBX_SDK_NAMESPACE::KFbxNode *meshNode );

		RawSkeletonDataPtr ExtractSkeletonData( NodeMatrixMap& oInvMatrices,  FBXBoneNodeProxyPtr& oProxyNode, FBX_SDK_NAMESPACE::KFbxSkin *skin, const Math::float3x4& invNodeTransform );
	}

}

#endif