#ifndef PROVIDERS_FBXCOMMON_H_INCLUDED
#define PROVIDERS_FBXCOMMON_H_INCLUDED

#include "FBXHelpers.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE FBXCommonNS
#include "Singleton.h"

namespace Mod
{
	class FBXCommon : public FBXCommonNS::Singleton<FBXCommon>
	{
		// types & constants
	public:
		typedef FBX_SDK_NAMESPACE::KFbxSdkManager	SDKManager;
		typedef FBX_SDK_NAMESPACE::KFbxImporter		Importer;
		typedef FBX_SDK_NAMESPACE::KFbxScene		Scene;

		struct ImportSettings
		{
			bool undoMaxCoordinates;
			bool truncateTexcoords;
			bool transformSkeltonVertsToWorld;
			float scale;
		} static SETTINGS;

		typedef Types2 < FBX_SDK_NAMESPACE::KFbxNode*, RawSkeletonDataPtr > :: Map ExtractedSkelMap;

		// construction/ destruction
	public:
		FBXCommon();
		~FBXCommon();

		// manipulation/ access
	public:
		SDKManager*			GetSDKManager()		const;
		Importer*			GetImporter()		const;

		Scene*				ImportScene( const String& fileName ) const;

		ExtractedSkelMap&	GetExtractedSkeletonMap();

		// data
	private:

		// order matters - importer to be autodestroyed first
		FBXWrapper< SDKManager	>	mSDKManager;
		FBXWrapper< Importer	>	mImporter;

		ExtractedSkelMap			mExtractedSkeletonMap;
	};
}

#endif