#include "FBXPrecompiled.h"

#include "FBXHelpers.cpp.h"

#include "FBXCommon.h"

#define MD_NAMESPACE FBXCommonNS
#include "Singleton.cpp.h"

namespace Mod
{
	using namespace FBXFILESDK_NAMESPACE;

	template class FBXCommonNS::Singleton<FBXCommon>;

	//------------------------------------------------------------------------

	FBXCommon::FBXCommon()
	{
		mSDKManager.Reset( KFbxSdkManager::Create() );
		MD_FERROR_ON_TRUE( mSDKManager.IsNull() );

		mImporter.Reset( KFbxImporter::Create(mSDKManager.Get(),"") );
		MD_FERROR_ON_TRUE( mSDKManager.IsNull() );
	}

	//------------------------------------------------------------------------

	FBXCommon::~FBXCommon()
	{

	}

	//------------------------------------------------------------------------

	FBXCommon::SDKManager*
	FBXCommon::GetSDKManager() const
	{
		return &*mSDKManager;
	}

	//------------------------------------------------------------------------

	FBXCommon::Importer*
	FBXCommon::GetImporter()	const
	{
		return &*mImporter;
	}

	//------------------------------------------------------------------------

	FBXCommon::Scene*
	FBXCommon::ImportScene( const String& fileName ) const
	{
		int lFileFormat = -1;

		const AnsiString& aFileName = ToAnsiString( fileName );

		FBXCommon::SDKManager* SDKManager	= GetSDKManager();
		FBXCommon::Importer* importer		= GetImporter();

		if (!SDKManager->GetIOPluginRegistry()->DetectFileFormat( aFileName.c_str(), lFileFormat) )
		{
			// Unrecognizable file format. Try to fall back to KFbxImporter::eFBX_BINARY
			lFileFormat = SDKManager->GetIOPluginRegistry()->FindReaderIDByDescription( "FBX binary (*.fbx)" );
		}

		importer->SetFileFormat(lFileFormat);

		if(!importer->Initialize(aFileName.c_str()))
			MD_THROW( L"FBXCommon::ImportScene: failed to open file " + fileName );

		KFbxScene* scene( KFbxScene::Create(&*SDKManager,"") );
		if( !scene )
			MD_THROW( L"FBXCommon::ImportScene: Cant create scene!" );

		if( !importer->Import(&*scene) )
			MD_THROW( L"FBXCommon::ImportScene: Cant import scene!" );

		return scene;
	}

	//------------------------------------------------------------------------

	FBXCommon::ExtractedSkelMap&
	FBXCommon::GetExtractedSkeletonMap()
	{
		return mExtractedSkeletonMap;
	}

	//------------------------------------------------------------------------

	/*
	struct FBXCommon::ImportSettings
	{
	bool undoMaxCoordinates;
	bool truncateTexcoords;
	bool transformSkeltonVertsToWorld;
	float scale;
	};
	*/		
	
	/*static*/
	FBXCommon::ImportSettings FBXCommon::SETTINGS = {true, false, true, 1};


}