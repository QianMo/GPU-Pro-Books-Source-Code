#include "FBXPrecompiled.h"

#include "Common/Src/auto_array.h"

#include "Math/Src/Operations.h"

#include "Providers.h"
#include "SkeletonRawDataImporterProvider.h"
#include "FBXSkeletonRawDataImporter.h"

#include "RawVertexData.h"
#include "TextureNamesSet.h"
#include "ModelCmptRawData.h"

#include "AnimationTrack.h"
#include "RawSkeletonData.h"

#include "FBXHelpers.cpp.h"

#include "FBXCommon.h"

#include "FBXModelCmptRawDataImporter.h"

namespace Mod
{

	using namespace FBX;
	using namespace FBXFILESDK_NAMESPACE;
	using namespace Math;

	//------------------------------------------------------------------