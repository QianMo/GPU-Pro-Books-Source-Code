#ifndef PROVIDERS_MODELCMPTRAWDATA_H_INCLUDED
#define PROVIDERS_MODELCMPTRAWDATA_H_INCLUDED

#include "Forw.h"

namespace Mod
{
	struct ModelCmptRawData
	{
		static const UINT32 NO_SKELETON_LINK = UINT32(-1);

		String				name;
		RawVertexDataPtr	rawVertexData;
		TextureNamesSetPtr	textureNamesSet;

		// skeleton index in recently imported skeleton array ( if such was co-imported )
		UINT32				skeletonIdx;

		ModelCmptRawData();
	};

	//------------------------------------------------------------------------

	inline
	ModelCmptRawData::ModelCmptRawData() :
	skeletonIdx( NO_SKELETON_LINK )
	{

	}
}

#endif