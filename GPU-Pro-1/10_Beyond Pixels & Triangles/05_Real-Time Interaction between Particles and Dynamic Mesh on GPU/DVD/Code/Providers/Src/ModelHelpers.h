#ifndef PROVIDERS_MODELHELPERS_H_INCLUDED
#define PROVIDERS_MODELHELPERS_H_INCLUDED

#include "WrapSys/Src/Forw.h"
#include "Math/Src/Forw.h"
#include "Common/Src/Forw.h"
#include "Wrap3D/Src/Forw.h"
#include "Forw.h"

namespace Mod
{
	BufferPtr		CreateModelIndexBuffer( const DevicePtr& dev, const String& baseKey );
	void			CreateCachedModelIndexBuffer( const RawVertexDataPtr& rvd, const String& baseKey );
	Math::BBox		GetModelBBox( const RawVertexDataPtr& rvd );
	Math::BBoxVec	GetSkeletalBBoxes( const RawVertexDataPtr& rvd );
	void			BindModelVertexBuffers(	const EntityParams& entityParams, const ModelVertexBuffers& buffers, const DevicePtr& dev );
	void			UnbindModelVertexBuffers( const EntityParams& entityParams, const ModelVertexBuffers& buffers, const DevicePtr& dev );
	
	void			WriteBBox( const WFilePtr& file, const Math::BBox& bbox );
	void			ReadBBox( const RFilePtr& file, Math::BBox& oBBox );

	void			WriteBBoxVec( const WFilePtr& file, const Math::BBoxVec& bboxes );
	void			ReadBBoxVec( const RFilePtr& file, Math::BBoxVec& oBBoxes );


}

#endif