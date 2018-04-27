#ifndef PROVIDERS_INDEXEDMODELCMPT_H_INCLUDED
#define PROVIDERS_INDEXEDMODELCMPT_H_INCLUDED

#include "ModelCmpt.h"

namespace Mod
{
	class IndexedModelCmpt : public ModelCmpt
	{
		// types
	public:
		typedef ModelCmpt Base;

		// construction/ destruction
	public:
		EXP_IMP explicit IndexedModelCmpt( const VertexIndexModelCmptConfig& cfg );
		EXP_IMP virtual ~IndexedModelCmpt();

		// manipulation/ access
	public:

		void ReadCachedData( const RFilePtr& modelCacheFile );

		static void CreateCachedData( const WFilePtr& modelCacheFile, const ModelCmptDefPtr& def, const RawVertexDataPtr& rawVertexData, const String& baseKey );

		// polymorphism
	private:
		virtual void BindImpl( const EntityParams& entityParams, const DevicePtr& dev ) OVERRIDE;
		virtual void DrawDefImpl( const DevicePtr& dev ) OVERRIDE;
		virtual void DrawMPIImpl( UINT32 numPasses, const DevicePtr& dev ) OVERRIDE;
		virtual void UnbindImpl( const DevicePtr& dev ) OVERRIDE;

		virtual void BindTransformImpl( const EntityParams& entityParams, const DevicePtr& dev ) OVERRIDE;
		virtual void TransformImpl( const DevicePtr& dev ) OVERRIDE;
		virtual void UnbindTransformImpl( const DevicePtr& dev ) OVERRIDE;

		virtual ModelVertexBuffers*		GetVertexBuffersImpl() const OVERRIDE;
		virtual ModelCmptType::Type		GetModelCmptTypeImpl() const OVERRIDE;
		virtual UINT64					GetVertexCountImpl() const OVERRIDE;

		// data
	private:
		ModelVertexBuffersPtr mVertexBuffers;
		BufferPtr mIndexBuffer;
	};
}

#endif