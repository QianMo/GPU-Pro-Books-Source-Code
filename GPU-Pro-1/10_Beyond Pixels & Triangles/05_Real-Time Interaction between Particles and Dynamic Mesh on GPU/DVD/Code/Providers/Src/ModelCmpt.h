#ifndef PROVIDERS_MODELCMPT_H_INCLUDED
#define PROVIDERS_MODELCMPT_H_INCLUDED

#include "Math/Src/BBox.h"
#include "Wrap3D/Src/Forw.h"

#include "Forw.h"

#include "ExportDefs.h"

#define MD_NAMESPACE ModelCmptNS
#include "ConfigurableImpl.h"

namespace Mod
{
	class ModelCmpt : public ModelCmptNS::ConfigurableImpl< ModelCmptConfig >
	{
		// types
	public:

		// construction/ destruction
	public:
		EXP_IMP explicit ModelCmpt( const ModelCmptConfig& cfg );
		EXP_IMP virtual ~ModelCmpt();

		// manipulation/ access
	public:
		EXP_IMP void	Bind( UINT32 effectLink, ESDT::EffectSubDefType type, const RenderParams& renderParams, const EntityParams& entityParams, const DevicePtr& dev );
		EXP_IMP void	Draw( const RenderParams& params, const DevicePtr& dev );
		EXP_IMP void	Unbind( const RenderParams& params, const EntityParams& entityParams, const DevicePtr& dev );

		EXP_IMP void	BindTransform( UINT32 effectLink, const RenderParams& params, const EntityParams& entityParams, const DevicePtr& dev );
		EXP_IMP void	Transform( const DevicePtr& dev );
		EXP_IMP void	UnbindTransform( const DevicePtr& dev );

		EXP_IMP const Math::BBox&	GetBBox() const;

		EXP_IMP UINT32	GetEffectLink( const EffectDefPtr& effDef ) const;

		EXP_IMP ModelCmptType::Type GetModelCmptType() const;

		EXP_IMP bool	HasFlag( MCF::ModelComponentFlag flag ) const;

		EXP_IMP UINT64	GetVertexCount() const;

	protected:
		EXP_IMP void SetBBox( const Math::BBox& bbox );

		// polymorphism
	private:
		virtual void			BindImpl( const EntityParams& entityParams, const DevicePtr& dev ) = 0;
		virtual void			DrawDefImpl( const DevicePtr& dev ) = 0;
		virtual void			DrawMPIImpl( UINT32 numPasses, const DevicePtr& dev ) = 0;
		virtual void			UnbindImpl( const DevicePtr& dev ) = 0;

		virtual void			BindTransformImpl( const EntityParams& entityParams, const DevicePtr& dev ) = 0;
		virtual void			TransformImpl( const DevicePtr& dev ) = 0;
		virtual void			UnbindTransformImpl( const DevicePtr& dev ) = 0;

		virtual ModelVertexBuffers*		GetVertexBuffersImpl() const = 0;

		virtual ModelCmptType::Type		GetModelCmptTypeImpl() const = 0;
		virtual UINT64					GetVertexCountImpl() const = 0;

		Math::BBox				mBBox;
	};
}

#endif