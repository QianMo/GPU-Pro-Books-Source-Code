#ifndef EXTRALIB_TRANSFORMCONTROLLER_H_INCLUDED
#define EXTRALIB_TRANSFORMCONTROLLER_H_INCLUDED

#include "SceneRender/Src/Forw.h"

#include "Forw.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE TransformControllerNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class TransformController : public TransformControllerNS::ConfigurableImpl<TransformControllerConfig>
	{
		// types
	public:
		typedef Parent				Base;
		typedef TransformController	Parent;

		// constructors / destructors
	public:
		explicit TransformController( const TransformControllerConfig& cfg );
		~TransformController();
	
		// manipulation/ access
	public:
		void	Update( float delta );

		// t must be  in [0,1]
		void	SetPosition( float t );
		float	GetPosition() const;
		void	UpdateNode( const NodePtr& node ) const;

		// align internal values according to current node transform
		void	Align( const NodePtr& node );

		// for VGUI stuff
		void	AccountAspectRatio( float aspect );

		bool	IsFinished() const;

		// polymorphism
	private:
		virtual void SetPositionImpl( float t )							= 0;
		virtual void UpdateNodeImpl( const NodePtr& node ) const		= 0;
		virtual void AlignImpl( float t, const NodePtr& node )			= 0;
		virtual void AccountAspectRatioImpl( float aspect )				= 0;

		// data
	private:
		float	mT;
		float	mSpeed;
		bool	mLoop;
	};
}

#endif