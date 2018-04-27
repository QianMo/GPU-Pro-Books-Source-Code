#ifndef EXTRALIB_CAMERACONTROLLER_H_INCLUDED
#define EXTRALIB_CAMERACONTROLLER_H_INCLUDED

#include "Forw.h"

#include "BlankExpImp.h"

#define MD_NAMESPACE CameraControllerNS
#include "ConfigurableImpl.h"

namespace Mod
{

	class CameraController : public CameraControllerNS::ConfigurableImpl<CameraControllerConfig>
	{
		// types
	public:
		typedef Parent Base;
		typedef CameraController Parent;

		// constructors / destructors
	public:
		explicit CameraController( const CameraControllerConfig& cfg );
		virtual ~CameraController();
	
		// manipulation/ access
	public:
		void Update( float dt );

		// polymorphism
	private:
		virtual void UpdateImpl( float dt ) = 0;

		// data
	private:

	};
}

#endif