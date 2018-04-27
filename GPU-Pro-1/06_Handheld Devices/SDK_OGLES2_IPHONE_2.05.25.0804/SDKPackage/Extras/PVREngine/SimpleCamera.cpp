/******************************************************************************

 @File         SimpleCamera.cpp

 @Title        Simple Camera

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  A simple yaw, pitch camera with fixed up vector

******************************************************************************/
#include "SimpleCamera.h"
#include "TimeController.h"
#include <math.h>


namespace pvrengine
{


	/******************************************************************************/

	SimpleCamera::SimpleCamera()
	{
		m_vPosition.x = m_vVelocity.x = f2vt(0.0f);
		m_vPosition.y = m_vVelocity.y = f2vt(0.0f);
		m_vPosition.z = m_vVelocity.z = f2vt(0.0f);
		m_fHeading = f2vt(0.0f);
		m_fElevation = f2vt(0.0f);
		m_fMoveSpeed = f2vt(5.0f);
		m_fRotSpeed = f2vt(0.01f);
		m_fFOV = f2vt(0.7f);
		m_bInverted = false;
	}

	/******************************************************************************/

	void SimpleCamera::updatePosition()
	{	
	// Most of this stuff is to try and smooth movement when controlled by the primitive keyboard input available

		PVRTVec3 vDec = m_vVelocity * f2vt(TimeController::inst().getDeltaTime()) * m_fMoveSpeed * f2vt(0.1f);

		while(vDec.lenSqr()>m_vVelocity.lenSqr())
		{
			vDec /= f2vt(2.0f);
		}

		m_vVelocity -= vDec;

		if(m_vVelocity.lenSqr()>m_fMoveSpeed*m_fMoveSpeed)
		{
			m_vVelocity = m_vVelocity.normalized()*m_fMoveSpeed;
		}
 		m_vPosition += m_vVelocity * f2vt((float)TimeController::inst().getDeltaTime());
	}

	/******************************************************************************/

	void SimpleCamera::setTarget(const PVRTVec3& vec)
	{
		PVRTVec3 vActual = m_vPosition - vec;
		setTo(vActual);
	}

	/******************************************************************************/

	void SimpleCamera::setTo(const PVRTVec3& vec)
	{

		// find angle from horizontal
		m_fElevation = f2vt((float) atan(VERTTYPEDIV(vec.y,f2vt(sqrt(vt2f(vec.z*vec.z+vec.x*vec.x))))));

		// find principle angle from straight ahead
		m_fHeading = f2vt((float) atan2(vt2f(vec.x),vt2f(vec.z)));

		m_fHeading -= PVRT_PI;

		while(m_fHeading < 0.0f)
			m_fHeading+=PVRT_TWO_PI;

	}

	/******************************************************************************/

	void SimpleCamera::getToAndUp(PVRTVec3& vTo, PVRTVec3& vUp) const
	{
		vTo = PVRTVec3(f2vt(0.0f),f2vt(0.0f),f2vt(1.0f));
		vUp = PVRTVec3(f2vt(0.0f),f2vt(1.0f),f2vt(0.0f));

		PVRTMat3 mRotY = PVRTMat3::RotationY(m_fHeading);
		PVRTMat3 mRotX = PVRTMat3::RotationX(m_fElevation);

		vTo = (vTo*mRotX)*mRotY;
		vUp = (vUp*mRotX)*mRotY;

	}

	/******************************************************************************/

	void SimpleCamera::getTargetAndUp(PVRTVec3& vTarget, PVRTVec3& vUp) const
	{
		vTarget = PVRTVec3(f2vt(0.0f),f2vt(0.0f),f2vt(1.0f));
		vUp = PVRTVec3(f2vt(0.0f),f2vt(1.0f),f2vt(0.0f));

		PVRTMat3 mRotY = PVRTMat3::RotationY(m_fHeading);
		PVRTMat3 mRotX = PVRTMat3::RotationX(m_fElevation);

		vTarget = (vTarget*mRotX)*mRotY;
		vUp = (vUp*mRotX)*mRotY;

		vTarget +=m_vPosition;
	}

	/******************************************************************************/

	PVRTVec3 SimpleCamera::getTo() const
	{
		PVRTVec3 vTo(f2vt(0.0f),f2vt(0.0f),f2vt(1.0f));

		PVRTMat3 mRotY = PVRTMat3::RotationY(m_fHeading);
		PVRTMat3 mRotX = PVRTMat3::RotationX(m_fElevation);

		vTo = (vTo*mRotX)*mRotY;

		return vTo;
	}

	/******************************************************************************/

	PVRTVec3 SimpleCamera::getUp() const
	{
		PVRTVec3 vUp(f2vt(0.0f),f2vt(1.0f),f2vt(0.0f));

		PVRTMat3 mRotY = PVRTMat3::RotationY(m_fHeading);
		PVRTMat3 mRotX = PVRTMat3::RotationX(m_fElevation);
		vUp = (vUp*mRotX)*mRotY;

		return vUp;
	}

	/******************************************************************************/

	PVRTVec3 SimpleCamera::getTarget() const
	{
		PVRTVec3 vTarget = getTo();
		vTarget.x += m_vPosition.x;
		vTarget.y += m_vPosition.y;
		vTarget.z += m_vPosition.z;

		return vTarget;
	}

	/******************************************************************************/

	void SimpleCamera::YawRight()
	{
		m_fHeading-=m_fRotSpeed;
		if(m_fHeading>=PVRT_TWO_PI)
			m_fHeading-=PVRT_TWO_PI;
	}

	/******************************************************************************/

	void SimpleCamera::YawLeft()
	{
		m_fHeading+=m_fRotSpeed;
		if(m_fHeading<0)
			m_fHeading+=PVRT_TWO_PI;
	}

	/******************************************************************************/

	void SimpleCamera::PitchUp()
	{
		if(m_bInverted)
		{
			m_fElevation+=m_fRotSpeed;
			if(m_fElevation>=PVRT_PI_OVER_TWO)
				m_fElevation=PVRT_PI_OVER_TWO - f2vt(0.001f);
		}
		else
		{
			m_fElevation-=m_fRotSpeed;
			if(m_fElevation<=-PVRT_PI_OVER_TWO)
				m_fElevation=-PVRT_PI_OVER_TWO + f2vt(0.001f);
		}
	}

	/******************************************************************************/

	void SimpleCamera::PitchDown()
	{
		if(m_bInverted)
		{
			m_fElevation-=m_fRotSpeed;
			if(m_fElevation<=-PVRT_PI_OVER_TWO)
				m_fElevation=-PVRT_PI_OVER_TWO + f2vt(0.001f);
		}
		else
		{
			m_fElevation+=m_fRotSpeed;
			if(m_fElevation>=PVRT_PI_OVER_TWO)
				m_fElevation=PVRT_PI_OVER_TWO - f2vt(0.001f);
		}
	}

	/******************************************************************************/

	void SimpleCamera::MoveForward()
	{
		PVRTVec3 vTo = getTo().normalized();
		m_vVelocity +=vTo*m_fMoveSpeed;
	}

	/******************************************************************************/

	void SimpleCamera::MoveBack()
	{
		PVRTVec3 vTo = getTo().normalized();
		m_vVelocity -=vTo*m_fMoveSpeed;
	}

}
/******************************************************************************
* Revisions
*
* $Log: SimpleCamera.cpp,v $
* Revision 1.24  2008/05/20 10:23:03  sas
* [INTERNAL-DEMOS] Update vector and matrix code.
*
* Revision 1.23  2008/05/19 10:09:44  sas
* [INTERNAL EXTRAS]
* Update to use changes to Tools.
*
* Revision 1.22  2008/02/11 17:06:51  gml
* [INTERNAL EXTRAS PVRENGINE] Tidied.
*
* Revision 1.21  2008/02/08 17:04:09  gml
* [INTERNAL EXTRAS PVRENGINE] Fixed setTarget() problem.
*
* Revision 1.20  2008/02/07 19:15:58  gml
* [INTERNAL EXTRAS PVRENGINE] Commented source
*
* Revision 1.19  2008/02/06 14:19:54  sbm
* [INTERNAL EXTRAS PVRENGINE PODPLAYER]
* - A fix to the precision modifiers in the default shaders to stop awful z-fighting.
* - Fixed some warnings.
* - Removed some unwanted stuff from the WindowsPC project files.
*
* Revision 1.18  2008/02/01 10:57:03  gml
* [INTERNAL EXTRAS PVRENGINE] Fixed some fixed point problems
*
* Revision 1.17  2008/01/31 15:57:39  sbm
* [INTERNAL EXTRAS PVRENGINE]
* - Fixes for PVREngine when built for a fixed point. Note: Just 'cause it now builds for fixed point platforms doens't mean it'll work.
*
* Revision 1.16  2008/01/30 14:23:21  sbm
* [EXTERNAL EXTRAS PVRENGINE PODPLAYER]
* - Adding Windows CE 6 CEPC make files for OpenGL.
* + Fixed a few warnings
*
* Revision 1.15  2008/01/30 11:46:22  gml
* [INTERNAL EXTRAS PVRENGINE] Reorganising in cvs again.
*
* Revision 1.2  2008/01/29 11:57:35  gml
* [INTERNAL EXTRAS PODPLAYER] Tried to fix settarget/setto functions
*
* Revision 1.1  2008/01/18 11:14:37  gml
* [INTERNAL EXTRAS PVRENGINE] Restructuring in repositrory.
*
* Revision 1.13  2008/01/17 12:26:52  gml
* [INTERNAL EXTRAS PVRENGINE] A lot of documentation, some tidy up.
*
* Revision 1.12  2008/01/03 16:56:06  gml
* [INTERNAL EXTRAS PVRENGINE] Changes for latest matrix/vector code.
*
* Revision 1.11  2007/12/11 16:24:50  gml
* [INTERNAL EXTRAS PVRENGINE]Changes to make OGLES s60 build work
*
* Revision 1.10  2007/11/29 17:04:33  gml
* [INTERNAL EXTRAS PVRENGINE] Made changes for integration into automatic builds
*
* Revision 1.9  2007/11/23 14:58:30  gml
* [INTERNAL EXTRAS PVRENGINE] Snapshot after tools check-in and before alterations to extensions handler and string class
*
* Revision 1.8  2007/11/22 14:36:50  gml
* [INTERNAL EXTRAS PVRENGINE] Added frame rate option - made skinning work in OGL
*
* Revision 1.7  2007/11/09 16:25:16  gml
* [INTERNAL EXTRAS PVRENGINE] MADE ENGINE WORK WITH NEW MATRIX/VECTOR CODE
*
* Revision 1.6  2007/11/01 14:40:33  gml
* [INTERNAL EXTRAS PVRENGINE] Fixed lighting in OGLES
*
* Revision 1.5  2007/10/05 10:12:55  gml
* [INTERNAL EXTRAS PVRENGINE] Added inverted option.
*
* Revision 1.4  2007/10/02 11:41:40  gml
* [INTERNAL EXTRAS PVRENGINE] Added frustum culling.
*
* Revision 1.3  2007/09/20 08:48:29  gml
* [INTERNAL EXTRAS PVRENGINE] Fixed perspective problem with camera at extreme elevations.
*
* Revision 1.2  2007/09/18 13:23:18  gml
* [INTERNAL EXTRAS PVRENGINE] Improved efficiency of unfirom loading so that frame uniforms are not calculated multiple times. Can support indexes 0-31 of unforms.
*
* Revision 1.1  2007/09/10 10:52:37  gml
* [INTERNAL EXTRAS PVRENGINE] Got basic camera working
*
* Revision 1.1  2007/08/29 11:55:34  gml
* [INTERNAL EXTRAS PVRENGINE] Initial addition to cvs.
*
******************************************************************************/
/******************************************************************************
End of file (SimpleCamera.cpp)
******************************************************************************/
