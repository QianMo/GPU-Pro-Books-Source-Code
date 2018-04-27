/******************************************************************************

 @File         SimpleCamera.h

 @Title        Simple Camera

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  A simple yaw, pitch camera with fixed up vector

******************************************************************************/
#ifndef SIMPLECAMERA_H
#define SIMPLECAMERA_H

/******************************************************************************
Includes
******************************************************************************/
#include "../PVRTools.h"

namespace pvrengine
{

	/*!****************************************************************************
	Class
	******************************************************************************/
	/*!***************************************************************************
	* @Class SimpleCamera
	* @Brief 	A simple yaw, pitch camera with fixed up vector.
	* @Description 	A simple yaw, pitch camera with fixed up vector.
	*****************************************************************************/
	class SimpleCamera
	{
	public:
		/*!***************************************************************************
		@Function			SimpleCamera
		@Description		blank constructor.
		*****************************************************************************/
		SimpleCamera();

		/*!***************************************************************************
		@Function			updatePosition
		@Description		updates the position of the camera according to its
		current velocity.
		*****************************************************************************/
		void			updatePosition();

		/*!***************************************************************************
		@Function			setPosition
		@Input				x	coordinate
		@Input				y	coordinate
		@Input				z	coordinate
		@Description		sets the position of the camera directly to the
		coordinates passed
		*****************************************************************************/
		void		setPosition(const VERTTYPE x, const VERTTYPE y, const VERTTYPE z)
		{
			m_vPosition.x = x;
			m_vPosition.y = y;
			m_vPosition.z = z;
		}

		/*!***************************************************************************
		@Function			setPosition
		@Input				vec	position coordinates
		@Description		sets the position of the camera directly to the
		coordinates passed
		*****************************************************************************/
		void			setPosition(const PVRTVec3& vec)
		{m_vPosition = vec;}

		/*!***************************************************************************
		@Function			setTarget
		@Input				vec	position coordinates
		@Description		sets the position of the point that the camera is looking
		at.
		*****************************************************************************/
		void			setTarget(const PVRTVec3& vec);

		/*!***************************************************************************
		@Function			setTo
		@Input				vec	position coordinates
		@Description		specifies a vector that describes the angle that the
		camera should be viewing
		*****************************************************************************/
		void			setTo(const PVRTVec3& vec);

		/*!***************************************************************************
		@Function			getPosition
		@Return				position coordinates
		@Description		gets the position of the camera
		*****************************************************************************/
		PVRTVec3		getPosition() const
		{
			return m_vPosition;
		}

		/*!***************************************************************************
		@Function			getToAndUp
		@Output				vTo direction camera is pointing
		@Output				vUp direction perpendicular to way camera is pointing
		@Description		Gets two vectors indicating the angle of the camera that
		can be used in view calculations
		*****************************************************************************/
		void			getToAndUp(PVRTVec3& vTo, PVRTVec3& vUp) const;

		/*!***************************************************************************
		@Function			getTargetAndUp
		@Output				vTo point at wich camera currently looks
		@Output				vUp direction perpendicular to way camera is pointing
		@Description		Gets two vectors indicating the angle of the camera that
		can be used in view calculations
		*****************************************************************************/
		void			getTargetAndUp(PVRTVec3& vTo, PVRTVec3& vUp) const;

		/*!***************************************************************************
		@Function			getTo
		@Return				direction camera is pointing
		@Description		Gets a vector indicating the angle of the camera that
		can be used in view calculations
		*****************************************************************************/
		PVRTVec3		getTo() const;

		/*!***************************************************************************
		@Function			getUp
		@Return				direction perpendicular to way camera is pointing
		@Description		Gets a vector indicating the angle of the camera that
		can be used in view calculations
		*****************************************************************************/
		PVRTVec3		getUp() const;

		/*!***************************************************************************
		@Function			getTarget
		@Return				point at wich camera currently looks
		@Description		Gets two vectors indicating the angle of the camera that
		can be used in view calculations
		*****************************************************************************/
		PVRTVec3		getTarget() const;

		/*!***************************************************************************
		@Function			getHeading
		@Return				heading value
		@Description		Retrieves heading value for this camera
		*****************************************************************************/
		VERTTYPE	getHeading() const
		{return m_fHeading;}

		/*!***************************************************************************
		@Function			getElevation
		@Return				elevation value
		@Description		Retrieves elevation value for this camera
		*****************************************************************************/
		VERTTYPE	getElevation() const
		{return m_fElevation;}

		/*!***************************************************************************
		@Function			getAspect
		@Return				Aspect ratio value
		@Description		Retrieves the aspect ratio for this camera
		*****************************************************************************/
		VERTTYPE	getAspect() const { return m_fAspect;}

		/*!***************************************************************************
		@Function			setAspect
		@Input				fAspect new aspect ratio value
		@Description		sets the aspect ratio value for this camera
		*****************************************************************************/
		void		setAspect(const VERTTYPE fAspect){ m_fAspect = fAspect;}

		/*!***************************************************************************
		@Function			setAspect
		@Input				fHeight height of view
		@Input				fWidth width of view
		@Description		sets the aspect ratio value for this camera
		*****************************************************************************/
		void		setAspect(const VERTTYPE fHeight, const VERTTYPE fWidth){ m_fAspect = VERTTYPEDIV(fWidth,fHeight);}

		/*!***************************************************************************
		@Function			getFOV
		@Return				Field of view value
		@Description		Retrieves the field of view value for this camera
		*****************************************************************************/
		VERTTYPE	getFOV() const
		{	return m_fFOV;	}

		/*!***************************************************************************
		@Function			setFOV
		@Input				fFOV new Field of view value
		@Description		sets the field of view value for this camera
		*****************************************************************************/
		void		setFOV(const VERTTYPE fFOV)
		{	m_fFOV = fFOV;	}

		/*!***************************************************************************
		@Function			setNear
		@Input				fNear near clipping value
		@Description		sets the near clipping depth value for this camera
		*****************************************************************************/
		void		setNear(const VERTTYPE fNear)
		{	m_fNear = fNear;	}
		/*!***************************************************************************
		@Function			getNear
		@Return				fNear near clipping value
		@Description		gets the near clipping depth value for this camera
		*****************************************************************************/
		VERTTYPE	getNear() const
		{	return m_fNear;	}

		/*!***************************************************************************
		@Function			setFar
		@Input				fFar far clipping value
		@Description		sets the far clipping depth value for this camera
		*****************************************************************************/
		void		setFar(const VERTTYPE fFar)
		{	m_fFar = fFar;	}

		/*!***************************************************************************
		@Function			getFar
		@Return				far clipping value
		@Description		gets the far clipping depth value for this camera
		*****************************************************************************/
		VERTTYPE	getFar() const
		{	return m_fFar;	}

		/*!***************************************************************************
		@Function			setMoveSpeed
		@Input				fMoveSpeed movement speed modifier
		@Description		sets the movement speed modifier value for this camera
		*****************************************************************************/
		void		setMoveSpeed(const VERTTYPE fMoveSpeed)
		{	m_fMoveSpeed = fMoveSpeed;	}

		/*!***************************************************************************
		@Function			getMoveSpeed
		@Return				movement speed modifier
		@Description		gets the movement speed modifier value for this camera
		*****************************************************************************/
		VERTTYPE	getMoveSpeed() const
		{	return m_fMoveSpeed;	}

		/*!***************************************************************************
		@Function			setRotSpeed
		@Input				fRotSpeed rotation speed modifier
		@Description		sets the rotation speed modifier value for this camera
		*****************************************************************************/
		void		setRotSpeed(const VERTTYPE fRotSpeed)
		{	m_fRotSpeed = fRotSpeed;	}

		/*!***************************************************************************
		@Function			getRotSpeed
		@Return				rotation speed modifier
		@Description		gets the rotation speed modifier value for this camera
		*****************************************************************************/
		VERTTYPE	getRotSpeed() const
		{	return m_fRotSpeed;	}

		/*!***************************************************************************
		@Function			setInverted
		@Input				bInverted
		@Description		sets whether up and down rotations should be reversed for
		this camera
		*****************************************************************************/
		void			setInverted(const bool bInverted)
		{	m_bInverted = bInverted; }

		/*!***************************************************************************
		@Function			getInverted
		@Return				bInverted
		@Description		gets whether up and down rotations are reversed for
		this camera
		*****************************************************************************/
		bool			getInverted() const
		{ return m_bInverted; }

		/*!***************************************************************************
		@Function			YawLeft
		@Description		Instructs camera to look left
		*****************************************************************************/
		void		YawLeft();

		/*!***************************************************************************
		@Function			YawRight
		@Description		Instructs camera to look right
		*****************************************************************************/
		void		YawRight();

		/*!***************************************************************************
		@Function			PitchUp
		@Description		Instructs camera to look up
		*****************************************************************************/
		void		PitchUp();

		/*!***************************************************************************
		@Function			PitchDown
		@Description		Instructs camera to look down
		*****************************************************************************/
		void		PitchDown();

		/*!***************************************************************************
		@Function			MoveForward
		@Description		Instructs camera to look up
		*****************************************************************************/
		void		MoveForward();

		/*!***************************************************************************
		@Function			MoveBack
		@Description		Instructs camera to look down
		*****************************************************************************/
		void		MoveBack();

	private:
		PVRTVec3	m_vPosition, m_vVelocity;	/*! where is the camera? how fast is it moving? */
		VERTTYPE	m_fHeading, m_fElevation;	/*! current angle of view */
		VERTTYPE	m_fMoveSpeed, m_fRotSpeed;	/*! multipliers for movement speed and rotation speed */
		VERTTYPE	m_fFOV, m_fAspect;			/*! field of view */
		VERTTYPE	m_fNear,m_fFar;				/*! clipping planes */
		bool		m_bInverted;				/*! is up and down inverted? */

	};

}

#endif // SIMPLECAMERA_H

/******************************************************************************
End of file (SimpleCamera.h)
******************************************************************************/
