/******************************************************************************

 @File         TimeController.h

 @Title        Time Controller

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Keeps track of time since last frame, direction of animation etc.

******************************************************************************/
#ifndef TIMECONTROLLER_H
#define TIMECONTROLLER_H

/******************************************************************************
Includes
******************************************************************************/
#include "../PVRTSingleton.h"

namespace pvrengine
{
	/*!***************************************************************************
	* @Class TimeController
	* @Brief 	Time Controller: Controller of Time.
	* @Description 	Time Controller: Controller of Time.
	*****************************************************************************/
	class TimeController : public CPVRTSingleton<TimeController>
	{
	public:
		/*!***************************************************************************
		@Function			TimeController
		@Description		blank constructor.
		*****************************************************************************/
		TimeController();

		/*!***************************************************************************
		@Function			setNumFrames
		@Input				i32NumFrames	number of frames of animation
		@Description		sets the number of frames for the animation to loop over
		*****************************************************************************/
		void	setNumFrames(const int i32NumFrames)	{m_i32NumFrames = i32NumFrames;}

		/*!***************************************************************************
		@Function			setAnimationSpeed
		@Input				fAnimationSpeed	speed of animation multiplier
		@Description		sets the speed of animation desired
		*****************************************************************************/
		void	setAnimationSpeed(const float fAnimationSpeed){	m_fAnimationSpeed = fAnimationSpeed;}

		/*!***************************************************************************
		@Function			getAnimationSpeed
		@Return				the animation speed multiplier
		@Description		Retrieve the animation speed value
		*****************************************************************************/
		float		getAnimationSpeed(){return m_fAnimationSpeed;}

		/*!***************************************************************************
		@Function			setFrame
		@Input				fFrame	animation frame
		@Description		sets the current frame for the animation
		*****************************************************************************/
		void	setFrame(const float fFrame){	m_fFrame = fFrame;}

		/*!***************************************************************************
		@Function			getFrame
		@Input				iTime the current time
		@Return				animation frame
		@Description		retrieve the current frame for the animation
		*****************************************************************************/
		float	getFrame(const int iTime);

		/*!***************************************************************************
		@Function			getFPS
		@Return				frames per second
		@Description		retrieves the calculated frames per second value
		*****************************************************************************/
		float	getFPS()	{return (float)m_i32FPS;}

		/*!***************************************************************************
		@Function			getForwards
		@Return				true forwards, false backwards
		@Description		returns whether animation is set to go forwards or backwards
		*****************************************************************************/
		bool	getForwards(){return m_bForwards;}

		/*!***************************************************************************
		@Function			setForwards
		@Input				bForwards true forwards, false backwards
		@Description		sets whether animation is to go forwards or backwards
		*****************************************************************************/
		void	setForwards(const bool bForwards){m_bForwards = bForwards;}

		/*!***************************************************************************
		@Function			getDeltaTime
		@Returns			delta time value
		@Description		retrieve delta time value to allow frame independent
		calculations
		*****************************************************************************/
		float	getDeltaTime(){return m_fDeltaTime;}

		/*!***************************************************************************
		@Function			setFreezeTime
		@Input				bFreeze true for pause, false to run
		@Description		set whether the time controller should advance time or not
		*****************************************************************************/
		void	setFreezeTime(const bool bFreeze){m_bFrozen = bFreeze;}

		/*!***************************************************************************
		@Function			start
		@Input				iTime for start of demo - usually the current time from
		the PVRShell
		@Description		starts the timecontroller handling time; mandatory
		*****************************************************************************/
		void	start(int iTime){m_iTimePrev = iTime;}

		/*!***************************************************************************
		@Function			rewind
		@Description		rewinds the frame number for the current animation at a
		higher than normal rate
		*****************************************************************************/
		void rewind();

		/*!***************************************************************************
		@Function			fastforward
		@Description		fastforwards the frame number for the current animation at a
		higher than normal rate
		*****************************************************************************/
		void fastforward();

	private:
		/*! Variables to handle animation in a time-based manner */
		int				m_iTimePrev;	/*! previous time to calculate delta time from */
		float			m_fFrame, m_fDeltaTime, m_fAnimationSpeed;	/*! current frame, current delta time, animation speed multiplier */
		int				m_i32FrameCount, m_i32DeltaTimeTotal, m_i32FPS;	/*! variables for calculating FPS */
		int				m_i32NumFrames ;	/*! number of frames to loop around */
		bool			m_bFrozen, m_bForwards;	/*! freeze animation, play forwards or backwards */

	};
}
#endif // TIMECONTROLLER_H

/******************************************************************************
End of file (TimeController.h)
******************************************************************************/
