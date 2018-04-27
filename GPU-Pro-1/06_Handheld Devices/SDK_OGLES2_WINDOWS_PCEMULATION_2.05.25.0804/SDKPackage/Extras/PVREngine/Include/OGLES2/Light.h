/******************************************************************************

 @File         Light.h

 @Title        Light

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Class to hold a Light with some convenient
               constructors/functions/operators

******************************************************************************/
#ifndef LIGHT_H
#define LIGHT_H

/******************************************************************************
Includes
******************************************************************************/

#include "../PVRTools.h"


namespace pvrengine
{
	/******************************************************************************
	Enums
	******************************************************************************/
	enum eLightType{
		eLight_Point=0,
		eLight_Directional,
		eLight_Spot				// Not supported by POD yet
	};

	/*!***************************************************************************
	* @Class Light
	* @Brief Class to hold a Light
	* @Description Class to hold a Light
	*****************************************************************************/
	class Light
	{
	public:
		/*!***************************************************************************
		@Function			Light
		@Description		blank constructor.
		*****************************************************************************/
		Light(){}

		/*!***************************************************************************
		@Function			shineLight
		@Input				i32Index - index of the light to shine
		@Description		sets up a hardware light (not functional in OpenGL)
		*****************************************************************************/
		virtual void shineLight(unsigned int i32Index)=0;

		/*!***************************************************************************
		@Function			getColour
		@Returns			the colour of the light
		@Description		retrieves the colour of this light
		*****************************************************************************/
		PVRTVec3 getColour(){return m_v3Colour;}

		/*!***************************************************************************
		@Function			getType
		@Return				type of the light
		@Description		gets the type of this light.
		*****************************************************************************/
		eLightType getType(){return m_eLightType;}
	protected:
		PVRTVec3 m_v3Colour;		/*!  colour of this light */
		eLightType m_eLightType;	/*!  type of light */

	};

	/*!***************************************************************************
	* @Class LightPoint
	* @Brief Class to hold a Point Light
	* @Description Class to hold a Point Light
	*****************************************************************************/
	class LightPoint : public Light
	{
	public:
		/*!***************************************************************************
		@Function			LightPoint
		@Input				v3Position - position of the new light
		@Input				v3Colour - colour of the light
		@Description		Constructor.
		*****************************************************************************/
		LightPoint(const PVRTVec3 v3Position, const PVRTVec3 v3Colour);

		/*!***************************************************************************
		@Function			LightPoint
		@Input				v4Position - position of the new light
		@Input				v3Colour - colour of the light
		@Description		Constructor.
		*****************************************************************************/
		LightPoint(const PVRTVec4 v4Position, const PVRTVec3 v3Colour);

		/*!***************************************************************************
		@Function			setPosition
		@Input				v4Position - position of the light
		@Description		Sets the position of the light.
		*****************************************************************************/
		void setPosition(const PVRTVec4 v4Position);

		/*!***************************************************************************
		@Function			setPosition
		@Input				v3Position - position of the light
		@Description		Sets the position of the light.
		*****************************************************************************/
		void setPosition(const PVRTVec3 v3Position);

		/*!***************************************************************************
		@Function			getPositionPVRTVec4
		@Return				position of the light
		@Description		gets the position of the light.
		*****************************************************************************/
		PVRTVec4 getPositionPVRTVec4(){return m_v4Position;}

		/*!***************************************************************************
		@Function			getPositionPVRTVec3
		@Return				position of the light
		@Description		gets the position of the light.
		*****************************************************************************/
		PVRTVec3 getPositionPVRTVec3(){return PVRTVec3(m_v4Position.x,m_v4Position.y,m_v4Position.z);}

		/*!***************************************************************************
		@Function			shineLight
		@Input				u32Index - index of the light to shine
		@Description		sets up a hardware light (not functional in OpenGL)
		*****************************************************************************/
		void shineLight(unsigned int u32Index);

	protected:
		PVRTVec4 m_v4Position;		/*! position of this light */
	};

	/*!***************************************************************************
	* @Class LightDirectional
	* @Brief Class to hold a Directional Light
	* @Description Class to hold a Directional Light
	*****************************************************************************/
	class LightDirectional : public Light
	{
	public:
		/*!***************************************************************************
		@Function			LightDirectional
		@Input				v3Direction - direction of the new light
		@Input				v3Colour - colour of the light
		@Description		Constructor.
		*****************************************************************************/
		LightDirectional(const PVRTVec3 v3Direction, const PVRTVec3 v3Colour);

		/*!***************************************************************************
		@Function			LightDirectional
		@Input				v4Direction - direction of the new light
		@Input				v3Colour - colour of the light
		@Description		Constructor.
		*****************************************************************************/
		LightDirectional(const PVRTVec4 v4Direction, const PVRTVec3 v3Colour);

		/*!***************************************************************************
		@Function			setDirection
		@Input				v3Direction - direction of the light
		@Description		Sets the direction of the light.
		*****************************************************************************/
		void setDirection(const PVRTVec3 v3Direction);

		/*!***************************************************************************
		@Function			setDirection
		@Input				v4Direction - direction of the light
		@Description		Sets the direction of the light.
		*****************************************************************************/
		void setDirection(const PVRTVec4 v4Direction);

		/*!***************************************************************************
		@Function			getDirectionPVRTVec4
		@Return				direction of the light
		@Description		gets the direction of the light.
		*****************************************************************************/
		PVRTVec4 getDirectionPVRTVec4(){return m_v4Direction;}

		/*!***************************************************************************
		@Function			getDirectionPVRTVec3
		@Return				direction of the light
		@Description		gets the direction of the light.
		*****************************************************************************/
		PVRTVec3 getDirectionPVRTVec3(){return PVRTVec3(m_v4Direction.x,m_v4Direction.y,m_v4Direction.z);}

		/*!***************************************************************************
		@Function			shineLight
		@Input				u32Index - index of the light to shine
		@Description		sets up a hardware light (not functional in OpenGL)
		*****************************************************************************/
		void shineLight(unsigned int u32Index);

	protected:
		PVRTVec4 m_v4Direction;		/*! direction of this light */
	};

}

#endif // LIGHT_H

/******************************************************************************
End of file (Light.h)
******************************************************************************/
