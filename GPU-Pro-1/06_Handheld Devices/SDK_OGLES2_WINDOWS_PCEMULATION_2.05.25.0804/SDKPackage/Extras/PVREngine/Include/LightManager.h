/******************************************************************************

 @File         LightManager.h

 @Title        A simple light manager for use with PVREngine

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A class for holding information about lights

******************************************************************************/
#ifndef LIGHTMANAGER_H
#define LIGHTMANAGER_H

/******************************************************************************
Includes
******************************************************************************/

#include "../PVRTools.h"
#include "../PVRTSingleton.h"
#include "dynamicArray.h"

namespace pvrengine
{

	class Light;
	/*!***************************************************************************
	* @Class LightManager
	* @Brief A class for holding information about lights
	* @Description A class for holding information about lights
	*****************************************************************************/
	class LightManager : public CPVRTSingleton<LightManager>
	{

	public:
		/*!***************************************************************************
		@Function			LightManager
		@Description		blank constructor.
		*****************************************************************************/
		LightManager();

		/*!***************************************************************************
		@Function			~LightManager
		@Description		destructor.
		*****************************************************************************/
		~LightManager();

		/*!***************************************************************************
		@Function			LightManager
		@Input				i32TotalLights
		@Description		constructor taking initial number of lights.
		*****************************************************************************/
		LightManager(int i32TotalLights);

		/*!***************************************************************************
		@Function			addLight
		@Input				sScene	a POD scene
		@Input				i32Index	number of light to take from scene
		@Return			handle to this light
		@Description		allows the extraction of lights directly from POD files.
		*****************************************************************************/
		unsigned int	addLight(const CPVRTModelPOD& sScene, const unsigned int i32Index );

		/*!***************************************************************************
		@Function			addLight
		@Input				vec3Direction	light direction
		@Input				vec3Colour	light colour
		@Return			handle to this light
		@Description		allows the extraction of lights directly from POD files.
		*****************************************************************************/
		unsigned int	addDirectionalLight(const PVRTVec3& vec3Direction,
			const PVRTVec3& vec3Colour);

		/*!***************************************************************************
		@Function			getLight
		@Input				u32Light	index of light
		@Return			bool success
		@Description		sets the path to the output file.
		*****************************************************************************/
		Light*	getLight(unsigned int u32Light) const
		{
			return m_daLights[u32Light];
		}

		/*!***************************************************************************
		@Function			getLights
		@Return			the dynamic array of all the lights in the manager
		@Description		Direct access to light repository.
		*****************************************************************************/
		dynamicArray<Light*>*	getLights(){ return &m_daLights;}

		/*!***************************************************************************
		@Function			shineLights
		@Description		very unsubtle function (atm) to initialise the lights
		in a scene
		*****************************************************************************/
		void	shineLights();


	private:
		dynamicArray<Light*> m_daLights;	/*! the light store */

		/*!***************************************************************************
		@Function			sort
		@Description		shared initialisation function for constructors
		*****************************************************************************/
		bool Init();	
	};

}
#endif // LIGHTMANAGER_H

/******************************************************************************
End of file (LightManager.h)
******************************************************************************/
