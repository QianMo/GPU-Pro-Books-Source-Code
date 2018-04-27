/******************************************************************************

 @File         OptionsMenu.h

 @Title        PODPLayer Options

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     OS/API Independent

 @Description  Options menu for PVREngine apps

******************************************************************************/
#ifndef OPTIONSMENU_H
#define OPTIONSMENU_H

/******************************************************************************
Includes
******************************************************************************/

#include "Globals.h"
#include "Option.h"

namespace pvrengine
{
	/*!***************************************************************************
	* @Class OptionsMenu
	* @Brief Options menu for PODPlayer.
	* @Description Options menu for PODPlayer.
	*****************************************************************************/
	class OptionsMenu
	{
	public:
	/*!***************************************************************************
	@Function			OptionsMenu
	@Description		blank constructor.
	*****************************************************************************/
		OptionsMenu(){}

	/*!***************************************************************************
	@Function			~OptionsMenu
	@Description		destructor.
	*****************************************************************************/
		~OptionsMenu();

	/*!***************************************************************************
	@Function			OptionsMenu
	@Input				pPrint3D	pointer to print3d
	@Description		basic constructor taking print3d (required for menu).
	*****************************************************************************/
		OptionsMenu(CPVRTPrint3D* pPrint3D);

	/*!***************************************************************************
	@Function			addOption
	@Input				pOption		a new option to add to the menu
	@Description		functino to add an option to the menu.
	*****************************************************************************/
		void addOption(Option* pOption);

	/*!***************************************************************************
	@Function			render
	@Description		Renders the options menu.
	*****************************************************************************/
		void render();

	/*!***************************************************************************
	@Function			nextValue
	@Description		Requests next value from the current option.
	*****************************************************************************/
		void			nextValue();

	/*!***************************************************************************
	@Function			prevValue
	@Description		Requests next value from the current option.
	*****************************************************************************/
		void			prevValue();

	/*!***************************************************************************
	@Function			nextOption
	@Description		Moves onto next option.
	*****************************************************************************/
		void			nextOption();

	/*!***************************************************************************
	@Function			prevOption
	@Description		Moves onto previous option.
	*****************************************************************************/
		void			prevOption();

	/*!***************************************************************************
	@Function			getValueBool
	@Description		retrieves boolean value from the current option.
	*****************************************************************************/
		bool			getValueBool(unsigned int eOption);

	/*!***************************************************************************
	@Function			getValueEnum
	@Description		retrieves enum value from the current option.
	*****************************************************************************/
		unsigned int	getValueEnum(unsigned int eOption);

	/*!***************************************************************************
	@Function			getValueInt
	@Description		retrieves int value from the current option.
	*****************************************************************************/
		int				getValueInt(unsigned int eOption);

	/*!***************************************************************************
	@Function			getValueFloat
	@Description		retrieves float value from the current option.
	*****************************************************************************/
		float			getValueFloat(unsigned int eOption);

	private:
		CPVRTPrint3D		*m_pPrint3D;		/*! print3d pointer for rendering text */
		dynamicArray<Option*> m_daOptions;		/*! the store of options */
		int m_i32CurrentOption;					/*! the handle of the current option */

		/*! for making the current option pulse */
		VERTTYPE					m_f32BrightnessMod, m_f32BrightnessModDirection;
	};
}

#endif // OPTIONSMENU_H

/******************************************************************************
End of file (OptionsMenu.h)
******************************************************************************/
