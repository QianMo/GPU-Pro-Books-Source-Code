/******************************************************************************

 @File         Option.h

 @Title        Option

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  A class to represent a single variable option in the options menu

******************************************************************************/
#ifndef OPTION_H
#define OPTION_H

/******************************************************************************
Includes
******************************************************************************/
#include "../PVRTools.h"
#include "dynamicArray.h"

namespace pvrengine
{
	/*!***************************************************************************
	** Forward Declaration of OptionsMenu class
	*****************************************************************************/
	class OptionsMenu;

	/*!***************************************************************************
	** Option superclass
	*****************************************************************************/
	/*!***************************************************************************
	* @Class Option
	* @Brief A class to represent a single variable option in the options menu.
	* @Description A class to represent a single variable option in the options menu.
	*****************************************************************************/
	class Option
	{
	public:
	/*!***************************************************************************
	@Function			Option
	@Description		blank constructor.
	*****************************************************************************/
		Option(){}

	/*!***************************************************************************
	@Function			Option
	@Input				strOption	 name of option
	@Description		Base constructor just taking name of option.
	*****************************************************************************/
		Option(const CPVRTString& strOption):m_strOption(strOption){}

	/*!***************************************************************************
	@Function			~Option
	@Description		destructor.
	*****************************************************************************/
		virtual ~Option(){}

	/*!***************************************************************************
	@Function			nextValue
	@Description		sets the next value as current
	*****************************************************************************/
		virtual void nextValue(){}

	/*!***************************************************************************
	@Function			prevValue
	@Description		sets the previous value as current
	*****************************************************************************/
		virtual void prevValue(){}

	/*!***************************************************************************
	@Function			getDisplayValue
	@Description		retrieves a displayable version of the current value of
	this option
	*****************************************************************************/
		virtual CPVRTString	getDisplayValue(){ return CPVRTString("ERROR");}

		friend class OptionsMenu;
	protected:
		CPVRTString			m_strOption;			// name of option
	};

	/*!***************************************************************************
	* @Class OptionEnum
	* @Brief An option class for discrete values usually with string names.
	* @Description An option class for discrete values usually with string names.
	*****************************************************************************/
	class OptionEnum : public Option
	{
	public:
	/*!***************************************************************************
	@Function			OptionEnum
	@Input				strOption		name of option
	@Input				pstrValues		the string array of value names
	@Input				u32NumValues	number of values for this option
	@Input				u32CurrentValue	intial value for option
	@Description		Constructor.
	*****************************************************************************/
		OptionEnum(const CPVRTString& strOption,
			const CPVRTString *pstrValues,
			const int u32NumValues,
			int u32CurrentValue=0);
	/*!***************************************************************************
	@Function			nextValue
	@Description		sets the next value as current
	*****************************************************************************/
		void nextValue();
	/*!***************************************************************************
	@Function			prevValue
	@Description		sets the previous value as current
	*****************************************************************************/
		void prevValue();
	/*!***************************************************************************
	@Function			getValue
	@Description		gets the current value
	*****************************************************************************/
		int getValue(){ return m_i32CurrentValue;}
	/*!***************************************************************************
	@Function			getDisplayValue
	@Description		retrieves a displayable version of the current value of
	this option
	*****************************************************************************/
		CPVRTString	getDisplayValue();
	private:
		const CPVRTString			*m_pstrValues;	// array of readable values
		const int	m_i32NumValues;			// number of values for this option
		int	m_i32CurrentValue;		// current value of option
	};

	/*!***************************************************************************
	* @Class OptionInt
	* @Brief An option class for continuous integer values.
	* @Description An option class for continuous integer values.
	*****************************************************************************/
	class OptionInt : public Option
	{
	public:
	/*!***************************************************************************
	@Function			OptionInt
	@Input				strOption		name of option
	@Input				i32Low			Lowest desired value for integer
	@Input				i32High			Highest desired value for integer
	@Input				i32Add			step interval for values
	@Input				i32CurrentValue	intial value for option
	@Description		Constructor.
	*****************************************************************************/
		OptionInt(const CPVRTString& strOption,
			int i32Low, int i32High, int i32Add = 1,
			int i32CurrentValue=0);
	/*!***************************************************************************
	@Function			nextValue
	@Description		sets the next value as current
	*****************************************************************************/
		void nextValue();
	/*!***************************************************************************
	@Function			prevValue
	@Description		sets the previous value as current
	*****************************************************************************/
		void prevValue();
	/*!***************************************************************************
	@Function			getValue
	@Description		gets the current value
	*****************************************************************************/
		int getValue(){ return m_i32CurrentValue;}
	/*!***************************************************************************
	@Function			getDisplayValue
	@Description		retrieves a displayable version of the current value of
	this option
	*****************************************************************************/
		CPVRTString	getDisplayValue();
	private:
		int	m_i32Low,m_i32High,m_i32Add;	// interval and step for integer values
		int	m_i32CurrentValue;				// current value of option
	};

	/*!***************************************************************************
	* @Class OptionFloat
	* @Brief An option class for continuous floating point values.
	* @Description An option class for continuous floating point values.
	*****************************************************************************/
	class OptionFloat : public Option
	{
	public:
	/*!***************************************************************************
	@Function			OptionFloat
	@Input				strOption		name of option
	@Input				fLow			Lowest desired value for float
	@Input				fHigh			Highest desired value for float
	@Input				fAdd			step interval for values
	@Input				fCurrentValue	initial value for option
	@Description		Constructor.
	*****************************************************************************/
		OptionFloat(const CPVRTString& strOption,
			float fLow, float fHigh, float fAdd,
			float fCurrentValue=0.0f);
	/*!***************************************************************************
	@Function			nextValue
	@Description		sets the next value as current
	*****************************************************************************/
		void nextValue();
	/*!***************************************************************************
	@Function			prevValue
	@Description		sets the previous value as current
	*****************************************************************************/
		void prevValue();
	/*!***************************************************************************
	@Function			getValue
	@Description		gets the current value
	*****************************************************************************/
		float getValue(){ return m_fCurrentValue;}
	/*!***************************************************************************
	@Function			getDisplayValue
	@Description		retrieves a displayable version of the current value of
	this option
	*****************************************************************************/
		CPVRTString	getDisplayValue();
	private:
		float	m_fLow,m_fHigh,m_fAdd;	// interval and step for float values
		float	m_fCurrentValue;		// current value of option
	};



	/*!***************************************************************************
	** commonly required string options
	*****************************************************************************/
	static CPVRTString strOnOff[] =
	{
		"Off","On"
	};

	static CPVRTString strTrueFalse[] =
	{
		"True","False"
	};
}

#endif // OPTION_H

/******************************************************************************
End of file (Option.h)
******************************************************************************/
