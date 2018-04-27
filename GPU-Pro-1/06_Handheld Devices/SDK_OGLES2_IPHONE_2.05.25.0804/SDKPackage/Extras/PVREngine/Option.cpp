/******************************************************************************

 @File         Option.cpp

 @Title        Option

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Option class for use with OptionsMenu

******************************************************************************/
#include "Option.h"

namespace pvrengine
{
	

	/******************************************************************************/

	OptionEnum::OptionEnum(const CPVRTString& strOption,
				const CPVRTString *pstrValues,
				int i32NumValues,
				int i32CurrentValue):
		Option(strOption),
		m_pstrValues(pstrValues),
		m_i32NumValues(i32NumValues),
		m_i32CurrentValue(i32CurrentValue)
	{}

	/******************************************************************************/

	void OptionEnum::nextValue()
	{
		m_i32CurrentValue++;
		if(m_i32CurrentValue>=m_i32NumValues)
			m_i32CurrentValue=0;
	}

	/******************************************************************************/

	void OptionEnum::prevValue()
	{
		m_i32CurrentValue--;
		if(m_i32CurrentValue<0)
			m_i32CurrentValue=m_i32NumValues-1;
	}

	/******************************************************************************/

	CPVRTString OptionEnum::getDisplayValue()
	{
		return m_pstrValues[m_i32CurrentValue];
	}

	/******************************************************************************/

	OptionInt::OptionInt(const CPVRTString& strOption,
				int i32Low, int i32High, int i32Add,
				int i32CurrentValue):
		Option(strOption),
		m_i32Low(i32Low),
		m_i32High(i32High),
		m_i32Add(i32Add),
		m_i32CurrentValue(i32CurrentValue)
	{
	}

	/******************************************************************************/

	void OptionInt::nextValue()
	{
		m_i32CurrentValue+=m_i32Add;
		if(m_i32CurrentValue>m_i32High)
		{
			m_i32CurrentValue = m_i32High;
		}
	}

	/******************************************************************************/

	void OptionInt::prevValue()
	{
		m_i32CurrentValue-=m_i32Add;
		if(m_i32CurrentValue<m_i32Low)
		{
			m_i32CurrentValue = m_i32Low;
		}
	}

	/******************************************************************************/

	CPVRTString	OptionInt::getDisplayValue()
	{
		char pszInt[20];
		sprintf(pszInt,"%d",m_i32CurrentValue);
		return CPVRTString(pszInt);
	}


	/******************************************************************************/

	OptionFloat::OptionFloat(const CPVRTString& strOption,
		float fLow, float fHigh, float fAdd,
		float fCurrentValue):
		Option(strOption),
		m_fLow(fLow),
		m_fHigh(fHigh),
		m_fAdd(fAdd),
		m_fCurrentValue(fCurrentValue)
	{
	}

	/******************************************************************************/

	void OptionFloat::nextValue()
	{
		m_fCurrentValue+=m_fAdd;
		if(m_fCurrentValue>m_fHigh)
		{
			m_fCurrentValue = m_fHigh;
		}
	}

	/******************************************************************************/

	void OptionFloat::prevValue()
	{
		m_fCurrentValue-=m_fAdd;
		if(m_fCurrentValue<m_fLow)
		{
			m_fCurrentValue = m_fLow;
		}
	}

	/******************************************************************************/

	CPVRTString	OptionFloat::getDisplayValue()
	{
		char pszFloat[20];
		sprintf(pszFloat,"%.2f",m_fCurrentValue);
		return CPVRTString(pszFloat);
	}

}

/******************************************************************************
End of file (Option.cpp)
******************************************************************************/
