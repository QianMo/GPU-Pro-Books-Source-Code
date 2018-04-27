/******************************************************************************

 @File         OptionsMenu.cpp

 @Title        PODPlayer Options Menu

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     OS/API Independent

 @Description  Options menu thing for PODPlayer

******************************************************************************/
#include "OptionsMenu.h"
#include "Option.h"
#include "TimeController.h"

namespace pvrengine
{

	const VERTTYPE cBrightnessRise = f2vt(8.f);
	const VERTTYPE cBrightnessFall = f2vt(-20.f);

	/******************************************************************************/

	OptionsMenu::OptionsMenu(CPVRTPrint3D *pPrint3D)
	{
		m_pPrint3D = pPrint3D;
		m_i32CurrentOption = 0;

		m_f32BrightnessMod=0x99;
		m_f32BrightnessModDirection =cBrightnessRise;
	}

	/******************************************************************************/

	OptionsMenu::~OptionsMenu()
	{
		for(unsigned int i=0;i<m_daOptions.getSize();i++)
		{
			PVRDELETE(m_daOptions[i]);
		}
	}

	/******************************************************************************/

	void OptionsMenu::addOption(Option* pOption)
	{
		m_daOptions.append(pOption);
	}

	/******************************************************************************/

	void OptionsMenu::nextValue()
	{
		m_daOptions[m_i32CurrentOption]->nextValue();
	}

	/******************************************************************************/

	void OptionsMenu::prevValue()
	{
		m_daOptions[m_i32CurrentOption]->prevValue();
	}

	/******************************************************************************/

	void OptionsMenu::nextOption()
	{
		m_i32CurrentOption++;
		if(m_i32CurrentOption>=(int)m_daOptions.getSize())
			m_i32CurrentOption=0;
	}

	/******************************************************************************/

	void OptionsMenu::prevOption()
	{
		m_i32CurrentOption--;
		if(m_i32CurrentOption<0)
			m_i32CurrentOption= m_daOptions.getSize()-1;
	}

	/******************************************************************************/

	bool OptionsMenu::getValueBool(unsigned int eOption)
	{
		return ((OptionEnum*)m_daOptions[eOption])->getValue()!=0;
	}

	/******************************************************************************/

	unsigned int OptionsMenu::getValueEnum(unsigned int eOption)
	{
		return ((OptionEnum*)m_daOptions[eOption])->getValue();
	}

	/******************************************************************************/

	int OptionsMenu::getValueInt(unsigned int eOption)
	{
		return ((OptionInt*)m_daOptions[eOption])->getValue();
	}

	/******************************************************************************/

	float OptionsMenu::getValueFloat(unsigned int eOption)
	{
		return ((OptionFloat*)m_daOptions[eOption])->getValue();
	}


	/******************************************************************************/

	void OptionsMenu::render()
	{

		// do pulse
		m_f32BrightnessMod+=VERTTYPEMUL(m_f32BrightnessModDirection,(float)TimeController::inst().getDeltaTime());
		if(m_f32BrightnessMod>0xff)
		{
			m_f32BrightnessMod = 0xff;
			m_f32BrightnessModDirection=(cBrightnessFall);
		}
		else if (m_f32BrightnessMod<0xbb)
		{
			m_f32BrightnessMod = 0xbb;
			m_f32BrightnessModDirection=(cBrightnessRise);
		}

		int numOptions = (int)m_daOptions.getSize();

		// draw menu
		for(int i=m_i32CurrentOption-8,j=0;j<18;i++,j++)
		{
			while(i<0)
			{
				i+=numOptions;
			}
			while(i>=numOptions)
			{
				i-=numOptions;
			}
			Option* pCurrentOption = m_daOptions[i];
			if(i==m_i32CurrentOption)
			{
				unsigned int i32BrightnessMod = (unsigned int) m_f32BrightnessMod;
				m_pPrint3D->Print3D(20.0f,10.0f+5.0f*j,0.6f,0xff000000
					+(i32BrightnessMod<<16)
					+(i32BrightnessMod<<8)
					+i32BrightnessMod,"%s",
					pCurrentOption->m_strOption.c_str());
				m_pPrint3D->Print3D(60.0f,10.0f+VERTTYPEMUL(5.0f,f2vt(j)),0.6f,0xff000000+(i32BrightnessMod<<8)+i32BrightnessMod,"%s",
					pCurrentOption->getDisplayValue().c_str());
			}
			else
			{
				m_pPrint3D->Print3D(20.0f,f2vt(10.0f)+VERTTYPEMUL(f2vt(5.0f),f2vt(j)),f2vt(0.5f),0x00997777|(0x15*(0x9-PVRTABS(j-8)))<<24,"%s",
					pCurrentOption->m_strOption.c_str());
				m_pPrint3D->Print3D(60.0f,f2vt(10.0f)+VERTTYPEMUL(f2vt(5.0f),f2vt(j)),f2vt(0.5f),0x00009999|(0x15*(0x9-PVRTABS(j-8)))<<24,"%s",
					pCurrentOption->getDisplayValue().c_str());
			}
		}
	}
}

/******************************************************************************
End of file (OptionsMenu.cpp)
******************************************************************************/
