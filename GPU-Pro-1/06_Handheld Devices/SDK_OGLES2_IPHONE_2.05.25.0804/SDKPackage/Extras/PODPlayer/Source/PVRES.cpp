/******************************************************************************

 @File         PVRES.cpp

 @Title        Simple parser for PVRES files

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Shows how to use the pfx format

******************************************************************************/
#include "PVRES.h"

using namespace pvrengine;


const CPVRTString strUNSET = ":UNSET:";

/******************************************************************************/

PVRES::PVRES()
{
	Init();
}

/******************************************************************************/

void PVRES::Init()
{
	m_bFullScreen = false;
	m_bShowFPS = true;
	m_strPODFileName=strUNSET;
	m_strPFXPath=strUNSET;
	m_strTexturePath=strUNSET;
	m_strTitle=strUNSET;
	m_strScriptFileName=strUNSET;
	m_fStartFrame=0.0f;
	m_fAnimationSpeed=0.0f;
	m_bVertSync = true;
	m_bLogToFile = true;
	m_bPowerSaving = true;
	m_i32FSAA = 0;
	m_fQuitAfterTime = -1.0f;
	m_i32QuitAfterFrame = -1;
	m_i32Height = 0;
	m_i32Width = 0;
	m_i32PosX = 0;
	m_i32PosY = 0;
	m_i32DrawMode = Mesh::eNormal;
}

/******************************************************************************/

PVRES::~PVRES()
{
}

/******************************************************************************/

void PVRES::setPODFileName(const CPVRTString& strPODFileName)
{
	if(m_strScriptFileName==strUNSET)
		m_strPODFileName = strPODFileName;
	else
		m_strPODFileName = PVRTStringGetContainingDirectoryPath(m_strScriptFileName)+"/"+strPODFileName;
	// check if texture or pfx paths have been set separately
	// otherwise use pod path
	if(m_strPFXPath==strUNSET)
	{	// set pod path
		m_strPFXPath = PVRTStringGetContainingDirectoryPath(m_strPODFileName);
	}
	if(m_strTexturePath==strUNSET)
	{	// set pod path
		m_strTexturePath = PVRTStringGetContainingDirectoryPath(m_strPODFileName);
	}
	if(m_strTitle==strUNSET)
	{	// set pod path
		m_strTitle = PVRTStringGetFileName(m_strPODFileName);
	}
}

/******************************************************************************
End of file (PVRES.cpp)
******************************************************************************/
