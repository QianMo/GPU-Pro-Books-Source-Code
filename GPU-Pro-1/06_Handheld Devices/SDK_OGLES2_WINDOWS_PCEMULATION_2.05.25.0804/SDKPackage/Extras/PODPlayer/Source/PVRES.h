/******************************************************************************

 @File         PVRES.h

 @Title        A simple script class for use with PVREngine

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A class for holding the information from a .pvres script file in a
               more useful form for the PVREngine.

******************************************************************************/
#ifndef PVRES_H
#define PVRES_H

/******************************************************************************
Includes
******************************************************************************/
#include "PVREngine.h"


/*!****************************************************************************
Class
******************************************************************************/
/*!***************************************************************************
* @Class PVRES
* @Brief 	A simple script class for use with PVREngine.
* @Description 	A simple script class for use with PVREngine.
*****************************************************************************/
class PVRES
{
public:
	/*!***************************************************************************
	@Function			PVRES
	@Description		blank constructor.
	*****************************************************************************/
	PVRES();
	/*!***************************************************************************
	@Function			~PVRES
	@Description		destructor.
	*****************************************************************************/
	~PVRES();

	/*!***************************************************************************
	@Function			setFullScreen
	@Input				bFullScreen	should the engine be in full screen mode
	@Description		sets record of full screen vs windowed mode.
	*****************************************************************************/
	void setFullScreen(bool bFullScreen)
	{m_bFullScreen = bFullScreen;}
	/*!***************************************************************************
	@Function			getFullScreen
	@Return				state of full screen mode flag
	@Description		accessor for fullscreen value of PVRES
	*****************************************************************************/
	bool getFullScreen() const
	{return m_bFullScreen;}

	/*!***************************************************************************
	@Function			setScriptFileName
	@Input				strScriptFileName the original script file path
	@Description		sets the path to the original script.
	*****************************************************************************/
	void setScriptFileName(const CPVRTString& strScriptFileName)
	{m_strScriptFileName = strScriptFileName;}

	/*!***************************************************************************
	@Function			getScriptFileName
	@Return				the original script file path
	@Description		gets the path to the original script.
	*****************************************************************************/
	CPVRTString getScriptFileName() const
	{return m_strScriptFileName;}

	/*!***************************************************************************
	@Function			setPODFileName
	@Input				strPODFileName the POD file path
	@Description		sets the path to the POD file specified in the PVRES.
	*****************************************************************************/
	void setPODFileName(const CPVRTString& strPODFileName);

	/*!***************************************************************************
	@Function			getPODFileName
	@Return				the POD file path
	@Description		gets the path to the POD file specified in the PVRES.
	*****************************************************************************/
	CPVRTString getPODFileName() const
	{return m_strPODFileName;}

	/*!***************************************************************************
	@Function			setPFXPath
	@Input				strPFXPath the PFX files path
	@Description		sets the path to the PFX files specified in the PVRES.
	*****************************************************************************/
	void setPFXPath(const CPVRTString& strPFXPath)
	{m_strPFXPath = strPFXPath;}

	/*!***************************************************************************
	@Function			getPFXPath
	@Return				the PFX files path
	@Description		gets the path to the PFX files specified in the PVRES.
	*****************************************************************************/
	CPVRTString getPFXPath() const
	{return m_strPFXPath;}

	/*!***************************************************************************
	@Function			setTexturePath
	@Input				strTexturePath the texture files path
	@Description		sets the path to the texture files specified in the PVRES.
	*****************************************************************************/
	void setTexturePath(const CPVRTString& strTexturePath)
	{m_strTexturePath = strTexturePath;}

	/*!***************************************************************************
	@Function			getTexturePath
	@Return				the texture files path
	@Description		gets the path to the texture files specified in the PVRES.
	*****************************************************************************/
	CPVRTString getTexturePath() const
	{return m_strTexturePath;}

	/*!***************************************************************************
	@Function			setTitle
	@Input				strTitle the title of this POD/demo
	@Description		sets the title of the POD/demo specified in the PVRES.
	*****************************************************************************/
	void setTitle(const CPVRTString& strTitle)
	{m_strTitle = strTitle;}

	/*!***************************************************************************
	@Function			getTitle
	@Return				the title of this POD/demo
	@Description		gets the title of the POD/demo specified in the PVRES.
	*****************************************************************************/
	CPVRTString getTitle() const
	{return m_strTitle;}

	/*!***************************************************************************
	@Function			setStartFrame
	@Input				fStartFrame the start frame of the scene
	@Description		sets the start frame from the POD specified in the PVRES.
	*****************************************************************************/
	void setStartFrame(const float fStartFrame)
	{m_fStartFrame = fStartFrame;}

	/*!***************************************************************************
	@Function			getStartFrame
	@Return				the start frame of the scene
	@Description		gets the start frame from the POD specified in the PVRES.
	*****************************************************************************/
	float getStartFrame() const
	{return m_fStartFrame;}

	/*!***************************************************************************
	@Function			setShowFPS
	@Input				bShowFPS show/hide onscreen FPS
	@Description		sets whether the FPS should be shown onscreen.
	*****************************************************************************/
	void setShowFPS(const bool bShowFPS)
	{m_bShowFPS = bShowFPS;}

	/*!***************************************************************************
	@Function			getShowFPS
	@Return				show/hide onscreen FPS
	@Description		gets whether the FPS should be shown onscreen.
	*****************************************************************************/
	bool getShowFPS() const
	{return m_bShowFPS;}

	/*!***************************************************************************
	@Function			setAnimationSpeed
	@Input				fAnimationSpeed multiplier for animation rate
	@Description		sets the multiplication factor used to advance the
	animation frame.
	*****************************************************************************/
	void setAnimationSpeed(const float fAnimationSpeed)
	{m_fAnimationSpeed = fAnimationSpeed;}

	/*!***************************************************************************
	@Function			getAnimationSpeed
	@Return				multiplier for animation rate
	@Description		gets the multiplication factor used to advance the
	animation frame.
	*****************************************************************************/
	float getAnimationSpeed() const
	{return m_fAnimationSpeed;}

	/*!***************************************************************************
	@Function			setVertSync
	@Input				bVertSync sync with screen refresh
	@Description		sets whether rendering should be synced with the current
	screen's refreshes.
	*****************************************************************************/
	void setVertSync(const bool bVertSync)
	{m_bVertSync = bVertSync;}

	/*!***************************************************************************
	@Function			getVertSync
	@Return				sync with screen refresh
	@Description		gets whether rendering should be synced with the current
	screen's refreshes.
	*****************************************************************************/
	bool getVertSync() const
	{return m_bVertSync;}

	/*!***************************************************************************
	@Function			setLogToFile
	@Input				bLogToFile write or not write
	@Description		sets whether the output log should be written directly to
	file or not
	*****************************************************************************/
	void setLogToFile(const bool bLogToFile)
	{m_bLogToFile = bLogToFile;}

	/*!***************************************************************************
	@Function			getLogToFile
	@Description		gets whether the output log should be written directly to
	file or not
	@Return				whether the log file should be written to or not
	*****************************************************************************/
	bool getLogToFile() const
	{return m_bLogToFile;}

	/*!***************************************************************************
	@Function			setPowerSaving
	@Input				bPowerSaving power save or not
	@Description		sets whether power saving mode is required or not
	*****************************************************************************/
	void setPowerSaving(const bool bPowerSaving)
	{m_bPowerSaving = bPowerSaving;}

	/*!***************************************************************************
	@Function			getPowerSaving
	@Description		sets whether power saving mode is required or not
	@Return				whether power saving is wanted or not
	*****************************************************************************/
	bool getPowerSaving() const
	{return m_bPowerSaving;}

	/*!***************************************************************************
	@Function			setFSAA
	@Input				i32FSAA 0 off; 1 2x ; 2 4x
	@Description		sets what level of full screen anti-aliasing to use
	*****************************************************************************/
	void setFSAA(const int i32FSAA)
	{m_i32FSAA = i32FSAA;}

	/*!***************************************************************************
	@Function			getFSAA
	@Description		gets what level of full screen anti-aliasing to use
	@Return				FSAA mode: 0 off; 1 2x ; 2 4x
	*****************************************************************************/
	int getFSAA() const
	{return m_i32FSAA;}

	/*!***************************************************************************
	@Function			setWidth
	@Input				i32Width desired width
	@Description		sets the width in pixels of the view
	*****************************************************************************/
	void setWidth(const int i32Width)
	{m_i32Width = i32Width;}

	/*!***************************************************************************
	@Function			getWidth
	@Description		gets the width in pixels of the view
	@Return				width of view in pixels
	*****************************************************************************/
	int getWidth() const
	{return m_i32Width;}

	/*!***************************************************************************
	@Function			setHeight
	@Input				i32Height desired height
	@Description		sets the height in pixels of the view
	*****************************************************************************/
	void setHeight(const int i32Height)
	{m_i32Height = i32Height;}

	/*!***************************************************************************
	@Function			getHeight
	@Description		gets the height in pixels of the view
	@Return				height of view in pixels
	*****************************************************************************/
	int getHeight() const
	{return m_i32Height;}

	/*!***************************************************************************
	@Function			setPosX
	@Input				i32PosX desired horizontal position of the window
	@Description		sets the desired horizontal position of the window
	*****************************************************************************/
	void setPosX(const int i32PosX)
	{m_i32PosX = i32PosX;}

	/*!***************************************************************************
	@Function			getPosX
	@Description		gets the desired horizontal position of the window
	@Return				desired horizontal position of the window
	*****************************************************************************/
	int getPosX() const
	{return m_i32PosX;}

	/*!***************************************************************************
	@Function			setPosY
	@Input				i32PosY desired vertical position of the window
	@Description		sets the desired vertical position of the window
	*****************************************************************************/
	void setPosY(const int i32PosY)
	{m_i32PosY = i32PosY;}

	/*!***************************************************************************
	@Function			getPosY
	@Description		gets the desired vertical position of the window
	@Return				desired vertical position of the window
	*****************************************************************************/
	int getPosY() const
	{return m_i32PosY;}

	/*!***************************************************************************
	@Function			setQuitAfterTime
	@Input				i32PosY desired amount of time after which to
	automatically exit
	@Description		sets the desired amount of time after which to
	automatically exit
	*****************************************************************************/
	void setQuitAfterTime(const float fQuitAfterTime)
	{m_fQuitAfterTime = fQuitAfterTime;}

	/*!***************************************************************************
	@Function			getQuitAfterTime
	@Description		gets the desired amount of time after which to
	automatically exit
	@Return				desired amount of time after which to
	automatically exit
	*****************************************************************************/
	float getQuitAfterTime() const
	{return m_fQuitAfterTime;}

	/*!***************************************************************************
	@Function			setQuitAfterFrame
	@Input				i32QuitAfterFrame desired frame after which to automatically exit
	@Description		sets the desired frame after which to automatically exit
	*****************************************************************************/
	void setQuitAfterFrame(const int i32QuitAfterFrame)
	{m_i32QuitAfterFrame = i32QuitAfterFrame;}

	/*!***************************************************************************
	@Function			getQuitAfterFrame
	@Description		gets the desired frame after which to automatically exit
	@Return				desired frame after which to automatically exit
	*****************************************************************************/
	int getQuitAfterFrame() const
	{return m_i32QuitAfterFrame;}

	/*!***************************************************************************
	@Function			setDrawMode
	@Input				i32DrawMode desired drawing mode for rendering
	@Description		sets the desired drawing mode for rendering meshes
	*****************************************************************************/
	void setDrawMode(const int i32DrawMode)
	{
		if(i32DrawMode>0 && i32DrawMode<pvrengine::Mesh::eNumDrawModes)
			m_i32DrawMode = i32DrawMode;
	}

	/*!***************************************************************************
	@Function			getDrawMode
	@Input				i32DrawMode desired drawing mode for rendering
	@Description		gets the desired drawing mode for rendering meshes
	*****************************************************************************/
	int getDrawMode() const
	{return m_i32DrawMode;}

protected:
	// data stored from the script
	bool m_bFullScreen,m_bShowFPS,m_bVertSync,m_bLogToFile,m_bPowerSaving;
	CPVRTString m_strScriptFileName, m_strPODFileName,
		m_strPFXPath, m_strTexturePath, m_strTitle;
	float m_fStartFrame, m_fAnimationSpeed, m_fQuitAfterTime;
	int m_i32FSAA, m_i32Height, m_i32Width, m_i32PosX, m_i32PosY, m_i32QuitAfterFrame, m_i32DrawMode;

	/*!***************************************************************************
	@Function			Init
	@Description		sets some defaut values.
	*****************************************************************************/
	void Init();
};

#endif // PVRES_H

/******************************************************************************
End of file (PVRES.h)
******************************************************************************/
