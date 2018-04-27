/******************************************************************************

 @File         PVRShellImpl.h

 @Title        PVRShellImpl

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Makes programming for 3D APIs easier by wrapping surface
               initialization, Texture allocation and other functions for use by a demo.

******************************************************************************/
#ifndef __PVRSHELLIMPL_H_
#define __PVRSHELLIMPL_H_

/*****************************************************************************
** Build options
*****************************************************************************/


/*****************************************************************************
** Macros
*****************************************************************************/
#define FREE(X) { if(X) { free(X); (X)=0; } }

#ifndef _ASSERT
#define _ASSERT(X) /**/
#endif

/*****************************************************************************
** Defines
*****************************************************************************/
#define STR_WNDTITLE (" - Build ")

/*!***************************************************************************
 @Struct PVRShellData
 @Brief Holds PVRShell internal data.
*****************************************************************************/
struct PVRShellData
{
	// Shell Interface Data
	char		*pszAppName;
	char		*pszExitMessage;
	int			nShellDimX;
	int			nShellDimY;
	int			nShellPosX;
	int			nShellPosY;
	bool		bFullScreen;
	bool		bLandscape;
	bool		bNeedPbuffer;
	bool		bNeedZbuffer;
	bool		bNeedStencilBuffer;
	bool		bNeedPixmap;
	bool		bNeedPixmapDisableCopy;
	bool		bLockableBackBuffer;
	bool		bSoftwareRender;
	bool		bNeedOpenVG;
	bool		bNeedAlphaFormatPre;
	bool		bUsingPowerSaving;
	bool		bOutputInfo;
	bool		bNoShellSwapBuffer;
	int			nSwapInterval;
	int			nInitRepeats;
	int			nDieAfterFrames;
	float		fDieAfterTime;
	int			nFSAAMode;
	int			nColorBPP;
	int			nCaptureFrameStart;
	int			nCaptureFrameStop;
	int			nPriority;

	// Internal Data
	bool		bShellPosWasDefault;
	int			nShellCurFrameNum;
#ifdef PVRSHELL_FPS_OUTPUT
	bool		bOutputFPS;
#endif
};

/*!***************************************************************************
 @Class PVRShellCommandLine
 @Brief Command-line interpreter
*****************************************************************************/
class PVRShellCommandLine
{
public:
	char		*m_psOrig, *m_psSplit;
	SCmdLineOpt	*m_pOpt;
	int			m_nOptLen, m_nOptMax;

public:
	/*!***********************************************************************
	@Function			PVRShellCommandLine
	@Description		Constructor
	*************************************************************************/
	PVRShellCommandLine();

	/*!***********************************************************************
	@Function			PVRShellCommandLine
	@Description		Destructor
	*************************************************************************/
	~PVRShellCommandLine();

	/*!***********************************************************************
	@Function		Parse
	@Input			pStr Input string
	@Description	Parse pStr for command-line options and store them in m_pOpt
	*************************************************************************/
	void Parse(const char *pStr);

	/*!***********************************************************************
	@Function		Apply
	@Input			shell
	@Description	Apply the command-line options to shell
	*************************************************************************/
	void Apply(PVRShell &shell);
};

/*!****************************************************************************
 * @Enum  EPVRShellState
 * @Brief Current Shell state
*****************************************************************************/
enum EPVRShellState {
	ePVRShellInitApp,
	ePVRShellInitInstance,
	ePVRShellRender,
	ePVRShellReleaseView,
	ePVRShellReleaseAPI,
	ePVRShellReleaseOS,
	ePVRShellQuitApp,
	ePVRShellExit
};

/*!***************************************************************************
 * @Class  PVRShellInit
 * @Brief  The PVRShell initialisation class
 ****************************************************************************/
class PVRShellInit : public PVRShellInitAPI, public PVRShellInitOS
{
public:
	friend class PVRShell;
	friend class PVRShellInitOS;
	friend class PVRShellInitAPI;

	PVRShell			*m_pShell;		/*!< Our PVRShell class */
	PVRShellCommandLine	m_CommandLine;	/*!< Our Commad-line class */

	bool		gShellDone;				/*!< Indicates that the application has finished */
	EPVRShellState	m_eState;			/*!< Current PVRShell state */

	// Key handling
	PVRShellKeyName	nLastKeyPressed;	/*!< Holds the last key pressed */
	PVRShellKeyName m_eKeyMapLEFT;		/*!< Holds the value to be returned when PVRShellKeyNameLEFT is requested */
	PVRShellKeyName m_eKeyMapUP;		/*!< Holds the value to be returned when PVRShellKeyNameUP is requested */
	PVRShellKeyName m_eKeyMapRIGHT;		/*!< Holds the value to be returned when PVRShellKeyNameRIGHT is requested */
	PVRShellKeyName m_eKeyMapDOWN;		/*!< Holds the value to be returned when PVRShellKeyNameDOWN is requested */

	// Read and Write path
	char	*m_pReadPath;				/*!<Holds the path where the application will read the data from */
	char	*m_pWritePath;				/*!<Holds the path where the application will write the data to */

#ifdef PVRSHELL_FPS_OUTPUT
	// Frames per second (FPS)
	int		m_i32FpsFrameCnt, m_i32FpsTimePrev;
#endif

public:

    /*!***********************************************************************
	 @Function		PVRShellInit
	 @description	Constructor
	*************************************************************************/
	PVRShellInit();

	/*!***********************************************************************
	 @Function		~PVRShellInit
	 @description	Destructor
	*************************************************************************/
	~PVRShellInit();

	/*!***********************************************************************
	 @Function		Init
	 @Input			Shell
	 @description	PVRShell Initialisation.
	*************************************************************************/
	void Init(PVRShell &Shell);

	/*!***********************************************************************
	 @Function		CommandLine
	 @Input			str A string containing the command-line
	 @description	Receives the command-line from the application.
	*************************************************************************/
	void CommandLine(char *str);

	/*!***********************************************************************
	@Function		CommandLine
	@Input			argc Number of strings in argv
	@Input			argv An array of strings
	@description	Receives the command-line from the application.
	*************************************************************************/
	void CommandLine(int argc, char **argv);

	/*!***********************************************************************
	 @Function		DoIsKeyPressed
	 @Input			key The key we're querying for
	 @description	Return 'true' if the specific key has been pressed.
	*************************************************************************/
	bool DoIsKeyPressed(const PVRShellKeyName key);

	/*!***********************************************************************
	 @Function		KeyPressed
	 @Input			key The key that has been pressed
	 @description	Used by the OS-specific code to tell the Shell that a key has been pressed.
	*************************************************************************/
	void KeyPressed(PVRShellKeyName key);

	/*!***********************************************************************
	 @Function		GetReadPath
	 @Returns		A path the application is capable of reading from
	 @description	Used by the OS-specific code to tell the Shell where to read external files from
	*************************************************************************/
	const char	*GetReadPath() const;

	/*!***********************************************************************
	 @Function		GetWritePath
	 @Returns		A path the applications is capable of writing to
	 @description	Used by the OS-specific code to tell the Shell where to write to
	*************************************************************************/
	const char	*GetWritePath() const;

	/*!******************************************************************************
	 @Function	  SetAppName
	 @Input		  str The application name
	 @Description Sets the default app name (to be displayed by the OS)
	*******************************************************************************/
	void SetAppName(const char * const str);

	/*!***********************************************************************
	 @Function		SetReadPath
	 @Input			str The read path
	 @description	Set the path to where the application expects to read from.
	*************************************************************************/
	void SetReadPath(const char * const str);

	/*!***********************************************************************
	 @Function		SetWritePath
	 @Input			str The write path
	 @description	Set the path to where the application expects to write to.
	*************************************************************************/
	void SetWritePath(const char * const str);

	/*!***********************************************************************
	 @Function		Run
	 @description	Called from the OS-specific code to perform the render.
					When this fucntion fails the application will quit.
	*************************************************************************/
	bool Run();

	/*!***********************************************************************
	 @Function		OutputInfo
	 @description	When prefOutputInfo is set to true this function outputs
					various pieces of non-API dependent information via
					PVRShellOutputDebug.
	*************************************************************************/
	void OutputInfo();

	/*!***********************************************************************
	 @Function		OutputAPIInfo
	 @description	When prefOutputInfo is set to true this function outputs
					various pieces of API dependent information via
					PVRShellOutputDebug.
	*************************************************************************/
	void OutputAPIInfo();

#ifdef PVRSHELL_FPS_OUTPUT
	/*!****************************************************************************
	@Function   	FpsUpdate
	@Description    Calculates a value for frames-per-second (FPS).
	*****************************************************************************/
	void FpsUpdate();
#endif

	/*
		OS functionality
	*/

	/*!***********************************************************************
	 @Function		OsInit
	 @description	Initialisation for OS-specific code.
	*************************************************************************/
	void		OsInit();

	/*!***********************************************************************
	 @Function		OsInitOS
	 @description	Saves instance handle and creates main window
					In this function, we save the instance handle in a global variable and
					create and display the main program window.
	*************************************************************************/
	bool		OsInitOS();

	/*!***********************************************************************
	 @Function		OsReleaseOS
	 @description	Destroys main window
	*************************************************************************/
	void		OsReleaseOS();

	/*!***********************************************************************
	@Function		OsExit
	@description	Destroys main window
	*************************************************************************/
	void		OsExit();

	/*!***********************************************************************
	 @Function		OsDoInitAPI
	 @description	Perform API initialization and bring up window / fullscreen
	*************************************************************************/
	bool		OsDoInitAPI();

	/*!***********************************************************************
	 @Function		OsDoReleaseAPI
	 @description	Clean up after we're done
	*************************************************************************/
	void		OsDoReleaseAPI();

	/*!***********************************************************************
	 @Function		OsRenderComplete
	 @description	Main message loop / render loop
	*************************************************************************/
	void		OsRenderComplete();

	/*!***********************************************************************
	 @Function		OsPixmapCopy
	 @description	When using pixmaps, copy the render to the display
	*************************************************************************/
	bool		OsPixmapCopy();

	/*!***********************************************************************
	 @Function		OsGetNativeDisplayType
	 @description	Called from InitAPI() to get the NativeDisplayType
	*************************************************************************/
	void		*OsGetNativeDisplayType();

	/*!***********************************************************************
	 @Function		OsGetNativePixmapType
	 @description	Called from InitAPI() to get the NativePixmapType
	*************************************************************************/
	void		*OsGetNativePixmapType();

	/*!***********************************************************************
	 @Function		OsGetNativeWindowType
	 @description	Called from InitAPI() to get the NativeWindowType
	*************************************************************************/
	void		*OsGetNativeWindowType();

	/*!***********************************************************************
	 @Function		OsGet
	 @Input			prefName	Name of value to get
	 @Modified		pn A pointer set to the value asked for
	 @Returns		true on success
	 @Description	Retrieves OS-specific data
	*************************************************************************/
	bool		OsGet(const prefNameIntEnum prefName, int *pn);

	/*!***********************************************************************
	 @Function		OsGet
	 @Input			prefName	Name of value to get
	 @Modified		pp A pointer set to the value asked for
	 @Returns		true on success
	 @Description	Retrieves OS-specific data
	*************************************************************************/
	bool		OsGet(const prefNamePtrEnum prefName, void **pp);

	/*!***********************************************************************
	 @Function		OsDisplayDebugString
	 @Input			str The debug string to display
	 @Description	Prints a debug string
	*************************************************************************/
	void OsDisplayDebugString(char const * const str);

	/*!***********************************************************************
	 @Function		OsGetTime
	 @Description	Gets the time in milliseconds
	*************************************************************************/
	unsigned long OsGetTime();

	/*
		API functionality
	*/
	/*!***********************************************************************
	 @Function		ApiInitAPI
	 @description	Initialisation for API-specific code.
	*************************************************************************/
	bool ApiInitAPI();

	/*!***********************************************************************
	 @Function		ApiReleaseAPI
	 @description	Releases all resources allocated by the API.
	*************************************************************************/
	void ApiReleaseAPI();

	/*!***********************************************************************
	 @Function		ApiScreenCaptureBuffer
	 @Input			Width Width of the region to capture
	 @Input			Height Height of the region to capture
	 @Modified		pBuf A buffer to put the screen capture into
	 @description	API-specific function to store the current content of the
					FrameBuffer into the memory allocated by the user.
	*************************************************************************/
	bool ApiScreenCaptureBuffer(int Width,int Height,unsigned char *pBuf);

	/*!***********************************************************************
	 @Function		ApiRenderComplete
	 @description	Perform API operations required after a frame has finished (e.g., flipping).
	*************************************************************************/
	void ApiRenderComplete();

	/*!***********************************************************************
	 @Function		ApiGet
	 @Input			prefName	Name of value to get
	 @Modified		pn A pointer set to the value asked for
	 @description	Get parameters which are specific to the API.
	*************************************************************************/
	bool ApiGet(const prefNameIntEnum prefName, int *pn);

	/*!***********************************************************************
	 @Function		ApiGet
	 @Input			prefName	Name of value to get
	 @Modified		pp A pointer set to the value asked for
	 @description	Get parameters which are specific to the API.
	*************************************************************************/
	bool ApiGet(const prefNamePtrEnum prefName, void **pp);

	/*!***********************************************************************
	 @Function		ApiActivatePreferences
	 @description	Run specific API code to perform the operations requested in preferences.
	*************************************************************************/
	void ApiActivatePreferences();
};

#endif /* __PVRSHELLIMPL_H_ */

/*****************************************************************************
 End of file (PVRShellImpl.h)
*****************************************************************************/
