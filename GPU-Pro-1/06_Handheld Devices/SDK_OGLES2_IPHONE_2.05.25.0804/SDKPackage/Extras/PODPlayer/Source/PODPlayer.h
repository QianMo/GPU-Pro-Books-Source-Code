/******************************************************************************

 @File         PODPlayer.h

 @Title        PODPlayer: Player of PODs

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     OS/API Independent

 @Description  API independent class declaration for PODPlayer application

******************************************************************************/
#ifndef PODPLAYER_H
#define PODPLAYER_H

/*!****************************************************************************

**********************************
*** About the PODPlayer Source ***
**********************************

PODPlayer is an example application for the PVREngine. As such it inherits
directly from the PVREngine class and demonstrates most of the functionality
that is available to applications built upon this framework.

PVREngine is based upon PVRShell, PVRTools and incorporates a lot of new
classes of its own. Like any other PVRShell application it has the usual 5
functions:

bool InitApplication();
bool InitView();
bool QuitApplication();
virtual bool ReleaseView();
virtual bool RenderScene();

In the InitApplication function the initialisation that is requried before a
rendering context is created is performed. This includes:

- Initialising the console log for the engine
- Reading the command line and whatever script file that might have been passed
- Setting up various configurations such as anti-aliasing, full screen that can't
be done after rendering has started

After this, the PODPlayer continues to initialise in the Init() function called
from RenderScene and this provides visual feedback to the user as the POD file
is processed:

- Print3D is initialised to provide rendered text.
- Materials for the meshes are examined.
- Effects and textures are loaded.
- Individual meshes are processed
- Lights are examined
- The application's menu is initialised

From then on the PODPlayer continues to render until the user chooses to quit.

******************************************************************************/

/******************************************************************************
Includes
******************************************************************************/

#include "PVREngine.h"
#include "PVRESParser.h"
#include "PVRES.h"

/*!****************************************************************************
Class implementing the PVRShell functions.
******************************************************************************/
class PODPlayer : public pvrengine::PVREngine
{
public:

protected:
	// Print3D class used to display text
	CPVRTPrint3D	m_Print3D;

	// Console
	pvrengine::ConsoleLog*		m_pConsole;

	// Camera
	pvrengine::SimpleCamera	m_sCamera;

	// Script structure
	PVRES			m_cPVRES;

	// Menu System
	pvrengine::OptionsMenu		*m_pOptionsMenu;

	// 3D POD Scene
	CPVRTModelPOD	m_Scene;

	// Mesh manager for 3D model
	pvrengine::MeshManager		*m_psMeshManager, *m_psTransparentMeshManager;

	// Light Manager
	pvrengine::LightManager	*m_pLightManager;

	// camera number to use from scene
	unsigned int m_u32CurrentCameraNum;

	// string for onscreen display.
	char				m_pszPrint3DString[1024];

	// string for using as title
	CPVRTString		m_strDemoTitle;

	// used to determine the intialisation state
	int				m_i32Initialising;	

	// preferences
	bool			m_bFreeCamera,		// use free camera
					m_bRotate,			// rotate projection?
					m_bOptions,			// in options menu?
					m_bOverlayOptions;	// overlay options on scene?

	pvrengine::UniformHandler*	m_pUniformHandler;
	pvrengine::TimeController* m_pTimeController;

	// does the multi-stage initialising providing feedback to the user
	bool Init(int& i32Initialising);

	// Inherited PVRShell Functions

	/*!****************************************************************************
	@Function		InitApplication
	@Return			bool		true if no error occured
	@Description	Code in InitApplication() will be called by PVRShell once per
	run, before the rendering context is created.
	Used to initialize variables that are not dependent on it
	(e.g. external modules, loading meshes, etc.)
	If the rendering context is lost, InitApplication() will
	not be called again.
	******************************************************************************/
	bool InitApplication();

	/*!****************************************************************************
	@Function		QuitApplication
	@Return			bool		true if no error occured
	@Description	Code in QuitApplication() will be called by PVRShell once per
	run, just before exiting the program.
	If the rendering context is lost, QuitApplication() will
	not be called.
	******************************************************************************/
	bool QuitApplication();

	/*!****************************************************************************
	@Function		ReleaseView
	@Return			bool		true if no error occured
	@Description	Code in ReleaseView() will be called by PVRShell when the
	application quits or before a change in the rendering context.
	******************************************************************************/
	virtual bool ReleaseView();

	/*!****************************************************************************
	@Function		RenderScene
	@Return			bool		true if no error occured
	@Description	Main rendering loop function of the program. The shell will
	call this function every frame.
	PVRShell will also manage important OS events.
	Will also manage relevent OS events. The user has access to
	these events through an abstraction layer provided by PVRShell.
	******************************************************************************/
	virtual bool RenderScene();

/*!****************************************************************************
@Function		doInput
@Description	Deals with keyboard input from the user
******************************************************************************/
	void doInput();

/*!****************************************************************************
@Function		doFPS
@Description	advances the frame and calcs the frames per sec
******************************************************************************/
	void doFPS();

/*!****************************************************************************
@Function		doCamera
@Description	sets up the view
******************************************************************************/
	void doCamera();

/*!****************************************************************************
@Function		getOptions
@Description	extracts the option choices made by the user from the OptionsMenu
******************************************************************************/
	void getOptions();

/*!****************************************************************************
@Function		do3DScene
@Description	does the actual 3D rendering
******************************************************************************/
	void do3DScene();
};

/*!* Options Menu enums */
enum EOptions
{
	eOptions_OverlayOptions=0,
	eOptions_Pause,
	eOptions_DrawMode,
	eOptions_PODCamera,
	eOptions_FreeCamera,
	eOptions_Invert,
	eOptions_MoveSpeed,
	eOptions_RotateSpeed,
	eOptions_FPS,
	eOptions_Direction,
	eOptions_AnimationSpeed,
	eOptions_NumOptions,
};


/*!* Custom options for Optinos Menu */
static CPVRTString strForwardBackward[] =
{
	"Backwards","Forwards"
};

/*!* Gives a black menu background */
static const unsigned int c_u32MenuBackgroundColour = 0x00000000;

#endif // PODPLAYER_H

/******************************************************************************
End of file (PODPlayer.h)
******************************************************************************/
