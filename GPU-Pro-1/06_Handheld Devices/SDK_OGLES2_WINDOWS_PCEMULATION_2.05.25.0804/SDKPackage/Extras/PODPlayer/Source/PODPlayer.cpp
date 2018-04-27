/******************************************************************************

 @File         PODPlayer.cpp

 @Title        PODPlayer

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     OS/API Independent

 @Description  Plays POD files using the PVREngine

******************************************************************************/
#include "PODPlayer.h"

using namespace pvrengine;

enum INIT_STAGES{
	eINIT_BASIC = 1,
	eINIT_PRINT3D,
	eINIT_BASICSCENE,
	eINIT_LIGHTS,
	eINIT_MATERIALS,
	eINIT_EFFECTS,
	eINIT_MESHES,
	eINIT_MENU,
	eINIT_FINISH,
}; /*!* Initialisation stages */

/******************************************************************************/

bool PODPlayer::InitApplication()
{
	// set up console/log
	m_pConsole = ConsoleLog::ptr();

	// grab pointers to these handlers and initialise them
	m_pUniformHandler = UniformHandler::ptr();
	m_pTimeController = TimeController::ptr();

	m_psMeshManager = NULL;
	m_psTransparentMeshManager = NULL;
	m_pOptionsMenu = NULL;

	// deal with command line
	int i32NumCLOptions = PVRShellGet(prefCommandLineOptNum);
	SCmdLineOpt* sCLOptions = (SCmdLineOpt*)PVRShellGet(prefCommandLineOpts);
	CPVRTString strFilename;

	PVRESParser cParser;

	bool bFoundFile = false;
	if(sCLOptions)
	{
		for(int i=0;i<i32NumCLOptions;++i)
		{
			if(!sCLOptions[i].pVal)
			{	// could be script or pod
				strFilename=sCLOptions[i].pArg;
				// determine whether a script or a POD or nothing
				CPVRTString strExtension = PVRTStringGetFileExtension(strFilename);
				if(strExtension.compare(".pvres")==0
					|| strExtension.compare(".PVRES")==0)
				{	// script file
					m_pConsole->log("Found script:%s\n", strFilename.c_str());
					cParser.setScriptFileName(strFilename);
					bFoundFile = true;
				}
				else
				{
					if(strExtension.compare(".pod")==0
						|| strExtension.compare(".POD")==0)
					{	// pod file
						m_pConsole->log("Found POD\n");
						cParser.setPODFileName(strFilename);
						bFoundFile = true;
					}
					else
					{
						m_pConsole->log("Unrecognised filetype.\n");
					}
				}
			}
		}
	}
	if(!bFoundFile)
	{	// no command line options so open default pvres
		CPVRTString strDefaultPVRES((char*)(PVRShellGet(prefReadPath)));
		strDefaultPVRES += "Sample.pvres";
		cParser.setScriptFileName(strDefaultPVRES);
	}

	m_cPVRES = cParser.Parse();


	// sets up whether the console should write out constantly or not
	CPVRTString strLogPath = CPVRTString((char*)PVRShellGet(prefWritePath));
	m_pConsole->setOutputFile(strLogPath+="log.txt");
	m_pConsole->setStraightToFile(m_cPVRES.getLogToFile());

	m_pConsole->log("PODPlayer v0.2 alpha\n\n Initialising...\n");


	CPVRTString error = cParser.getError();
	if(!error.empty())
	{
		m_pConsole->log("Couldn't parse script: %s:\n%s",m_cPVRES.getScriptFileName().c_str(),error.c_str());
		PVRShellSet(prefExitMessage, m_pConsole->getLastLogLine().c_str());
		return false;
	}

	// Deal with results of script read


	// Load the scene from the .pod file into a CPVRTModelPOD object.
	if(m_Scene.ReadFromFile(m_cPVRES.getPODFileName().c_str()) != PVR_SUCCESS)
	{
		m_pConsole->log("Error: couldn't open POD file: %s\n",m_cPVRES.getPODFileName().c_str());
		PVRShellSet(prefExitMessage, m_pConsole->getLastLogLine().c_str());
		return false;
	}

	// The cameras are stored in the file. Check if it contains at least one.
	if(m_Scene.nNumCamera == 0)
	{
		m_bFreeCamera = true;
	}
	else
	{
		m_bFreeCamera = false;
	}
	// use camera 0 to begin with
	m_u32CurrentCameraNum = 0;

	// Ensure that all meshes use an indexed triangle list
	for(unsigned int i = 0; i < m_Scene.nNumMesh; ++i)
	{
		if(m_Scene.pMesh[i].nNumStrips || !m_Scene.pMesh[i].sFaces.pData)
		{
			m_pConsole->log("ERROR: The meshes in the scene should use an indexed triangle list\n");
			PVRShellSet(prefExitMessage, m_pConsole->getLastLogLine().c_str());
			return false;
		}
	}

	PVRShellSet(prefFSAAMode,m_cPVRES.getFSAA());					// set fullscreen anti-aliasing
	PVRShellSet(prefPowerSaving,m_cPVRES.getPowerSaving());			// set power saving mode
	PVRShellSet(prefHeight,m_cPVRES.getHeight());					// set height of window
	PVRShellSet(prefWidth,m_cPVRES.getWidth());						// set width of window
	PVRShellSet(prefPositionX,m_cPVRES.getPosX());					// set horizontal position of window
	PVRShellSet(prefPositionY,m_cPVRES.getPosY());					// set vertical position of window
	PVRShellSet(prefQuitAfterTime,m_cPVRES.getQuitAfterTime());		// time after which PODPlayer will automatically quit
	PVRShellSet(prefQuitAfterFrame,m_cPVRES.getQuitAfterFrame());	// frame after which PODplayer will automatically quit
	PVRShellSet(prefSwapInterval,m_cPVRES.getVertSync()?1:0);		// set vertical sync with monitor
	PVRShellSet(prefFullScreen, m_cPVRES.getFullScreen());			// set fullscreen

	m_pUniformHandler->setScene(&m_Scene);
	m_pUniformHandler->setLightManager(LightManager::ptr());

	// Initialize variables used for the animation
	m_bOptions = false;												// don't show options at start up
	m_bOverlayOptions = false;										// don't overlay the options by default
	m_pTimeController->setNumFrames(m_Scene.nNumFrame);				// set the number of frames to animate across
	m_pTimeController->setFrame(m_cPVRES.getStartFrame());			// set the frame from which to start the animation
	m_pTimeController->setAnimationSpeed(m_cPVRES.getAnimationSpeed());	// set the speed with which to animate


	// set PODPlayer to initialising state
	m_i32Initialising = 1;

	m_pConsole->log("Initial setup Succeeded\n");
	return true;
}


/******************************************************************************/

bool PODPlayer::QuitApplication()
{
	// delete meshes
	PVRDELETE(m_psMeshManager);
	PVRDELETE(m_psTransparentMeshManager);
	// delete options
	PVRDELETE(m_pOptionsMenu);
	return true;
}

/******************************************************************************/

bool PODPlayer::RenderScene()
{
	//TODO: put this GL specific code somewhere
	int j=0;
	do
	{
		j = glGetError();
		if(j)
		{
			ConsoleLog::inst().log("GL Error: %d %s\n",j,glGetString(j));;
		}
	}while(j);

	if(m_i32Initialising)
	{	// do initialise
		return Init(m_i32Initialising);
	}
	doInput();
	// Clears the color and depth buffer
	m_PVRESettings.Clear();
	doFPS();

	// are we in the options menu
	if(m_bOptions)
	{
		if(m_bOverlayOptions)
		{	// restore background colour
			do3DScene();
		}
		m_Print3D.DisplayDefaultTitle("PODPlayer Options", "", ePVRTPrint3DLogoIMG);
		m_pOptionsMenu->render();
	}
	else
	{
		do3DScene();
		if(m_cPVRES.getShowFPS())
		{
			m_Print3D.Print3D(1.0f, 15.0f, 0.5f, 0xFFff99aa, "FPS:%.1f",m_pTimeController->getFPS());
		}

		// Displays the demo name using the tools. For a detailed explanation, see the training course IntroducingPVRTools
		m_Print3D.DisplayDefaultTitle(m_strDemoTitle.c_str(), "", ePVRTPrint3DLogoIMG);
	}
	m_Print3D.Flush();
	return true;
}

/******************************************************************************/

bool PODPlayer::ReleaseView()
{
	m_pConsole->log("Exiting...\n\n");

	// Release Print3D Textures
	m_Print3D.ReleaseTextures();

	return true;
}

/*!****************************************************************************
@Function		doInput
@Description	Deals with keyboard input
******************************************************************************/
void PODPlayer::doInput()
{
	bool bUpdateOptions = false;	// grab options from menu?
	if(PVRShellIsKeyPressed(PVRShellKeyNameSELECT))
	{	// switch between options screen and normal
		m_bOptions=!m_bOptions;
		bUpdateOptions = true;
	}

	if(m_bOptions)
	{	// in options screen
		if(PVRShellIsKeyPressed(PVRShellKeyNameLEFT))
		{
			m_pOptionsMenu->prevValue();
			bUpdateOptions = true;
		}
		else if(PVRShellIsKeyPressed(PVRShellKeyNameRIGHT))
		{
			m_pOptionsMenu->nextValue();
			bUpdateOptions = true;
		}

		if(PVRShellIsKeyPressed(PVRShellKeyNameUP))
		{
			m_pOptionsMenu->prevOption();
			bUpdateOptions = true;
		}
		else if(PVRShellIsKeyPressed(PVRShellKeyNameDOWN))
		{
			m_pOptionsMenu->nextOption();
			bUpdateOptions = true;
		}
	}
	else
	{	// not options screen

		if(m_bFreeCamera)
		{

			if(PVRShellIsKeyPressed(PVRShellKeyNameLEFT))
			{
				m_sCamera.YawLeft();
			}
			else if(PVRShellIsKeyPressed(PVRShellKeyNameRIGHT))
			{
				m_sCamera.YawRight();
			}

			if(PVRShellIsKeyPressed(PVRShellKeyNameUP))
			{
				m_sCamera.PitchDown();
			}
			else if(PVRShellIsKeyPressed(PVRShellKeyNameDOWN))
			{
				m_sCamera.PitchUp();
			}

			if(PVRShellIsKeyPressed(PVRShellKeyNameACTION1))
			{	// move forward
				m_sCamera.MoveForward();
			}
			else if(PVRShellIsKeyPressed(PVRShellKeyNameACTION2))
			{	// move backward
				m_sCamera.MoveBack();
			}

		}
		else
		{
			if(PVRShellIsKeyPressed(PVRShellKeyNameLEFT))
			{	// rewind
				m_pTimeController->rewind();
			}
			else if(PVRShellIsKeyPressed(PVRShellKeyNameRIGHT))
			{	// fast forward
				m_pTimeController->fastforward();
			}

		}
	}

	// grab settings changes from the options menu if appropriate
	if(bUpdateOptions)
		getOptions();


}

/*!****************************************************************************
@Function		doFPS
@Description	does frames per second calc, deals with advancing the animation
in general
******************************************************************************/
void PODPlayer::doFPS()
{
	/*
	Calculates the frame number to animate in a time-based manner.
	Uses the shell function PVRShellGetTime() to get the time in milliseconds.
	*/

	float fFrame = m_pTimeController->getFrame(PVRShellGetTime());

	// Sets the scene animation to this frame
	m_Scene.SetFrame(f2vt(fFrame));
	// Sets value for if the shaders need it
	m_pUniformHandler->setFrame(fFrame);
}

/*!****************************************************************************
@Function		doCamera
@Description	Deals with the current POD or free camera to set up the
projection and view matrices
******************************************************************************/
void PODPlayer::doCamera()
{
	// set up the camera
	if(m_bFreeCamera)
	{	// from the free camera class
		m_sCamera.updatePosition();

		// Calculates the projection matrix
		m_pUniformHandler->setProjection(m_sCamera.getFOV(),
			f2vt(m_sCamera.getAspect()),
			m_sCamera.getNear(),
			m_sCamera.getFar(),
			m_bRotate);

		// build the model view matrix from the camera position, target and an up vector.
		PVRTVec3 vUp, vTo, vPosition(m_sCamera.getPosition());
		m_sCamera.getTargetAndUp(vTo,vUp);
		m_pUniformHandler->setView(vPosition,vTo,vUp);
	}
	else
	{	// from the POD camera values
		PVRTVec3 vFrom, vTo, vUp;
		VERTTYPE fFOV;

		// get the camera position, target and field of view (fov) with GetCameraPos()
		fFOV = m_Scene.GetCamera( vFrom, vTo, vUp, m_u32CurrentCameraNum);

		// Calculates the projection matrix
		m_pUniformHandler->setProjection(fFOV,
			m_sCamera.getAspect(),
			m_Scene.pCamera[m_u32CurrentCameraNum].fNear,
			m_Scene.pCamera[m_u32CurrentCameraNum].fFar,
			m_bRotate);

		// build the model view matrix from the camera position, target and an up vector.
		m_pUniformHandler->setView(vFrom,vTo,vUp);
	}
}

/*!****************************************************************************
@Function		do3DScene
@Description	Renders the actual POD scene according to the current settings
******************************************************************************/
void PODPlayer::do3DScene()
{

	// clears the active material for the new frame
	MaterialManager::inst().ReportActiveMaterial(NULL);
	m_pUniformHandler->ResetFrameUniforms();


	doCamera();

	// deal with lights
	dynamicArray<Light*> *pdaLights = m_pLightManager->getLights();
	for(unsigned int i=0;i<pdaLights->getSize();i++)
	{	// update the lights from the POD
		Light* pLight = (*pdaLights)[i];
		PVRTVec3 vPos,vDir;
		if(pLight->getType()==eLight_Point)
		{
			vPos = m_Scene.GetLightPosition(i);
			((LightPoint*)pLight)->setPosition(vPos);
		}
		else
		{
			vDir = m_Scene.GetLightDirection(i);
			((LightDirectional*)pLight)->setDirection(vDir);
		}
		pLight->shineLight(i);
	}

	// get the meshes to draw
	dynamicArray<Mesh*> *pdaMeshes = m_psMeshManager->getMeshes();
	if(pdaMeshes->getSize())
	{
		for(unsigned int i=0;i<pdaMeshes->getSize();i++)
		{
			// Gets the node model matrix for this frame
			Mesh *pMesh = (*pdaMeshes)[i];
			PVRTMat4 mWorld = m_Scene.GetWorldMatrix(*pMesh->getNode());
			m_pUniformHandler->setWorld(mWorld);
			// if mesh is visible - frustum test for camera
			if(m_pUniformHandler->isVisibleSphere(pMesh->getCentre(),pMesh->getRadius()))
			{	// draw it
				pMesh->draw();
			}
			// else forget it
		}

	}

	// get the transparent meshes to draw
	pdaMeshes = m_psTransparentMeshManager->getMeshes();
	if(pdaMeshes->getSize())
	{
		//do some simple blending
		m_PVRESettings.setBlend(true);
		m_PVRESettings.setDepthWrite(false);

		// cull
		for(unsigned int i=0;i<pdaMeshes->getSize();i++)
		{
			// Gets the node model matrix for this frame
			Mesh *pMesh = (*pdaMeshes)[i];
			PVRTMat4 mWorld = m_Scene.GetWorldMatrix(*pMesh->getNode());
			m_pUniformHandler->setWorld(mWorld);
			// if mesh is visible - frustum test for camera
			if(m_pUniformHandler->isVisibleSphere(pMesh->getCentre(),pMesh->getRadius()))
			{	// draw it
				pMesh->draw();
			}
			// else forget it
		}
	}

	m_PVRESettings.setBlend(false);
	m_PVRESettings.setDepthWrite(true);
}

/*!****************************************************************************
@Function		getOptions
@Description	Updates the PODPlayer settings from the options chosen in the 
OptionsMenu
******************************************************************************/
void PODPlayer::getOptions()
{
	// Overlay options??
	m_bOverlayOptions = m_pOptionsMenu->getValueBool(eOptions_OverlayOptions);
	if(m_bOverlayOptions || !m_bOptions)
	{
		m_PVRESettings.setBackColour((VERTTYPE) m_Scene.pfColourBackground[0],(VERTTYPE) m_Scene.pfColourBackground[1],(VERTTYPE) m_Scene.pfColourBackground[2]);
	}
	else
	{
		m_PVRESettings.setBackColour(c_u32MenuBackgroundColour);
	}

	// Pause
	m_pTimeController->setFreezeTime(m_pOptionsMenu->getValueBool(eOptions_Pause));

	// FPS
	m_cPVRES.setShowFPS(m_pOptionsMenu->getValueBool(eOptions_FPS));

	// POD Camera
	m_u32CurrentCameraNum = m_pOptionsMenu->getValueInt(eOptions_PODCamera);

	// Do free camera
	bool bFreeCamera = m_pOptionsMenu->getValueBool(eOptions_FreeCamera);
	if(!m_bFreeCamera && bFreeCamera)
	{	// not already free but needs to be now so match view
		PVRTVec3 vFrom, vTo, vUp;
		m_sCamera.setFOV(m_Scene.GetCamera( vFrom, vTo, vUp,m_u32CurrentCameraNum));

		m_sCamera.setPosition(vFrom);
		m_sCamera.setTarget(vTo);
		m_sCamera.setNear(m_Scene.pCamera[m_u32CurrentCameraNum].fNear);
		m_sCamera.setFar(m_Scene.pCamera[m_u32CurrentCameraNum].fFar);
	}
	m_bFreeCamera = bFreeCamera;

	// invert free camera up down controls
	m_sCamera.setInverted(m_pOptionsMenu->getValueBool(eOptions_Invert));

	// movement speed and rotation speed
	m_sCamera.setMoveSpeed(f2vt(m_pOptionsMenu->getValueFloat(eOptions_MoveSpeed)));
	m_sCamera.setRotSpeed(f2vt(m_pOptionsMenu->getValueFloat(eOptions_RotateSpeed)));

	// Draw Mode
	m_psMeshManager->setDrawMode((Mesh::DrawMode)m_pOptionsMenu->getValueEnum(eOptions_DrawMode));
	m_psTransparentMeshManager->setDrawMode((Mesh::DrawMode)m_pOptionsMenu->getValueEnum(eOptions_DrawMode));

	// Direction of play
	m_pTimeController->setForwards(m_pOptionsMenu->getValueBool(eOptions_Direction));

	// Frame rate
	m_pTimeController->setAnimationSpeed(m_pOptionsMenu->getValueFloat(eOptions_AnimationSpeed));
}

/*!****************************************************************************
@Function		Init
@Return			bool		whether this stage of initialisation was successful
@Description	Switches through each stage of initialisation for the PODPlayer
******************************************************************************/
bool PODPlayer::Init(int& i32Initialising)
{
	switch(i32Initialising)
	{
	case eINIT_BASIC:
		// this is the best way to determine if the view is rotated.
		m_bRotate = PVRShellGet(prefIsRotated) && PVRShellGet(prefFullScreen);

		m_PVRESettings.Init();		// mandatory initialisation step
		m_PVRESettings.setClearFlags(PVR_COLOUR_BUFFER|PVR_DEPTH_BUFFER);
		m_PVRESettings.setDepthTest(true);
		m_PVRESettings.setCullMode(PVR_BACK);

		m_sCamera.setAspect((VERTTYPE)PVRShellGet(prefHeight),(VERTTYPE)PVRShellGet(prefWidth));

		m_PVRESettings.Clear();

		m_pConsole->log("Basic initialisation complete\n");
		break;
	case eINIT_PRINT3D:
		m_PVRESettings.Clear(); //TODO: Clear twice shouldn't be necessary

		m_PVRESettings.InitPrint3D(m_Print3D,PVRShellGet(prefWidth),PVRShellGet(prefHeight), m_bRotate);
		m_pConsole->log("Print3D initialisation complete\n");
		break;
	case eINIT_BASICSCENE:
		{
			m_strDemoTitle = m_PVRESettings.getAPIName() + " " + (m_cPVRES.getTitle());

			sprintf(m_pszPrint3DString,"Opened Scene.");
			m_pConsole->log("Opened Scene.\n");
		}
		break;
	case eINIT_MATERIALS:
		{
			//	Load the material files
			for(unsigned int i=0;i<m_Scene.nNumMaterial;++i)
			{
				m_pConsole->log("Loading Material: %s\n",m_Scene.pMaterial[i].pszName);
				// because POD stores the textures that may be used by a material separately from the material (why? why? why?  WTF?!?)
				// have to retrieve and pass texture now even if it's completely ignored
				Material *psMaterial = MaterialManager::inst().LoadMaterial(m_cPVRES.getPFXPath(),
					m_cPVRES.getTexturePath(),
					m_Scene.pMaterial[i],
					m_Scene.pTexture[m_Scene.pMaterial[i].nIdxTexDiffuse]);
				if(!psMaterial)
				{
					m_pConsole->log("Error: Material failed to initialise\n");
					PVRShellSet(prefExitMessage, m_pConsole->getLastLogLine().c_str());
					return false;
				}
			}

			sprintf(m_pszPrint3DString,"Initialised Materials.");
			m_pConsole->log("Initialised materials.\n");
		}
		break;
	case eINIT_MESHES:
		{
			m_pConsole->log("Initialising meshes.\n");

			m_psMeshManager = new MeshManager();
			m_psTransparentMeshManager = new MeshManager();

			unsigned int u32NumMeshes = m_Scene.nNumMeshNode;

			if(u32NumMeshes)
			{
				for(unsigned int i = 0; i < u32NumMeshes ; i++)
				{
					SPODNode* pNode = &m_Scene.pNode[i];

					// Gets pMesh referenced by the pNode
					SPODMesh* pMesh = &m_Scene.pMesh[pNode->nIdx];

					// sort meshes into opaque and transparent
					if(MaterialManager::inst().getMaterial(pNode->nIdxMaterial)->getOpacity()==1.0f)
						m_psMeshManager->addMesh(m_Scene,pNode, pMesh);
					else
						m_psTransparentMeshManager->addMesh(m_Scene, pNode, pMesh);
				}
			}
			// sort the meshes in terms of materials used
			// to avoid unnecessary shader loading etc.
			m_pConsole->log("Sorting meshes...");
			m_psMeshManager->sort();
			m_psTransparentMeshManager->sort();
			m_pConsole->log("Sorted\n");

			sprintf(m_pszPrint3DString,"Initialised meshes.");
			m_pConsole->log("Initialised meshes.\n");
		}
		break;
	case eINIT_LIGHTS:
		{
			m_pLightManager  = LightManager::ptr();
			// go through lights in scene and add them to the manager
			// We check if the scene contains any lights
			if (m_Scene.nNumLight == 0)
			{
				m_pConsole->log("Warning: no lights found in scene.\n");
				m_pConsole->log("Adding a default directional light at (0,-1,0).\n");

				// add a default light

				m_pLightManager->addDirectionalLight(PVRTVec3(0.f,-1.f,0.f),PVRTVec3(1.f,1.f,1.f));



			}
			else
			{
				for(unsigned int i=0;i<m_Scene.nNumLight;i++)
				{
					// add light i from this scene to the manager
					m_pLightManager->addLight(m_Scene,i);
				}
			}

			sprintf(m_pszPrint3DString,"Initialised lights.");
			m_pConsole->log("Initialised lights.\n");
		}
		break;
	case eINIT_MENU:
		{
			// initialises options menu and sets options values
			m_pOptionsMenu = new OptionsMenu(&m_Print3D);
			m_pOptionsMenu->addOption(new OptionEnum("Overlay Options",strOnOff,2,0));
			m_pOptionsMenu->addOption(new OptionEnum("Pause",strOnOff,2,0));
			m_pOptionsMenu->addOption(new OptionEnum("Draw Mode",Mesh::g_strDrawModeNames,Mesh::eNumDrawModes,m_cPVRES.getDrawMode()));
			m_pOptionsMenu->addOption(new OptionInt("POD Camera",0,m_Scene.nNumCamera-1,1,0));
			m_pOptionsMenu->addOption(new OptionEnum("Free Camera",strOnOff,2,m_bFreeCamera?1:0));
			m_pOptionsMenu->addOption(new OptionEnum("  Invert Up/Down", strOnOff,2,m_sCamera.getInverted()));
			m_pOptionsMenu->addOption(new OptionFloat("  Movement Speed",0.5f,100.0f,2.0f,10.0f));
			m_pOptionsMenu->addOption(new OptionFloat("  Rotation Speed",0.01f,0.5f,0.05f,0.05f));
			m_pOptionsMenu->addOption(new OptionEnum("Show FPS",strOnOff,2,m_cPVRES.getShowFPS()));
			m_pOptionsMenu->addOption(new OptionEnum("Play Direction",strForwardBackward,2,m_pTimeController->getForwards()));
			m_pOptionsMenu->addOption(new OptionFloat("Animation Speed",-10,10,0.2f,m_pTimeController->getAnimationSpeed()));

			getOptions();	// update settings to reflect options

			sprintf(m_pszPrint3DString,"Initialised menu.");
			m_pConsole->log("Initialised menu.\n");
		}
		break;

	case eINIT_FINISH:
		i32Initialising=0;
		m_pConsole->log("Initialisation Complete.\n");
		m_pConsole->log(" ");
		sprintf(m_pszPrint3DString," ");
		// starts time controller from this moment
		m_pTimeController->start(PVRShellGetTime());
		return true;
	}
	if(i32Initialising>eINIT_PRINT3D)	// i.e. if print3D is intialised
	{
		m_Print3D.Print3D(4.0f,(float)i32Initialising*5.0f,0.5f,0xff00ffff,m_pszPrint3DString);
		m_Print3D.DisplayDefaultTitle(m_strDemoTitle.c_str(), " ", ePVRTPrint3DLogoIMG);
		m_Print3D.Flush();
	}
	i32Initialising++;
	return true;
}



/*!****************************************************************************
@Function		NewDemo
@Return			PVRShell*		The demo supplied by the user
@Description	This function must be implemented by the user of the shell.
The user should return its PVRShell object defining the
behaviour of the application.
******************************************************************************/
PVRShell* NewDemo()
{
	return new PODPlayer();
}
/******************************************************************************
End of file (PODPlayer.cpp)
******************************************************************************/
