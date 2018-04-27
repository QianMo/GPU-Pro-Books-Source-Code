/******************************************************************************

 @File         OGLES2Skybox2.cpp

 @Title        Introducing the POD 3d file format

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independant

 @Description  Shows how to load POD files and play the animation with basic
               lighting

******************************************************************************/
#include <string.h>
#include <math.h>

#include "PVRShell.h"
#include "OGLES2Tools.h"

/******************************************************************************
 Defines
******************************************************************************/

// Camera constants. Used for making the projection matrix

// Shader semantics recognised by this program
enum EUniformSemantic
{
	eUsUnknown,
	eUsPOSITION,
	eUsNORMAL,
	eUsUV,
	eUsWORLDVIEWPROJECTION,
	eUsWORLDVIEW,
	eUsWORLDVIEWIT,
	eUsVIEWIT,
	eUsLIGHTDIRECTION,
	eUsTEXTURE
};

/****************************************************************************
** Constants
****************************************************************************/
const static SPVRTPFXUniformSemantic c_psUniformSemantics[] =
{
	{ "POSITION",				eUsPOSITION },
	{ "NORMAL",					eUsNORMAL },
	{ "UV",						eUsUV },
	{ "WORLDVIEWPROJECTION",	eUsWORLDVIEWPROJECTION },
	{ "WORLDVIEW",				eUsWORLDVIEW },
	{ "WORLDVIEWIT",			eUsWORLDVIEWIT },
	{ "VIEWIT",					eUsVIEWIT},
	{ "LIGHTDIRECTION",			eUsLIGHTDIRECTION },
	{ "TEXTURE",				eUsTEXTURE }
};

struct SEffect
{
	CPVRTPFXEffect	*pEffect;
	SPVRTPFXUniform	*psUniforms;
	unsigned int	 ui32UniformCount;

	SEffect()
	{
		pEffect = 0;
		psUniforms = 0;
		ui32UniformCount = 0;
	}
};

const float g_fFrameRate = 1.0f / 30.0f;
const unsigned int g_ui32NoOfEffects = 8;
const unsigned int g_ui32TexNo = 5;

const bool g_bBlendShader[g_ui32NoOfEffects] = {
	false,
	false,
	false,
	false,
	true,
	false,
	false,
	true
};

/******************************************************************************
 Content file names
******************************************************************************/

// POD scene files
const char c_szSceneFile[] = "Scene.pod";

// Textures
const char * const g_aszTextureNames[g_ui32TexNo] = {
	"Balloon.pvr",
	"Balloon_pvr.pvr",
	"Noise.pvr",
	"Skybox.pvr",
	"SkyboxMidnight.pvr"
};

// PFX file
const char * const g_pszEffectFileName = "effects.pfx";

/*!****************************************************************************
 Class implementing the PVRShell functions.
******************************************************************************/
class OGLES2Skybox2 : public PVRShell
{
	// Print3D class used to display text
	CPVRTPrint3D	m_Print3D;

	// IDs for the various textures
	GLuint m_ui32TextureIDs[g_ui32TexNo];
	// 3D Model
	CPVRTModelPOD	m_Scene;

	// Projection and Model View matrices
	PVRTMat4		m_mProjection, m_mView;

	// Variables to handle the animation in a time-based manner
	int				m_iTimePrev;
	float			m_fFrame;

	// The effect currently being displayed
	int m_i32Effect;

	// The Vertex buffer object handle array.
	GLuint			*m_aiVboID;
	GLuint			m_iSkyVboID;

	/* View Variables */
	VERTTYPE fViewAngle;
	VERTTYPE fViewDistance, fViewAmplitude, fViewAmplitudeAngle;
	VERTTYPE fViewUpDownAmplitude, fViewUpDownAngle;

	/* Vectors for calculating the view matrix and saving the camera position*/
	PVRTVec3 vTo, vUp, vCameraPosition;

	/* Skybox */
	VERTTYPE* g_SkyboxVertices;
	VERTTYPE* g_SkyboxUVs;

	//animation
	float fBurnAnim;

	bool bPause;
	float fDemoFrame;

	// The variables required for the effects
	CPVRTPFXParser	*m_pEffectParser;
	SEffect *m_pEffects;

public:
	OGLES2Skybox2()
	{
		/* Init values to defaults */
		fViewAngle = PVRT_PI_OVER_TWO;

		fViewDistance = 100.0f;
		fViewAmplitude = 60.0f;
		fViewAmplitudeAngle = 0.0f;

		fViewUpDownAmplitude = 50.0f;
		fViewUpDownAngle = 0.0f;

		vTo.x = 0;
		vTo.y = 0;
		vTo.z = 0;

		vUp.x = 0;
		vUp.y = 1;
		vUp.z = 0;
	}
	virtual bool InitApplication();
	virtual bool InitView();
	virtual bool ReleaseView();
	virtual bool QuitApplication();
	virtual bool RenderScene();

	void DrawMesh(SPODMesh* mesh);
	void ComputeViewMatrix();
	void DrawSkybox();
	bool LoadEffect(SEffect *pSEffect, const char * pszEffectName, const char *pszFileName);
	bool LoadTextures(CPVRTString* const pErrorStr);
	bool DestroyEffect(SEffect *pSEffect);
	void ChangeSkyboxTo(SEffect *pSEffect, GLuint ui32NewSkybox);
};


/*!****************************************************************************
 @Function		InitApplication
 @Return		bool		true if no error occured
 @Description	Code in InitApplication() will be called by PVRShell once per
				run, before the rendering context is created.
				Used to initialize variables that are not dependant on it
				(e.g. external modules, loading meshes, etc.)
				If the rendering context is lost, InitApplication() will
				not be called again.
******************************************************************************/
bool OGLES2Skybox2::InitApplication()
{
	// Get and set the read path for content files
	CPVRTResourceFile::SetReadPath((char*)PVRShellGet(prefReadPath));

	// Load the scene
	if (m_Scene.ReadFromFile(c_szSceneFile) != PVR_SUCCESS)
	{
		PVRShellSet(prefExitMessage, "ERROR: Couldn't load the .pod file\n");
		return false;
	}

	// Initialise variables used for the animation
	m_fFrame = 0;
	m_iTimePrev = PVRShellGetTime();

	return true;
}

/*!****************************************************************************
 @Function		QuitApplication
 @Return		bool		true if no error occured
 @Description	Code in QuitApplication() will be called by PVRShell once per
				run, just before exiting the program.
				If the rendering context is lost, QuitApplication() will
				not be called.
******************************************************************************/
bool OGLES2Skybox2::QuitApplication()
{
	// Frees the memory allocated for the scene
	m_Scene.Destroy();

    return true;
}

bool OGLES2Skybox2::LoadTextures(CPVRTString* const pErrorStr)
{
	for(int i = 0; i < 3; ++i)
	{
		if(PVRTTextureLoadFromPVR(g_aszTextureNames[i], &m_ui32TextureIDs[i]) != PVR_SUCCESS)
			return false;
			
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	// Load cube maps
	for(int i = 3; i < 5; ++i)
	{
		if(PVRTTextureLoadFromPVR(g_aszTextureNames[i], &m_ui32TextureIDs[i]))
		{
			*pErrorStr = CPVRTString("ERROR: Could not open texture file ") + g_aszTextureNames[i];
			return false;
		}

		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	return true;
}
/*!****************************************************************************
 @Function		InitView
 @Return		bool		true if no error occured
 @Description	Code in InitView() will be called by PVRShell upon
				initialization or after a change in the rendering context.
				Used to initialize variables that are dependant on the rendering
				context (e.g. textures, vertex buffers, etc.)
******************************************************************************/
bool OGLES2Skybox2::InitView()
{
	// Sets the clear color
	glClearColor(0.6f, 0.8f, 1.0f, 1.0f);

	// Enables depth test using the z-buffer
	glEnable(GL_DEPTH_TEST);

	CPVRTString ErrorStr;

	/*
		Load textures
	*/
	if(!LoadTextures(&ErrorStr))
	{
		PVRShellSet(prefExitMessage, ErrorStr.c_str());
		return false;
	}

	/*********************/
	/* Create the Skybox */
	/*********************/
	PVRTCreateSkybox( 500.0f, true, 512, &g_SkyboxVertices, &g_SkyboxUVs );


	/**********************/
	/* Create the Effects */
	/**********************/

	{
		// Parse the file
		m_pEffectParser = new CPVRTPFXParser();

		if(m_pEffectParser->ParseFromFile(g_pszEffectFileName, &ErrorStr) != PVR_SUCCESS)
		{
			delete m_pEffectParser;
			PVRShellSet(prefExitMessage, ErrorStr.c_str());
			return false;
		}

		m_pEffects = new SEffect[m_pEffectParser->m_nNumEffects];

		// Skybox shader
		if(!LoadEffect(&m_pEffects[0], "skybox_effect", g_pszEffectFileName))
		{
			delete m_pEffectParser;
			delete[] m_pEffects;
			return false;
		}

		// The Balloon Shaders
		if(!LoadEffect(&m_pEffects[1], "balloon_effect1", g_pszEffectFileName) ||
			!LoadEffect(&m_pEffects[2], "balloon_effect2", g_pszEffectFileName) ||
			!LoadEffect(&m_pEffects[3], "balloon_effect3", g_pszEffectFileName) ||
			!LoadEffect(&m_pEffects[4], "balloon_effect4", g_pszEffectFileName) ||
			!LoadEffect(&m_pEffects[5], "balloon_effect5", g_pszEffectFileName) ||
			!LoadEffect(&m_pEffects[6], "balloon_effect6", g_pszEffectFileName) ||
			!LoadEffect(&m_pEffects[7], "balloon_effect7", g_pszEffectFileName))
		{
			delete m_pEffectParser;
			delete[] m_pEffects;
			return false;
		}
	}

	// Create Geometry Buffer Objects.
	m_aiVboID = new GLuint[m_Scene.nNumMeshNode];
	glGenBuffers(m_Scene.nNumMeshNode, m_aiVboID);

	for(unsigned int i = 0; i < m_Scene.nNumMeshNode ; ++i)
	{
		SPODNode* pNode = &m_Scene.pNode[i];

		// Gets pMesh referenced by the pNode
		SPODMesh* pMesh = &m_Scene.pMesh[pNode->nIdx];

		// Genereta a vertex buffer and set the interleaved vertex datas.
		glBindBuffer(GL_ARRAY_BUFFER, m_aiVboID[i]);
		glBufferData(GL_ARRAY_BUFFER, pMesh->sVertex.nStride*pMesh->nNumVertex, pMesh->pInterleaved, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	}

	glGenBuffers(1, &m_iSkyVboID);
	glBindBuffer(GL_ARRAY_BUFFER, m_iSkyVboID);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VERTTYPE)*3 * 8, &g_SkyboxVertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/**********************
	** Projection Matrix **
	**********************/
	/* Projection */
	bool bRotate = PVRShellGet(prefIsRotated) && PVRShellGet(prefFullScreen);
	m_mProjection = PVRTMat4::PerspectiveFovRH(PVRT_PI / 6, (float) PVRShellGet(prefWidth) / (float) PVRShellGet(prefHeight), 4.0f, 1000.0f, PVRTMat4::OGL, bRotate);

	/* Init Values */
	bPause = false;
	fDemoFrame = 0.0;
	fBurnAnim = 0.0f;

	m_i32Effect = 1;

	/*
		Initialise Print3D
	*/
	if(m_Print3D.SetTextures(0,PVRShellGet(prefWidth),PVRShellGet(prefHeight), bRotate) != PVR_SUCCESS)
	{
		PVRShellSet(prefExitMessage, "ERROR: Cannot initialise Print3D\n");
		return false;
	}

	return true;
}

bool OGLES2Skybox2::LoadEffect(SEffect *pSEffect, const char * pszEffectName, const char *pszFileName)
{
	if(!pSEffect)
		return false;

	unsigned int	nUnknownUniformCount;
	CPVRTString		error;

	// Load an effect from the file
	if(!pSEffect->pEffect)
	{
		pSEffect->pEffect = new CPVRTPFXEffect();

		if(!pSEffect->pEffect)
		{
			delete m_pEffectParser;
			PVRShellSet(prefExitMessage, "Failed to create effect.\n");
			return false;
		}
	}

	if(pSEffect->pEffect->Load(*m_pEffectParser, pszEffectName, pszFileName, &error) != PVR_SUCCESS)
	{
		PVRShellSet(prefExitMessage, error.c_str());
		return false;
	}

	// Generate uniform array
	if(pSEffect->pEffect->BuildUniformTable(&pSEffect->psUniforms, &pSEffect->ui32UniformCount, &nUnknownUniformCount,
							c_psUniformSemantics, sizeof(c_psUniformSemantics) / sizeof(*c_psUniformSemantics), &error) != PVR_SUCCESS)
	{
		PVRShellSet(prefExitMessage, error.c_str());
		return false;
	}

	if(nUnknownUniformCount)
	{
		PVRShellOutputDebug(error.c_str());
		PVRShellOutputDebug("Unknown uniform semantic count: %d\n", nUnknownUniformCount);
	}

	/* Set the textures for each effect */
	const SPVRTPFXTexture	*psTex;
	unsigned int			nCnt, i,j ;


	psTex = pSEffect->pEffect->GetTextureArray(nCnt);

	for(i = 0; i < nCnt; ++i)
	{
		for(j = 0; j < g_ui32TexNo; ++j)
		{
			if(strcmp(g_aszTextureNames[j], psTex[i].p) == 0)
			{
				if(j == 3 || j == 4)
					pSEffect->pEffect->SetTexture(i, m_ui32TextureIDs[j], PVRTEX_CUBEMAP);
				else
					pSEffect->pEffect->SetTexture(i, m_ui32TextureIDs[j]);

				break;
			}
		}
	}

	return true;
}

bool OGLES2Skybox2::DestroyEffect(SEffect *pSEffect)
{
	if(pSEffect)
	{
		if(pSEffect->pEffect)
		{
			const SPVRTPFXTexture	*psTex;
			unsigned int			nCnt, i;

			psTex = pSEffect->pEffect->GetTextureArray(nCnt);

			for(i = 0; i < nCnt; ++i)
			{
				glDeleteTextures(1, &(psTex[i].ui));
			}

			delete pSEffect->pEffect;
			pSEffect->pEffect = 0;
		}

		FREE(pSEffect->psUniforms);
	}

	return true;
}

/*!****************************************************************************
 @Function		ReleaseView
 @Return		bool		true if no error occured
 @Description	Code in ReleaseView() will be called by PVRShell when the
				application quits or before a change in the rendering context.
******************************************************************************/
bool OGLES2Skybox2::ReleaseView()
{
	// Free the textures
	unsigned int i;

	for(i = 0; i < g_ui32TexNo; ++i)
	{
		glDeleteTextures(1, &(m_ui32TextureIDs[i]));
	}

	// Release Print3D Textures
	m_Print3D.ReleaseTextures();

	// Release Vertex buffer objects.
	glDeleteBuffers(m_Scene.nNumMeshNode, m_aiVboID);
	glDeleteBuffers(1, &m_iSkyVboID);
	delete[] m_aiVboID;

	// Destroy the Skybox
	PVRTDestroySkybox( g_SkyboxVertices, g_SkyboxUVs );

	for(i = 0; i < m_pEffectParser->m_nNumEffects; ++i)
		DestroyEffect(&m_pEffects[i]);

	delete[] m_pEffects;
	delete m_pEffectParser;

	return true;
}

/*!****************************************************************************
 @Function		RenderScene
 @Return		bool		true if no error occured
 @Description	Main rendering loop function of the program. The shell will
				call this function every frame.
				eglSwapBuffers() will be performed by PVRShell automatically.
				PVRShell will also manage important OS events.
				Will also manage relevent OS events. The user has access to
				these events through an abstraction layer provided by PVRShell.
******************************************************************************/
bool OGLES2Skybox2::RenderScene()
{
	unsigned int i, j;

	// Clears the colour and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/*
		Calculates the frame number to animate in a time-based manner.
		Uses the shell function PVRShellGetTime() to get the time in milliseconds.
	*/

	int iTime      = PVRShellGetTime();
	int iDeltaTime = iTime - m_iTimePrev;

	float fDelta = (float)iDeltaTime * g_fFrameRate;
	m_iTimePrev	= iTime;

	if(!bPause)
	{
		m_fFrame   += fDelta;
		fDemoFrame += fDelta;
		fBurnAnim  += fDelta * 0.02f;

		if(fBurnAnim >= 1.0f)
			fBurnAnim = 1.0f;
	}

	/* KeyBoard input processing */

	if(PVRShellIsKeyPressed(PVRShellKeyNameACTION1))
		bPause=!bPause;

	if(PVRShellIsKeyPressed(PVRShellKeyNameACTION2))
		fBurnAnim = 0.0f;

	/* Keyboard Animation and Automatic Shader Change over time */
	if(!bPause && (fDemoFrame > 500 || m_i32Effect == 2 && fDemoFrame > 80))
	{
		if(++m_i32Effect >= (int) g_ui32NoOfEffects)
		{
			m_i32Effect = 1;
			m_fFrame = 0.0f;
		}

		fDemoFrame = 0.0f;
		fBurnAnim  = 0.0f;
	}

	/* Change Shader Effect */

	if(PVRShellIsKeyPressed(PVRShellKeyNameRIGHT))
	{
		if(++m_i32Effect >= (int) g_ui32NoOfEffects)
			m_i32Effect = 1;

		fDemoFrame = 0.0f;
		fBurnAnim  = 0.0f;
		m_fFrame = 0.0f;
	}
	if(PVRShellIsKeyPressed(PVRShellKeyNameLEFT))
	{
		if(--m_i32Effect < 1)
			m_i32Effect = g_ui32NoOfEffects - 1;

		fDemoFrame = 0.0f;
		fBurnAnim  = 0.0f;
		m_fFrame = 0.0f;
	}

	/* Change Skybox Texture */
	if(PVRShellIsKeyPressed(PVRShellKeyNameUP))
	{
		for(i = 0; i < g_ui32NoOfEffects; ++i)
			ChangeSkyboxTo(&m_pEffects[i], m_ui32TextureIDs[4]);

		fBurnAnim = 0.0f;
	}

	if(PVRShellIsKeyPressed(PVRShellKeyNameDOWN))
	{
		for(i = 0; i < g_ui32NoOfEffects; ++i)
			ChangeSkyboxTo(&m_pEffects[i], m_ui32TextureIDs[3]);

		fBurnAnim = 0.0f;
	}

	/* Setup Shader and Shader Constants */

	/* Calculate the model view matrix turning around the balloon */
	ComputeViewMatrix();

	int location;

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	DrawSkybox();

	m_pEffects[m_i32Effect].pEffect->Activate();

	for(i = 0; i < m_Scene.nNumMeshNode; i++)
	{
		SPODNode* pNode = &m_Scene.pNode[i];

		// Gets pMesh referenced by the pNode
		SPODMesh* pMesh = &m_Scene.pMesh[pNode->nIdx];

		// Gets the node model matrix
		PVRTMat4 mWorld, mWORLDVIEW;
		mWorld = m_Scene.GetWorldMatrix(*pNode);

		mWORLDVIEW = m_mView * mWorld;

		glBindBuffer(GL_ARRAY_BUFFER, m_aiVboID[i]);

		for(j = 0; j < m_pEffects[m_i32Effect].ui32UniformCount; ++j)
		{
			switch(m_pEffects[m_i32Effect].psUniforms[j].nSemantic)
			{
				case eUsPOSITION:
				{
					glVertexAttribPointer(m_pEffects[m_i32Effect].psUniforms[j].nLocation, 3, GL_FLOAT, GL_FALSE, pMesh->sVertex.nStride, pMesh->sVertex.pData);
					glEnableVertexAttribArray(m_pEffects[m_i32Effect].psUniforms[j].nLocation);
				}
				break;
				case eUsNORMAL:
				{
					glVertexAttribPointer(m_pEffects[m_i32Effect].psUniforms[j].nLocation, 3, GL_FLOAT, GL_FALSE, pMesh->sNormals.nStride, pMesh->sNormals.pData);
					glEnableVertexAttribArray(m_pEffects[m_i32Effect].psUniforms[j].nLocation);
				}
				break;
				case eUsUV:
				{
					glVertexAttribPointer(m_pEffects[m_i32Effect].psUniforms[j].nLocation, 2, GL_FLOAT, GL_FALSE, pMesh->psUVW[0].nStride, pMesh->psUVW[0].pData);
					glEnableVertexAttribArray(m_pEffects[m_i32Effect].psUniforms[j].nLocation);
				}
				break;
				case eUsWORLDVIEWPROJECTION:
				{
					PVRTMat4 mMVP;

					/* Passes the model-view-projection matrix (MVP) to the shader to transform the vertices */
					mMVP = m_mProjection * mWORLDVIEW;
					glUniformMatrix4fv(m_pEffects[m_i32Effect].psUniforms[j].nLocation, 1, GL_FALSE, mMVP.f);
				}
				break;
				case eUsWORLDVIEW:
				{
					glUniformMatrix4fv(m_pEffects[m_i32Effect].psUniforms[j].nLocation, 1, GL_FALSE, mWORLDVIEW.f);
				}
				break;
				case eUsWORLDVIEWIT:
				{
					PVRTMat4 mWORLDVIEWI, mWORLDVIEWIT;

					mWORLDVIEWI = mWORLDVIEW.inverse();
					mWORLDVIEWIT= mWORLDVIEWI.transpose();
					
					PVRTMat3 WORLDVIEWIT = PVRTMat3(mWORLDVIEWIT);

					glUniformMatrix3fv(m_pEffects[m_i32Effect].psUniforms[j].nLocation, 1, GL_FALSE, WORLDVIEWIT.f);
				}
				break;
				case eUsVIEWIT:
				{
					PVRTMat4 mViewI, mViewIT;

					mViewI  = m_mView.inverse();
					mViewIT = mViewI.transpose();

					PVRTMat3 ViewIT = PVRTMat3(mViewIT);

					glUniformMatrix3fv(m_pEffects[m_i32Effect].psUniforms[j].nLocation, 1, GL_FALSE, ViewIT.f);
				}
				break;
				case eUsLIGHTDIRECTION:
				{
					PVRTVec4 vLightDirectionEyeSpace;
	
					// Passes the light direction in eye space to the shader
					vLightDirectionEyeSpace = m_mView * PVRTVec4(1.0,1.0,-1.0,0.0);
					glUniform3f(m_pEffects[m_i32Effect].psUniforms[j].nLocation, vLightDirectionEyeSpace.x, vLightDirectionEyeSpace.y, vLightDirectionEyeSpace.z);
				}
				break;
				case eUsTEXTURE:
				{
					// Set the sampler variable to the texture unit
					glUniform1i(m_pEffects[m_i32Effect].psUniforms[j].nLocation, m_pEffects[m_i32Effect].psUniforms[j].nIdx);
				}
				break;
			}
		}

		location = glGetUniformLocation(m_pEffects[m_i32Effect].pEffect->m_uiProgram, "myEyePos");

		if(location != -1)
			glUniform3f(location, vCameraPosition.x, vCameraPosition.y, vCameraPosition.z);

		//set animation
		location = glGetUniformLocation(m_pEffects[m_i32Effect].pEffect->m_uiProgram, "fAnim");

		if(location != -1)
			glUniform1f(location, fBurnAnim);

		location = glGetUniformLocation(m_pEffects[m_i32Effect].pEffect->m_uiProgram, "myFrame");

		if(location != -1)
			glUniform1f(location, m_fFrame);

		glEnable(GL_CULL_FACE);
		glDisable(GL_BLEND);

		if(g_bBlendShader[m_i32Effect])
		{
			glEnable(GL_BLEND);

			// Correct render order for alpha blending through culling
			// Draw Back faces
			location = glGetUniformLocation(m_pEffects[m_i32Effect].pEffect->m_uiProgram, "bBackFace");
			glUniform1i(location, 1);
			glCullFace(GL_BACK);

			DrawMesh(pMesh);

			// Draw Front faces
			glUniform1f(location, 0.0f);
		}
		else
		{
			location = glGetUniformLocation(m_pEffects[m_i32Effect].pEffect->m_uiProgram, "bBackFace");
			glUniform1i(location, 0);
			glDisable(GL_BLEND);
		}

		glCullFace(GL_FRONT);

		/* Everything should now be setup, therefore draw the mesh*/
		DrawMesh(pMesh);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		for(j = 0; j < m_pEffects[m_i32Effect].ui32UniformCount; ++j)
		{
			switch(m_pEffects[m_i32Effect].psUniforms[j].nSemantic)
			{
			case eUsPOSITION:
				{
					glDisableVertexAttribArray(m_pEffects[m_i32Effect].psUniforms[j].nLocation);
				}
				break;
			case eUsNORMAL:
				{
					glDisableVertexAttribArray(m_pEffects[m_i32Effect].psUniforms[j].nLocation);
				}
				break;
			case eUsUV:
				{
					glDisableVertexAttribArray(m_pEffects[m_i32Effect].psUniforms[j].nLocation);
				}
				break;
			}
		}
	}

	// Displays the demo name using the tools. For a detailed explanation, see the training course IntroducingPVRTools
	if(!bPause)
		m_Print3D.DisplayDefaultTitle("Skybox2", "", ePVRTPrint3DLogoIMG);
	else
		m_Print3D.DisplayDefaultTitle("Skybox2", "Paused", ePVRTPrint3DLogoIMG);

	m_Print3D.Flush();

	return true;
}

/*!****************************************************************************
 @Function		DrawMesh
 @Input			mesh		The mesh to draw
 @Description	Draws a SPODMesh after the model view matrix has been set and
				the meterial prepared.
******************************************************************************/
void OGLES2Skybox2::DrawMesh(SPODMesh* pMesh)
{
	/*
		The geometry can be exported in 4 ways:
		- Non-Indexed Triangle list
		- Indexed Triangle list
		- Non-Indexed Triangle strips
		- Indexed Triangle strips
	*/
	if(!pMesh->nNumStrips)
	{
		if(pMesh->sFaces.pData)
		{
			// Indexed Triangle list
			glDrawElements(GL_TRIANGLES, pMesh->nNumFaces*3, GL_UNSIGNED_SHORT, pMesh->sFaces.pData);
		}
		else
		{
			// Non-Indexed Triangle list
			glDrawArrays(GL_TRIANGLES, 0, pMesh->nNumFaces*3);
		}
	}
	else
	{
		if(pMesh->sFaces.pData)
		{
			// Indexed Triangle strips
			int offset = 0;
			for(int i = 0; i < (int)pMesh->nNumStrips; i++)
			{
				glDrawElements(GL_TRIANGLE_STRIP, pMesh->pnStripLength[i]+2, GL_UNSIGNED_SHORT, pMesh->sFaces.pData + offset*2);
				offset += pMesh->pnStripLength[i]+2;
			}
		}
		else
		{
			// Non-Indexed Triangle strips
			int offset = 0;
			for(int i = 0; i < (int)pMesh->nNumStrips; i++)
			{
				glDrawArrays(GL_TRIANGLE_STRIP, offset, pMesh->pnStripLength[i]+2);
				offset += pMesh->pnStripLength[i]+2;
			}
		}
	}
}

/*******************************************************************************
 * Function Name  : ComputeViewMatrix
 * Description    : Calculate the view matrix turning around the balloon
 *******************************************************************************/
void OGLES2Skybox2::ComputeViewMatrix()
{
	PVRTVec3 vFrom;

	/* Calculate the distance to balloon */
	float fDistance = fViewDistance + fViewAmplitude * (float) sin(fViewAmplitudeAngle);
	fDistance = fDistance / 5.0f;
	fViewAmplitudeAngle += 0.004f;

	/* Calculate the vertical position of the camera */
	float updown = fViewUpDownAmplitude * (float) sin(fViewUpDownAngle);
	updown = updown / 5.0f;
	fViewUpDownAngle += 0.005f;

	/* Calculate the angle of the camera around the balloon */
	vFrom.x = fDistance * (float) cos(fViewAngle);
	vFrom.y = updown;
	vFrom.z = fDistance * (float) sin(fViewAngle);
	fViewAngle += 0.003f;

	/* Compute and set the matrix */
	m_mView = PVRTMat4::LookAtRH(vFrom, vTo, vUp);

	/* Remember the camera position to draw the Skybox around it */
	vCameraPosition = vFrom;
}

void OGLES2Skybox2::ChangeSkyboxTo(SEffect *pSEffect, GLuint ui32NewSkybox)
{
	unsigned int i,i32Cnt;
	const SPVRTPFXTexture	*psTex;
	psTex = pSEffect->pEffect->GetTextureArray(i32Cnt);

	for(i = 0; i < i32Cnt; ++i)
	{
		if(strcmp(g_aszTextureNames[3], psTex[i].p) == 0)
		{
			pSEffect->pEffect->SetTexture(i, ui32NewSkybox, PVRTEX_CUBEMAP);
			return;
		}
	}
}

/*******************************************************************************
 * Function Name  : DrawSkybox
 * Description    : Draws the Skybox
 *******************************************************************************/
void OGLES2Skybox2::DrawSkybox()
{
	// Use the loaded Skybox shader program
	m_pEffects[0].pEffect->Activate();

	int iVertexUniform = 0;
	for(unsigned int j = 0; j < m_pEffects[0].ui32UniformCount; ++j)
	{
		switch(m_pEffects[0].psUniforms[j].nSemantic)
		{
			case eUsPOSITION:
			{
				iVertexUniform = j;
				glEnableVertexAttribArray(m_pEffects[0].psUniforms[j].nLocation);
			}
			break;
			case eUsWORLDVIEWPROJECTION:
			{
				PVRTMat4 mTrans, mMVP;
				mTrans = PVRTMat4::Translation(-vCameraPosition.x, -vCameraPosition.y, -vCameraPosition.z);
				
			
				mMVP = m_mProjection * mTrans * m_mView;

				/* Passes the model-view-projection matrix (MVP) to the shader to transform the vertices */
				glUniformMatrix4fv(m_pEffects[0].psUniforms[j].nLocation, 1, GL_FALSE, mMVP.f);
			}
			break;
			case eUsTEXTURE:
			{
				// Set the sampler variable to the texture unit
				glUniform1i(m_pEffects[0].psUniforms[j].nLocation, m_pEffects[0].psUniforms[j].nIdx);
			}
			break;
		}
	}

	for (int i = 0; i < 6; i++)
	{
		// Set Data Pointers
		glVertexAttribPointer(m_pEffects[0].psUniforms[iVertexUniform].nLocation, 3, GL_FLOAT, GL_FALSE, sizeof(VERTTYPE)*3, &g_SkyboxVertices[i*4*3]);

		// Draw
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}

	/* Disable States */
	glDisableVertexAttribArray(m_pEffects[0].psUniforms[iVertexUniform].nLocation);

}
/*!****************************************************************************
 @Function		NewDemo
 @Return		PVRShell*		The demo supplied by the user
 @Description	This function must be implemented by the user of the shell.
				The user should return its PVRShell object defining the
				behaviour of the application.
******************************************************************************/
PVRShell* NewDemo()
{
	return new OGLES2Skybox2();
}

/******************************************************************************
 End of file (OGLES2Skybox2.cpp)
******************************************************************************/
