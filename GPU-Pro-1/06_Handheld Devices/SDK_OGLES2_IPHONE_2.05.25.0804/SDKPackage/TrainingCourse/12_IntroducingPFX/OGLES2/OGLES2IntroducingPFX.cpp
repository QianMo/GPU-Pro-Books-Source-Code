/******************************************************************************

 @File         OGLES2IntroducingPFX.cpp

 @Title        Introducing the POD 3d file format

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Shows how to load POD files and play the animation with basic
               lighting

******************************************************************************/
#include <string.h>

#include "PVRShell.h"
#include "OGLES2Tools.h"

/******************************************************************************
 Constants
******************************************************************************/

// Camera constants used to generate the projection matrix
const float CAM_NEAR	= 75.0f;
const float CAM_FAR		= 2000.0f;

const float DEMO_FRAME_RATE	= 1.f / 30.f;

/******************************************************************************
 Content file names
******************************************************************************/

// Effect file
const char c_szPfxFile[]	= "effect.pfx";

// PVR texture files
const char c_szBaseTexFile[]		= "Basetex.pvr";
const char c_szReflectTexFile[]		= "Reflection.pvr";

// POD scene files
const char c_szSceneFile[]			= "Scene.pod";

/****************************************************************************
** Enumerations
****************************************************************************/

// Shader semantics recognised by this program
enum EUniformSemantic
{
	eUsUnknown,
	eUsPOSITION,
	eUsNORMAL,
	eUsUV,
	eUsWORLDVIEWPROJECTION,
	eUsWORLDVIEWIT,
	eUsLIGHTDIREYE,
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
	{ "WORLDVIEWIT",			eUsWORLDVIEWIT },
	{ "LIGHTDIREYE",			eUsLIGHTDIREYE },
	{ "TEXTURE",				eUsTEXTURE }
};

/*!****************************************************************************
 Class implementing the PVRShell functions.
******************************************************************************/
class OGLESIntroducingPFX : public PVRShell
{
	// Print3D class used to display text
	CPVRTPrint3D	m_Print3D;

	// 3D Model
	CPVRTModelPOD	m_Scene;

	// Projection and Model View matrices
	PVRTMat4		m_mProjection, m_mView;

	// Variables to handle the animation in a time-based manner
	int				m_iTimePrev;
	float			m_fFrame;

	// The effect file handlers
	CPVRTPFXParser	*m_pEffectParser;
	CPVRTPFXEffect	*m_pEffect;

	SPVRTPFXUniform	*m_psUniforms;
	unsigned int	m_nUniformCnt;

	// The Vertex buffer object handle array.
	GLuint			*m_aiVboID;

public:
	virtual bool InitApplication();
	virtual bool InitView();
	virtual bool ReleaseView();
	virtual bool QuitApplication();
	virtual bool RenderScene();

	void DrawMesh(SPODMesh* mesh);
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
bool OGLESIntroducingPFX::InitApplication()
{
	// Get and set the read path for content files
	CPVRTResourceFile::SetReadPath((char*)PVRShellGet(prefReadPath));

	// Load the scene
	if (m_Scene.ReadFromFile(c_szSceneFile) != PVR_SUCCESS)
	{
		PVRShellSet(prefExitMessage, "ERROR: Couldn't load the .pod file\n");
		return false;
	}

	// The cameras are stored in the file. We check it contains at least one.
	if(m_Scene.nNumCamera == 0)
	{
		PVRShellSet(prefExitMessage, "ERROR: The scene does not contain a camera\n");
		return false;
	}

	// Ensure that all meshes use an indexed triangle list
	for(unsigned int i = 0; i < m_Scene.nNumMesh; ++i)
	{
		if(m_Scene.pMesh[i].nNumStrips || !m_Scene.pMesh[i].sFaces.pData)
		{
			PVRShellSet(prefExitMessage, "ERROR: The meshes in the scene should use an indexed triangle list\n");
			return false;
		}
	}

	// Initialize variables used for the animation
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
bool OGLESIntroducingPFX::QuitApplication()
{
	// Frees the memory allocated for the scene
	m_Scene.Destroy();

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
bool OGLESIntroducingPFX::InitView()
{
	/*
		Initialize Print3D
	*/
	bool bRotate = PVRShellGet(prefIsRotated) && PVRShellGet(prefFullScreen);

	if(m_Print3D.SetTextures(0,PVRShellGet(prefWidth),PVRShellGet(prefHeight), bRotate) != PVR_SUCCESS)
	{
		PVRShellSet(prefExitMessage, "ERROR: Cannot initialise Print3D\n");
		return false;
	}

	// Sets the clear color
	glClearColor(0.6f, 0.8f, 1.0f, 1.0f);

	// Enables depth test using the z-buffer
	glEnable(GL_DEPTH_TEST);

	/*
		Loads the light direction from the scene.
	*/
	// We check the scene contains at least one
	if (m_Scene.nNumLight == 0)
	{
		PVRShellSet(prefExitMessage, "ERROR: The scene does not contain a light\n");
		return false;
	}

	/*
		Load the effect file
	*/
	{
		unsigned int	nUnknownUniformCount;
		CPVRTString	error;

		// Parse the file
		m_pEffectParser = new CPVRTPFXParser;
		if(m_pEffectParser->ParseFromFile(c_szPfxFile, &error) != PVR_SUCCESS)
		{
			PVRShellSet(prefExitMessage, error.c_str());
			return false;
		}

		// Load an effect from the file
		m_pEffect = new CPVRTPFXEffect();
		if(m_pEffect->Load(*m_pEffectParser, "Effect", c_szPfxFile, &error) != PVR_SUCCESS)
		{
			PVRShellSet(prefExitMessage, error.c_str());
			return false;
		}

		// Generate uniform array
		if(m_pEffect->BuildUniformTable(
			&m_psUniforms, &m_nUniformCnt, &nUnknownUniformCount,
			c_psUniformSemantics, sizeof(c_psUniformSemantics) / sizeof(*c_psUniformSemantics),
			&error) != PVR_SUCCESS)
		{
			PVRShellOutputDebug(error.c_str());
			return false;
		}
		if(nUnknownUniformCount)
		{
			PVRShellOutputDebug(error.c_str());
			PVRShellOutputDebug("Unknown uniform semantic count: %d\n", nUnknownUniformCount);
		}
	}

	/*
		Loads the textures.
		For a more detailed explanation, see Texturing and IntroducingPVRTools
	*/
	{
		const SPVRTPFXTexture	*psTex;
		unsigned int			nCnt, i;
		GLuint					ui;

		psTex = m_pEffect->GetTextureArray(nCnt);

		for(i = 0; i < nCnt; ++i)
		{
			if(strcmp(psTex[i].p, "Reflection.pvr") == 0)
			{
				if(PVRTTextureLoadFromPVR(c_szReflectTexFile, &ui) != PVR_SUCCESS)
				{
					PVRShellSet(prefExitMessage, "ERROR: Cannot load the texture\n");
					return false;
				}
			
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

				m_pEffect->SetTexture(i, ui);
			}
			else if (strcmp(psTex[i].p, "Basetex.pvr") == 0)
			{
				if(PVRTTextureLoadFromPVR(c_szBaseTexFile, &ui) != PVR_SUCCESS)
				{
					PVRShellSet(prefExitMessage, "ERROR: Cannot load the texture\n");
					return false;
				}
			
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				
				m_pEffect->SetTexture(i, ui);
			}
			else
			{
				PVRShellOutputDebug("Warning: effect file requested unrecognised texture: \"%s\"\n", psTex[i].p);
				m_pEffect->SetTexture(i, 0);
			}
		}
	}

	// Create buffer objects.
	m_aiVboID = new GLuint[m_Scene.nNumMeshNode];
	glGenBuffers(m_Scene.nNumMeshNode, m_aiVboID);
	for(int i = 0; i < (int)m_Scene.nNumMeshNode ; i++)
	{
		SPODNode* pNode = &m_Scene.pNode[i];

		// Gets pMesh referenced by the pNode
		SPODMesh* pMesh = &m_Scene.pMesh[pNode->nIdx];

		// Generate a vertex buffer and set the interleaved vertex data.
		glBindBuffer(GL_ARRAY_BUFFER, m_aiVboID[i]);
		glBufferData(GL_ARRAY_BUFFER, pMesh->sVertex.nStride*pMesh->nNumVertex, pMesh->pInterleaved, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	return true;
}

/*!****************************************************************************
 @Function		ReleaseView
 @Return		bool		true if no error occured
 @Description	Code in ReleaseView() will be called by PVRShell when the
				application quits or before a change in the rendering context.
******************************************************************************/
bool OGLESIntroducingPFX::ReleaseView()
{
	// Release textures
	{
		const SPVRTPFXTexture	*psTex;
		unsigned int			nCnt, i;

		psTex = m_pEffect->GetTextureArray(nCnt);

		for(i = 0; i < nCnt; ++i)
		{
			glDeleteTextures(1, &psTex[i].ui);
		}
	}

	// Release the effect[s] then the parser
	delete m_pEffect;
	delete m_pEffectParser;
	FREE(m_psUniforms);

	// Release Print3D Textures
	m_Print3D.ReleaseTextures();

	// Release Vertex buffer objects.
	glDeleteBuffers(m_Scene.nNumMeshNode, m_aiVboID);
	delete[] m_aiVboID;

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
bool OGLESIntroducingPFX::RenderScene()
{
	// Clears the color and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Use the loaded effect
	m_pEffect->Activate();

	/*
		Calculates the frame number to animate in a time-based manner.
		Uses the shell function PVRShellGetTime() to get the time in milliseconds.
	*/
	int iTime = PVRShellGetTime();
	int iDeltaTime = iTime - m_iTimePrev;
	m_iTimePrev	= iTime;
	m_fFrame	+= (float)iDeltaTime * DEMO_FRAME_RATE;
	if (m_fFrame > m_Scene.nNumFrame-1)
		m_fFrame = 0;

	// Sets the scene animation to this frame
	m_Scene.SetFrame(m_fFrame);

	{
		PVRTVec3	vFrom, vTo, vUp;
		VERTTYPE	fFOV;
		vUp.x = 0.0f;
		vUp.y = 1.0f;
		vUp.z = 0.0f;

		// We can get the camera position, target and field of view (fov) with GetCameraPos()
		fFOV = m_Scene.GetCameraPos(vFrom, vTo, 0) * 0.4f;

		/*
			We can build the world view matrix from the camera position, target and an up vector.
			For this we use PVRTMat4LookAtRH().
		*/
		m_mView = PVRTMat4::LookAtRH(vFrom, vTo, vUp);

		// Calculates the projection matrix
		bool bRotate = PVRShellGet(prefIsRotated) && PVRShellGet(prefFullScreen);
		m_mProjection = PVRTMat4::PerspectiveFovRH(fFOV, (float)PVRShellGet(prefWidth)/(float)PVRShellGet(prefHeight), CAM_NEAR, CAM_FAR, PVRTMat4::OGL, bRotate);
	}

	/*
		A scene is composed of nodes. There are 3 types of nodes:
		- MeshNodes :
			references a mesh in the pMesh[].
			These nodes are at the beginning of the pNode[] array.
			And there are nNumMeshNode number of them.
			This way the .pod format can instantiate several times the same mesh
			with different attributes.
		- lights
		- cameras
		To draw a scene, you must go through all the MeshNodes and draw the referenced meshes.
	*/
	for (int i=0; i<(int)m_Scene.nNumMeshNode; i++)
	{
		SPODNode* pNode = &m_Scene.pNode[i];

		// Gets pMesh referenced by the pNode
		SPODMesh* pMesh = &m_Scene.pMesh[pNode->nIdx];

		glBindBuffer(GL_ARRAY_BUFFER, m_aiVboID[i]);

		// Gets the node model matrix
		PVRTMat4 mWorld;
		mWorld = m_Scene.GetWorldMatrix(*pNode);

		PVRTMat4 mWorldView;
		mWorldView = m_mView * mWorld;

		for(unsigned int j = 0; j < m_nUniformCnt; ++j)
		{
			switch(m_psUniforms[j].nSemantic)
			{
			case eUsPOSITION:
				{
					glVertexAttribPointer(m_psUniforms[j].nLocation, 3, GL_FLOAT, GL_FALSE, pMesh->sVertex.nStride, pMesh->sVertex.pData);
					glEnableVertexAttribArray(m_psUniforms[j].nLocation);
				}
				break;
			case eUsNORMAL:
				{
					glVertexAttribPointer(m_psUniforms[j].nLocation, 3, GL_FLOAT, GL_FALSE, pMesh->sNormals.nStride, pMesh->sNormals.pData);
					glEnableVertexAttribArray(m_psUniforms[j].nLocation);
				}
				break;
			case eUsUV:
				{
					glVertexAttribPointer(m_psUniforms[j].nLocation, 2, GL_FLOAT, GL_FALSE, pMesh->psUVW[0].nStride, pMesh->psUVW[0].pData);
					glEnableVertexAttribArray(m_psUniforms[j].nLocation);
				}
				break;
			case eUsWORLDVIEWPROJECTION:
				{
					PVRTMat4 mWVP;

					/* Passes the world-view-projection matrix (WVP) to the shader to transform the vertices */
					mWVP = m_mProjection * mWorldView;
					glUniformMatrix4fv(m_psUniforms[j].nLocation, 1, GL_FALSE, mWVP.f);
				}
				break;
			case eUsWORLDVIEWIT:
				{
					PVRTMat4 mWorldViewI, mWorldViewIT;

					/* Passes the inverse transpose of the world-view matrix to the shader to transform the normals */
					mWorldViewI  = mWorldView.inverse();
					mWorldViewIT = mWorldViewI.transpose();

					PVRTMat3 WorldViewIT = PVRTMat3(mWorldViewIT);

					glUniformMatrix3fv(m_psUniforms[j].nLocation, 1, GL_FALSE, WorldViewIT.f);
				}
				break;
			case eUsLIGHTDIREYE:
				{
					// Reads the light direction from the scene.
					PVRTVec4 vLightDirection;
					PVRTVec3 vPos;
					vLightDirection = m_Scene.GetLightDirection(0);

					vLightDirection.x = -vLightDirection.x;
					vLightDirection.y = -vLightDirection.y;
					vLightDirection.z = -vLightDirection.z;

					/*
						Sets the w component to 0, so when passing it to glLight(), it is
						considered as a directional light (as opposed to a spot light).
					*/
					vLightDirection.w = 0;

					// Passes the light direction in eye space to the shader
					PVRTVec4 vLightDirectionEyeSpace;
					vLightDirectionEyeSpace = m_mView * vLightDirection;

					glUniform3f(m_psUniforms[j].nLocation, vLightDirectionEyeSpace.x, vLightDirectionEyeSpace.y, vLightDirectionEyeSpace.z);
				}
				break;
			case eUsTEXTURE:
				{
					// Set the sampler variable to the texture unit
					glUniform1i(m_psUniforms[j].nLocation, m_psUniforms[j].nIdx);
				}
				break;
			}
		}

		/*
			Now that the model-view matrix is set and the materials ready,
			call another function to actually draw the mesh.
		*/
		DrawMesh(pMesh);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		for(unsigned int j = 0; j < m_nUniformCnt; ++j)
		{
			switch(m_psUniforms[j].nSemantic)
			{
			case eUsPOSITION:
				{
					glDisableVertexAttribArray(m_psUniforms[j].nLocation);
				}
				break;
			case eUsNORMAL:
				{
					glDisableVertexAttribArray(m_psUniforms[j].nLocation);
				}
				break;
			case eUsUV:
				{
					glDisableVertexAttribArray(m_psUniforms[j].nLocation);
				}
				break;
			}
		}
	}

	// Displays the demo name using the tools. For a detailed explanation, see the training course IntroducingPVRTools
	m_Print3D.DisplayDefaultTitle("IntroducingPFX", "", ePVRTPrint3DLogoIMG);
	m_Print3D.Flush();

	return true;
}

/*!****************************************************************************
 @Function		DrawMesh
 @Input			mesh		The mesh to draw
 @Description	Draws a SPODMesh after the model view matrix has been set and
				the meterial prepared.
******************************************************************************/
void OGLESIntroducingPFX::DrawMesh(SPODMesh* pMesh)
{
	/*
		Submit geometry as an indexed triangle list.

		"IntroducingPOD" demonstrates how to support all possible formats
		(e.g. non-indexed, or stripped).
	*/
	glDrawElements(GL_TRIANGLES, pMesh->nNumFaces*3, GL_UNSIGNED_SHORT, pMesh->sFaces.pData);
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
	return new OGLESIntroducingPFX();
}

/******************************************************************************
 End of file (OGLES2IntroducingPFX.cpp)
******************************************************************************/
