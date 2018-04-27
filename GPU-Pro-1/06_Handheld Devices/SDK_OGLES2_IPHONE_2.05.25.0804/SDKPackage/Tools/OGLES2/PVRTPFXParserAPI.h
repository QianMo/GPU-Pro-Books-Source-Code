/******************************************************************************

 @File         PVRTPFXParserAPI.h

 @Title        OGLES2/PVRTPFXParserAPI.h

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Windows + Linux

 @Description  Declaration of PFX file parser

******************************************************************************/
#ifndef _PVRTPFXPARSERAPI_H_
#define _PVRTPFXPARSERAPI_H_

#include "../PVRTError.h"

/****************************************************************************
** Structures
****************************************************************************/

/*! Application supplies an array of these so PVRTPFX can translate strings to numbers */
struct SPVRTPFXUniformSemantic
{
	const char		*p;	// String containing semantic
	unsigned int	n;	// Application-defined semantic value
};

/*! PVRTPFX returns an array of these to indicate GL locations & semantics to the application */
struct SPVRTPFXUniform
{
	unsigned int	nLocation;	// GL location of the Uniform
	unsigned int	nSemantic;	// Application-defined semantic value
	unsigned int	nIdx;		// Index; for example two semantics might be LIGHTPOSITION0 and LIGHTPOSITION1
};

/*! An array of these is gained from PVRTPFX so the application can fill in the texture handles*/
struct SPVRTPFXTexture
{
	const char	*p;		// texture FileName
	GLuint		ui;		// Loaded texture handle
};

/*!**************************************************************************
@Class CPVRTPFXEffect
@Brief PFX effect
****************************************************************************/
class CPVRTPFXEffect
{
public:
	SPVRTContext	*m_psContext;
	CPVRTPFXParser	*m_pParser;
	unsigned int	m_nEffect;

	GLuint			m_uiProgram;		// Loaded program
	unsigned int	*m_pnTextureIdx;	// Array of indices into the texture array

	SPVRTPFXTexture	*m_psTextures;		// Array of loaded textures

public:
	/*!***************************************************************************
	@Function			CPVRTPFXEffect Blank Constructor
	@Description		Sets the context and initialises the member variables to zero.
	*****************************************************************************/
	CPVRTPFXEffect();

	/*!***************************************************************************
	@Function			CPVRTPFXEffect Constructor
	@Description		Sets the context and initialises the member variables to zero.
	*****************************************************************************/
	CPVRTPFXEffect(SPVRTContext &sContext);

	/*!***************************************************************************
	@Function			CPVRTPFXEffect Destructor
	@Description		Calls Destroy().
	*****************************************************************************/
	~CPVRTPFXEffect();

	/*!***************************************************************************
	@Function			Load
	@Input				src					PFX Parser Object
	@Input				pszEffect			Effect name
	@Input				pszFileName			Effect file name
	@Output				pReturnError		Error string
	@Returns			EPVRTError			PVR_SUCCESS if load succeeded
	@Description		Loads the specified effect from the CPVRTPFXParser object.
						Compiles and links the shaders. Initialises texture data.
	*****************************************************************************/
	EPVRTError Load(CPVRTPFXParser &src, const char * const pszEffect, const char * const pszFileName, CPVRTString *pReturnError);

	/*!***************************************************************************
	@Function			Destroy
	@Description		Deletes the gl program object and texture data.
	*****************************************************************************/
	void Destroy();

	/*!***************************************************************************
	@Function			Activate
	@Returns			PVR_SUCCESS if activate succeeded
	@Description		Selects the gl program object and binds the textures.
	*****************************************************************************/
	EPVRTError Activate();

	/*!***************************************************************************
	@Function			BuildUniformTable
	@Output				ppsUniforms					pointer to uniform data array
	@Output				pnUniformCount				pointer to number of uniforms
	@Output				pnUnknownUniformCount		pointer to number of unknown uniforms
	@Input				psUniformSemantics			pointer to uniform semantic data array
	@Input				nSemantics					number of uniform semantics
	@Output				pReturnError				error string
	@Returns			EPVRTError					PVR_SUCCESS if succeeded
	@Description		Builds the uniform table from the semantics.
	*****************************************************************************/
	EPVRTError BuildUniformTable(
		SPVRTPFXUniform					** const ppsUniforms,
		unsigned int					* const pnUniformCount,
		unsigned int					* const pnUnknownUniformCount,
		const SPVRTPFXUniformSemantic	* const psUniformSemantics,
		const unsigned int				nSemantics,
		CPVRTString					*pReturnError);

	/*!***************************************************************************
	@Function			GetTextureArray
	@Output				nCount					number of textures
	@Returns			SPVRTPFXTexture*		pointer to the texture data array
	@Description		Gets the texture data array.
	*****************************************************************************/
	const SPVRTPFXTexture *GetTextureArray(unsigned int &nCount) const;

	/*!***************************************************************************
	@Function			SetTexture
	@Input				nIdx				texture number
	@Input				ui					opengl texture handle
	@Input				u32flags			texture flags
	@Description		Sets the textrue and applys the filtering.
	*****************************************************************************/
	void SetTexture(const unsigned int nIdx, const GLuint ui, const unsigned int u32flags=0);

	/*!***************************************************************************
	@Function			SetDefaultSemanticValue
	@Input				pszName				name of uniform
	@Input				psDefaultValue      pointer to default value
	@Description		Sets the dafault value for the uniform semantic.
	*****************************************************************************/
	void SetDefaultUniformValue(const char *const pszName, const SPVRTSemanticDefaultData *psDefaultValue);

};

#endif /* _PVRTPFXPARSERAPI_H_ */

/*****************************************************************************
 End of file (PVRTPFXParserAPI.h)
*****************************************************************************/
