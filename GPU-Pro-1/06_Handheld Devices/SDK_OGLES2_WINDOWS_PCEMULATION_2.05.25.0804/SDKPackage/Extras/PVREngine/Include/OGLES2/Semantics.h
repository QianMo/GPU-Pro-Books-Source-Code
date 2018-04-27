/******************************************************************************

 @File         Semantics.h

 @Title        PVREngine main header file for OGLES2 API

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent/OGLES2

 @Description  Uniform semantics and other enums for the PVREngine with OGLES2

******************************************************************************/
#ifndef _SEMANTICS_H_
#define _SEMANTICS_H_

namespace pvrengine
{
	/*! Shader semantics recognised by this program */
	enum EUniformSemantic
	{
		// Attributes
		eUsUnknown,
		eUsPosition,
		eUsNormal,
		eUsTangent,
		eUsBinormal,
		eUsUV,
		eUsBoneIndex,
		eUsBoneWeight,

		// other uniforms
		eUsWORLD,
		eUsWORLDI,
		eUsWORLDIT,
		eUsVIEW,
		eUsVIEWI,
		eUsVIEWIT,
		eUsPROJECTION,
		eUsPROJECTIONI,
		eUsPROJECTIONIT,
		eUsWORLDVIEW,
		eUsWORLDVIEWI,
		eUsWORLDVIEWIT,
		eUsWORLDVIEWPROJECTION,
		eUsWORLDVIEWPROJECTIONI,
		eUsWORLDVIEWPROJECTIONIT,
		eUsVIEWPROJECTION,
		eUsVIEWPROJECTIONI,
		eUsVIEWPROJECTIONIT,
		eUsOBJECT,
		eUsOBJECTI,
		eUsOBJECTIT,

		eUsBONECOUNT,
		eUsBONEMATRIXARRAY,
		eUsBONEMATRIXARRAYI,
		eUsBONEMATRIXARRAYIT,

		eUsMATERIALOPACITY,
		eUsMATERIALSHININESS,
		eUsMATERIALCOLORAMBIENT,
		eUsMATERIALCOLORDIFFUSE,
		eUsMATERIALCOLORSPECULAR,
		eUsLIGHTCOLOR,
		eUsLIGHTPOSMODEL,
		eUsLIGHTPOSWORLD,
		eUsLIGHTPOSEYE,
		eUsLIGHTDIRMODEL,
		eUsLIGHTDIRWORLD,
		eUsLIGHTDIREYE,
		eUsEYEPOSMODEL,
		eUsEYEPOSWORLD,
		eUsTEXTURE,
		eUsANIMATION,
		eUsGEOMETRYCOUNTER,
		eUsVIEWPORTPIXELSIZE,
		eUsVIEWPORTCLIPPING,
		eUsTIME,
		eUsLASTTIME,
		eUsELAPSEDTIME,
		eUsBOUNDINGCENTER,
		eUsBOUNDINGSPHERERADIUS,
		eUsBOUNDINGBOXSIZE,
		eUsBOUNDINGBOXMIN,
		eUsBOUNDINGBOXMAX,
		eUsRANDOM,
		eUsMOUSEPOSITION,
		eUsLEFTMOUSEDOWN,
		eUsRIGHTMOUSEDOWN,

		eNumSemantics
	};

	/*! shader semantics that need only be calculated once per frame */
	/*! number of frame uniforms */
	const int i32NUMFRAMEUNIFORMS =  24;
	const static EUniformSemantic eFrameUniforms[i32NUMFRAMEUNIFORMS+1] =
	{
		eUsVIEW,
			eUsVIEWI,
			eUsVIEWIT,
			eUsPROJECTION,
			eUsPROJECTIONI,
			eUsPROJECTIONIT,
			eUsVIEWPROJECTION,
			eUsVIEWPROJECTIONI,
			eUsVIEWPROJECTIONIT,
			eUsLIGHTCOLOR,
			eUsLIGHTPOSWORLD,
			eUsLIGHTPOSEYE,
			eUsLIGHTDIRWORLD,
			eUsLIGHTDIREYE,
			eUsEYEPOSWORLD,
			eUsANIMATION,
			eUsVIEWPORTPIXELSIZE,
			eUsVIEWPORTCLIPPING,
			eUsTIME,
			eUsLASTTIME,
			eUsELAPSEDTIME,
			eUsMOUSEPOSITION,
			eUsLEFTMOUSEDOWN,
			eUsRIGHTMOUSEDOWN,
			eUsUnknown
	};

	/*! Shader semantics that need only be calculated once per material */
	const static EUniformSemantic eMaterialUniforms[] =
	{
		eUsMATERIALOPACITY,
			eUsMATERIALSHININESS,
			eUsMATERIALCOLORAMBIENT,
			eUsMATERIALCOLORDIFFUSE,
			eUsMATERIALCOLORSPECULAR,
			eUsTEXTURE,
			eUsUnknown
	};

	/*! Shader semantics that need to be calculated for each mesh */
	const static EUniformSemantic eMeshUniforms[] =
	{
		eUsPosition,
			eUsNormal,
			eUsTangent,
			eUsBinormal,
			eUsUV,
			eUsBoneIndex,
			eUsBoneWeight,
			eUsWORLD,
			eUsWORLDI,
			eUsWORLDIT,
			eUsWORLDVIEW,
			eUsWORLDVIEWI,
			eUsWORLDVIEWIT,
			eUsWORLDVIEWPROJECTION,
			eUsWORLDVIEWPROJECTIONI,
			eUsWORLDVIEWPROJECTIONIT,
			eUsLIGHTPOSMODEL,
			eUsLIGHTDIRMODEL,
			eUsEYEPOSWORLD,
			eUsGEOMETRYCOUNTER,
			eUsBOUNDINGCENTER,
			eUsBOUNDINGSPHERERADIUS,
			eUsBOUNDINGBOXSIZE,
			eUsBOUNDINGBOXMIN,
			eUsBOUNDINGBOXMAX,
			eUsRANDOM,
			eUsEYEPOSMODEL,
			eUsUnknown
	};

	/*! Shader semantics that need to be calculated for each bone batch */
	const static EUniformSemantic eSkinningUniforms[] =
	{
		eUsBONECOUNT,
			eUsBONEMATRIXARRAY,
			eUsBONEMATRIXARRAYI,
			eUsBONEMATRIXARRAYIT,
			eUsUnknown
	};

	const static SPVRTPFXUniformSemantic c_psUniformSemantics[] =
	{
		{ "POSITION",				eUsPosition },
		{ "NORMAL",					eUsNormal },
		{ "TANGENT",				eUsTangent },
		{ "BINORMAL",				eUsBinormal },
		{ "UV",						eUsUV },
		{ "BONEINDEX",				eUsBoneIndex, },
		{ "BONEWEIGHT",				eUsBoneWeight, },

		{ "WORLD",					eUsWORLD },
		{ "WORLDI",					eUsWORLDI },
		{ "WORLDIT",				eUsWORLDIT },
		{ "VIEW",					eUsVIEW },
		{ "VIEWI",					eUsVIEWI },
		{ "VIEWIT",					eUsVIEWIT },
		{ "PROJECTION",				eUsPROJECTION },
		{ "PROJECTIONI",			eUsPROJECTIONI },
		{ "PROJECTIONIT",			eUsPROJECTIONIT },
		{ "WORLDVIEW",				eUsWORLDVIEW },
		{ "WORLDVIEWI",				eUsWORLDVIEWI },
		{ "WORLDVIEWIT",			eUsWORLDVIEWIT },
		{ "WORLDVIEWPROJECTION",	eUsWORLDVIEWPROJECTION },
		{ "WORLDVIEWPROJECTIONI",	eUsWORLDVIEWPROJECTIONI },
		{ "WORLDVIEWPROJECTIONIT",	eUsWORLDVIEWPROJECTIONIT },
		{ "VIEWPROJECTION",			eUsVIEWPROJECTION, },
		{ "VIEWPROJECTIONI",		eUsVIEWPROJECTIONI, },
		{ "VIEWPROJECTIONIT",		eUsVIEWPROJECTIONIT, },
		{ "OBJECT",					eUsOBJECT, },
		{ "OBJECTI",				eUsOBJECTI, },
		{ "OBJECTIT",				eUsOBJECTIT, },

		{ "MATERIALOPACITY",		eUsMATERIALOPACITY },
		{ "MATERIALSHININESS",		eUsMATERIALSHININESS },
		{ "MATERIALCOLORAMBIENT",	eUsMATERIALCOLORAMBIENT },
		{ "MATERIALCOLORDIFFUSE",	eUsMATERIALCOLORDIFFUSE },
		{ "MATERIALCOLORSPECULAR",	eUsMATERIALCOLORSPECULAR },

		{ "BONECOUNT",				eUsBONECOUNT, },
		{ "BONEMATRIXARRAY",		eUsBONEMATRIXARRAY, },
		{ "BONEMATRIXARRAYI",		eUsBONEMATRIXARRAYI, },
		{ "BONEMATRIXARRAYIT",		eUsBONEMATRIXARRAYIT, },

		{ "LIGHTCOLOR",				eUsLIGHTCOLOR },
		{ "LIGHTPOSMODEL",			eUsLIGHTPOSMODEL },
		{ "LIGHTPOSWORLD",			eUsLIGHTPOSWORLD },
		{ "LIGHTPOSEYE",			eUsLIGHTPOSEYE },
		{ "LIGHTDIRMODEL",			eUsLIGHTDIRMODEL },
		{ "LIGHTDIRWORLD",			eUsLIGHTDIRWORLD },
		{ "LIGHTDIREYE",			eUsLIGHTDIREYE },

		{ "EYEPOSMODEL",			eUsEYEPOSMODEL },
		{ "EYEPOSWORLD",			eUsEYEPOSWORLD },
		{ "TEXTURE",				eUsTEXTURE },
		{ "ANIMATION",				eUsANIMATION },
		{ "GEOMETRYCOUNTER",		eUsGEOMETRYCOUNTER },

		{ "VIEWPORTPIXELSIZE",		eUsVIEWPORTPIXELSIZE},
		{ "VIEWPORTCLIPPING",		eUsVIEWPORTCLIPPING},
		{ "TIME",					eUsTIME	},
		{ "LASTTIME",				eUsLASTTIME	},
		{ "ELAPSEDTIME",			eUsELAPSEDTIME		},
		{ "BOUNDINGCENTER",			eUsBOUNDINGCENTER	},
		{ "BOUNDINGSPHERERADIUS",	eUsBOUNDINGSPHERERADIUS},
		{ "BOUNDINGBOXSIZE",		eUsBOUNDINGBOXSIZE},
		{ "BOUNDINGBOXMIN",			eUsBOUNDINGBOXMIN},
		{ "BOUNDINGBOXMAX",			eUsBOUNDINGBOXMAX	},
		{ "RANDOM",					eUsRANDOM		},
		{ "MOUSEPOSITION",			eUsMOUSEPOSITION},
		{ "LEFTMOUSEDOWN",			eUsLEFTMOUSEDOWN	},
		{ "RIGHTMOUSEDOWN",			eUsRIGHTMOUSEDOWN	},
	}; /*! Map from PFX string to semantic enum - these are the same as used by PVRShaman*/

}

#endif // _SEMANTICS_H_

/******************************************************************************
End of file (Semantics.cpp)
******************************************************************************/
