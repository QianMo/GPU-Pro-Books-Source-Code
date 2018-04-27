/******************************************************************************

 @File         PVRESParser.h

 @Title        A simple script parser for use with PVREngine

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent - OGL/OGLES/OGLES2 Specific at the moment

 @Description  A simple script parser for use with PVREngine

******************************************************************************/
#ifndef PVRESPARSER_H
#define PVRESPARSER_H

/******************************************************************************
Includes
******************************************************************************/

#include "PVREngine.h"
#include "PVRES.h"


/*!****************************************************************************
Class
******************************************************************************/
/*!***************************************************************************
* @Class PVRESParser
* @Brief 	A simple script parser for use with PVREngine.
* @Description 	A simple script parser for use with PVREngine.
*****************************************************************************/
class PVRESParser
{

public:
	/*!***************************************************************************
	@Function			PVRESParser
	@Description		blank constructor.
	*****************************************************************************/
	PVRESParser();

	/*!***************************************************************************
	@Function			~PVRESParser
	@Description		destructor.
	*****************************************************************************/
	~PVRESParser();

	/*!***************************************************************************
	@Function			Parse
	@Return				A PVRES data object
	@Description		Parses the previously set script file. I.E. reads the
	file, all tags and stores all values in a more friendly data structure
	*****************************************************************************/
	PVRES Parse();

	/*!***************************************************************************
	@Function			setScriptFileName
	@Input				strScriptFileName script file path
	@Description		sets the script file path.
	*****************************************************************************/
	void setScriptFileName(const CPVRTString& strScriptFileName);

	/*!***************************************************************************
	@Function			setPODFileName
	@Input				strPODFileName POD file path
	@Description		sets the POD file path.
	*****************************************************************************/
	void setPODFileName(const CPVRTString& strPODFileName);

	/*!***************************************************************************
	@Function			setScript
	@Input				strScriptFileName script
	@Description		sets the actual script string.
	*****************************************************************************/
	void setScript(const CPVRTString& strScriptFileName);

	/*!***************************************************************************
	@Function			clearScriptFileName
	@Description		clears the script file path.
	*****************************************************************************/
	void clearScriptFileName();

	/*!***************************************************************************
	@Function			clearScript
	@Description		clears the script string.
	*****************************************************************************/
	void clearScript();

	/*!***************************************************************************
	@Function			getError
	@Return				error string
	@Description		retrieves a reported script parsing error.
	*****************************************************************************/
	CPVRTString getError() { return m_strError;}

protected:
	CPVRTString m_strScript;			/*! the script to be parsed */
	bool m_bScriptFileSpecified;		/*! has a valid script file been set */
	PVRES *m_pcPVRES;					/*! the data structure to be filled */
	CPVRTString m_strError;			/*! error string */

};

#endif // PVRESPARSER_H

/******************************************************************************
End of file (PVRESParser.h)
******************************************************************************/
