/******************************************************************************

 @File         ConsoleLog.h

 @Title        Console Log

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  A Class for storing output from the PVREngine and apps that use
               the engine

******************************************************************************/
#ifndef CONSOLELOG_H
#define CONSOLELOG_H

/******************************************************************************
Includes
******************************************************************************/

#include "../PVRTools.h"
#include "../PVRTSingleton.h"
#include "dynamicArray.h"

namespace pvrengine
{
	/*!***************************************************************************
	 * @Class ConsoleLog
	 * @Brief A Class for storing output from the PVREngine and apps that use the engine.
 	 * @Description A Class for storing output from the PVREngine and apps that use the engine.
 	 *****************************************************************************/
	class ConsoleLog : public CPVRTSingleton<ConsoleLog>
	{
	public:
		/*!***************************************************************************
		@Function			ConsoleLog
		@Description		Constructor.
		*****************************************************************************/
		ConsoleLog();

		/*!***************************************************************************
		@Function			~ConsoleLog
		@Description		Destructor.
		*****************************************************************************/
		~ConsoleLog();

		/*!***************************************************************************
		@Function			log
		@Input				pszFormat	a formatted string
		@Description		main function to actually add to the log. Uses same format
		as printf.
		*****************************************************************************/
		void log(const char* pszFormat, ...);

		/*!***************************************************************************
		@Function			setOutputFile
		@Input				strLogFile	logfile path
		@Return				bool success
		@Description		sets the path to the output file.
		*****************************************************************************/
		bool setOutputFile(const CPVRTString strLogFile);

		/*!***************************************************************************
		@Function			setStraightToFile
		@Input				bStraightToFile bool
		@Description		sets whether the log writes straight to its logfile.
		*****************************************************************************/
		void setStraightToFile(bool bStraightToFile);

		/*!***************************************************************************
		@Function			getStraightToFile
		@Return				bool success
		@Description		gets whether the log is writing straight to file or not
		*****************************************************************************/
		bool getStraightToFile();

		/*!***************************************************************************
		@Function			getLog
		@Return				dynamic array of log's contents
		@Description		retrieves the entire existing log
		*****************************************************************************/
		dynamicArray<CPVRTString> getLog();

		/*!***************************************************************************
		@Function			getLastLogLine
		@Return			CPVRTString of last line
		@Description		retrieves the last line from log
		*****************************************************************************/
		CPVRTString getLastLogLine();

		/*!***************************************************************************
		@Function			writeToFile
		@Return				bool success
		@Description		writes the entire current log to the specified file
		overwriting the existing file
		*****************************************************************************/
		bool writeToFile();

		/*!***************************************************************************
		@Function			appendToFile
		@Return				bool success
		@Description		appends the entire current log to the specified file
		*****************************************************************************/
		bool appendToFile();

		/*!***************************************************************************
		@Function			clearLog
		@Description		clears the log in memory
		*****************************************************************************/
		void clearLog();

	private:
		/****************************************************************************
		** Variables
		****************************************************************************/
		/*! current location of log file */
		CPVRTString	m_strLogFile;
		/*! array containing the actual log */
		dynamicArray<CPVRTString>	m_daLog;
		/*! should the log be written straight to file or just kept in memory */
		bool	m_bStraightToFile;
		/*! actual file pointer to log file */
		FILE*	m_LogFile;

	};

}

#endif // CONSOLELOG_H

/******************************************************************************
End of file (ConsoleLog.h)
******************************************************************************/
