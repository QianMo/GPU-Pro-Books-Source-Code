/******************************************************************************

 @File         ConsoleLog.cpp

 @Title        Console Log

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  A Class for storing output from the PVREngine and apps that use
               the engine. Main features are the ability to output a file of the
               log's contents, whether this is written to constantly or not. The
               primary function is log() which takes formatted input and logs to
               the console.

******************************************************************************/
#include "ConsoleLog.h"
#include <string.h>
#include <stdarg.h>

namespace pvrengine
{
	

	/******************************************************************************/

	ConsoleLog::ConsoleLog():
		m_bStraightToFile(false),
			m_LogFile(NULL)
	{}

	/******************************************************************************/

	ConsoleLog::~ConsoleLog()
	{
		if(m_LogFile)
		{
			fflush(m_LogFile);
			fclose(m_LogFile);
		}
	}

	/******************************************************************************/

	// TODO: make this deal with longer lines than 4096 characters
	// or at least work in some kind of more sensible way
	void ConsoleLog::log(const char* pszFormat, ...)
	{
		va_list args;
		va_start(args,pszFormat);
		char pszString[4096];
		vsprintf(pszString,pszFormat,args);
		va_end(args);
		m_daLog.append(CPVRTString(pszString));
		if(m_LogFile)
		{
			fwrite(pszString,1,strlen(pszString),m_LogFile);
			fflush(m_LogFile);
		}
		PVRTERROR_OUTPUT_DEBUG(pszString);
	}

	/******************************************************************************/

	bool ConsoleLog::setOutputFile(const CPVRTString strLogFile)
	{
		bool bClose=true;
		m_strLogFile = strLogFile;
		if(m_LogFile)
		{	// writing straight out set
			bClose=false;
			fclose(m_LogFile);
			m_LogFile=NULL;
		}
		// open new file and continue
		m_LogFile = fopen(m_strLogFile.c_str(),"w");
		if(!m_LogFile)
		{
			return false;
		}
		if(bClose)
		{
			fclose(m_LogFile);
			m_LogFile = NULL;
		}
		return true;
	}

	/******************************************************************************/

	void ConsoleLog::setStraightToFile(bool bStraightToFile)
	{
		if(bStraightToFile)
		{	// check if valid file set
			if(m_LogFile)
			{	// file is already open
				fclose(m_LogFile);
				m_LogFile=NULL;
			}
			m_LogFile = fopen(m_strLogFile.c_str(),"w");
		}
		else
		{	
			if(m_LogFile)
			{
				fclose(m_LogFile);
				m_LogFile=NULL;
			}
		}
	}

	/******************************************************************************/

	bool ConsoleLog::getStraightToFile()
	{
		return(m_LogFile==NULL);
	}

	/******************************************************************************/

	dynamicArray<CPVRTString> ConsoleLog::getLog()
	{
		return m_daLog;
	}

	/******************************************************************************/

	CPVRTString ConsoleLog::getLastLogLine()
	{
		if(m_daLog.getSize())
		{
			return m_daLog[m_daLog.getSize()-1];
		}
		return CPVRTString("");
	}

	/******************************************************************************/

	bool ConsoleLog::writeToFile()
	{
		bool bCloseFile=true;
		if(m_LogFile)
		{	// already open and writing to 
			// so replace the file with the entire log
			bCloseFile = false;
			fclose(m_LogFile);
			m_LogFile = NULL;
		}
		m_LogFile = fopen(m_strLogFile.c_str(),"w");
		if(m_LogFile)
		{
			for(unsigned int i=0;i<m_daLog.getSize();i++)
				fwrite(m_daLog[i].c_str(),1,m_daLog[i].size(),m_LogFile);
		}
		else
		{
			return false;
		}
		if(bCloseFile)
		{
			fclose(m_LogFile);
			m_LogFile=NULL;
		}
		return true;
	}

	/******************************************************************************/

	bool ConsoleLog::appendToFile()
	{
		bool bCloseFile=false;
		if(!m_LogFile)
		{ // need to open file
			m_LogFile = fopen(m_strLogFile.c_str(),"a");
			if(!m_LogFile)
			{
				return false;
			}
			bCloseFile=true;
		}
		for(unsigned int i=0;i<m_daLog.getSize();i++)
			fwrite(m_daLog[i].c_str(),1,strlen(m_daLog[i].c_str()),m_LogFile);

		if(bCloseFile)
		{
			fclose(m_LogFile);
			m_LogFile = NULL;
		}
		return true;
	}

	/******************************************************************************/

	void ConsoleLog::clearLog()
	{
		m_daLog = dynamicArray<CPVRTString>();
	}


}

/******************************************************************************
End of file (ConsoleLog.cpp)
******************************************************************************/
