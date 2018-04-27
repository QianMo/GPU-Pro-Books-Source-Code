/******************************************************************************

 @File         ContextManager.cpp

 @Title        

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Manages contexts so that extensions can be accessed. TODO: make
               work with multiple contexts

******************************************************************************/
#include "ContextManager.h"

namespace pvrengine
{



/****************************************************************************
** Functions
****************************************************************************/

	/******************************************************************************/

	ContextManager::ContextManager():
		m_daContext(dynamicArray<SPVRTContext>(1)),	// unlikely to use more than one context
			m_i32CurrentContext(0)
	{
	}

	/******************************************************************************/

	CPVRTgles2Ext* ContextManager::getExtensions()
	{
		return NULL;	// no extensions in OGLES2
	}

	/******************************************************************************/

	SPVRTContext* ContextManager::getCurrentContext()
	{
		if(m_i32CurrentContext>=0)
			return &m_daContext[m_i32CurrentContext];
		else
			return NULL;
	}

	/******************************************************************************/

	void ContextManager::initContext()
	{
		m_i32CurrentContext = 0;
	}

	/******************************************************************************/

	int ContextManager::addNewContext()
	{
		m_daContext.expandToSize(m_daContext.getSize()+1);
		return m_daContext.getSize();
	}

	/******************************************************************************/

	void ContextManager::setCurrentContext(const unsigned int i32Context)
	{
		if(i32Context>0 && i32Context<m_daContext.getSize())
		{
			m_i32CurrentContext = i32Context;
		}
	}

}

/*****************************************************************************
 End of file (ContextManager.cpp)
*****************************************************************************/
