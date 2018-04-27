/******************************************************************************

 @File         ContextManager.h

 @Title        

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     OGLES2

 @Description  Manages contexts so that extensions can be accessed. This is
               intended to be a foundation for a more sophisticated manager in
               the future. TODO: make work with multiple contexts

******************************************************************************/
#ifndef _CONTEXTMANAGER_H_
#define _CONTEXTMANAGER_H_

#include "../PVRTools.h"
#include "../PVRTSingleton.h"
#include "dynamicArray.h"

namespace pvrengine
{
	/*!***************************************************************************
	* @Class ContextManager
	* @Brief Manages contexts predominantly for extensions.
	* @Description Manages contexts predominantly for extensions.
 	*****************************************************************************/
	class ContextManager : public CPVRTSingleton<ContextManager>
	{
	public:
		/*!***************************************************************************
		@Function			ContextManager
		@Description		Constructor.
		*****************************************************************************/
		ContextManager();

		/*!***************************************************************************
		@Function			ContextManager
		@Description		Destructor.
		*****************************************************************************/
		~ContextManager(){}

		/*!***************************************************************************
		@Function			getExtensions
		@Return				pointer to the extensions
		@Description		Allows access to the OpenGL extensions class for the
							current context.
		*****************************************************************************/
		CPVRTgles2Ext* getExtensions();

		/*!***************************************************************************
		@Function			getCurrentContext
		@Return				pointer to the extensions
		@Description		Allows access to the OpenGL extensions class for the
							current context.
		*****************************************************************************/
		SPVRTContext*	getCurrentContext();

		/*!***************************************************************************
		@Function			initContext
		@Description		Initialises the current context.
		*****************************************************************************/
		void initContext();

		/*!***************************************************************************
		@Function			addNewContext
		@Return			handle of current context
		@Description		adds a new context to the manager and returns its handle.
		*****************************************************************************/
		int addNewContext();

		/*!***************************************************************************
		@Function			setCurrentContext
		@Input				i32Context	context handle
		@Description		sets the current context.
		*****************************************************************************/
		void setCurrentContext(const unsigned int i32Context);
	private:
		/****************************************************************************
		** Variables
		****************************************************************************/
		/*! the list of contexts */
		dynamicArray<SPVRTContext> m_daContext;
		/*! handle of current context */
		int m_i32CurrentContext;
	};
}

#endif // _CONTEXTMANAGER_H_

/*****************************************************************************
 End of file (ContextManager.h)
*****************************************************************************/
