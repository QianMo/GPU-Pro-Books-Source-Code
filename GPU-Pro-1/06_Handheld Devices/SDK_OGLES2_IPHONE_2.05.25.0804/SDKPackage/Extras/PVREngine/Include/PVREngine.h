/******************************************************************************

 @File         PVREngine.h

 @Title        API independent class declaration for PVREngine

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Main include file for the PVREngine.

******************************************************************************/
#ifndef PVRENGINE_H
#define PVRENGINE_H

/*****************************************************************************/
/*! @mainpage PVREngine 0.2a
******************************************************************************

@section _overview_ Overview
*****************************

The PVREngine is a collection of classes provided by Imagination Technologies to
aid in the development of 3D rendering applications. It builds upon PVRShell and
PVRTools to allow further abstraction of API and OS, tries to handle common
rendering functions and provides a framework upon which to build 3D applications.
It is also an example of how such code can be put together for these purposes.

The PODPlayer application source, this document and the comments in the engine
code are provided to help in understanding the workings of the PVREngine.

@section _using_ Using the PVREngine
*****************************

Steps to use the PVREngine in your project:

- Include this file in your source
@code
#include "PVREngine.h"
@endcode

- Inherit from the PVREngine class. 
@code
class MyApplication: public pvrengine::PVREngine
{
public:
	virtual bool InitApplication();
	virtual bool InitView();
	virtual bool ReleaseView();
	virtual bool QuitApplication();
	virtual bool RenderScene();
};
@endcode

- As the PVREngine class inherits from PVRShell the user also needs to register his
application class through the NewDemo function:
@code
PVRShell* NewDemo()
{
	return new MyApplication();
}
@endcode

- Specify the API for a particular build of your project by defining BUILD_*APINAME*
for the preprocessor where *APINAME* is OGL,OGLES2 etc.

- Define the correct include paths for your chosen API and link against the correct
library files: the PVREngine .lib and the PVRTools .lib for your target API and OS.

- Note that all classes and structs in the PVREngine occupy the pvrengine namespace.

For more information on PVRShell please examine its documentation also included
in the Imagination Technologies SDK.

@section _rendering_ PVREngine Rendering
*****************************

The PVREngine seeks to enable the efficient rendering of 3D geometry using POD files
as the principle source of this.

@subsection _uniformhandler_ pvrengine::UniformHandler

The uniform handler is a CPVRTSingleton class allowing the engine to keep track of essential
values such as the transformation matrices and position of lights. This class handles
the efficient calculation and binding of the attributes and uniforms required by any
shaders in use. Values that need only be calculated once per frame are done so only once.

As it holds the information required to render the actual view, facility is provided here
for culling geometry invisible to the camera.

This is a CPVRTSingleton class: an instance (or a pointer using the ptr() function) may be
retrieved with a line such as:

@code
pvrengine::UniformHandler& log = pvrengine::UniformHandler::inst();
@endcode

The other CPVRTSingleton classes used in the engine can be similarly acquired.

@subsection _mesh_ Mesh

Individual meshes are drawn separately using their assigned materials allowing gross
culling to be performed on them or whatever selective rendering is required. This is
encapsulated in the pvrengine::Mesh class.

@subsection _material_ Material

pvrengine::Material objects are used to abstract the effects and textures used for an
individual mesh so that duplication of shader compilation and other material data is
avoided. At initialisation shader uniforms and attributes required for rendering each
type of material are collated, textures are loaded and initialised and shaders compiled.

@subsection _managers_ Managers

Classes: \b pvrengine::MaterialManager, \b pvrengine::LightManager, \b pvrengine::MeshManager,
\b pvrengine::TextureManager \b pvrengine::ContextManager are provided to avoid duplication
of and easy disposal of resources at program exit. For instance:

- All meshes that are to be rendered using the same shaders or textures will share a single
material.
- All materials that use the same texture will share this texture - it won't be loaded more
than once.

@section _utility_ Utility Classes

@subsection _console_ ConsoleLog

The pvrengine::ConsoleLog class provides a place to store text output from your application.
It can be set to log to file constantly or when the write function is called. This is a
CPVRTSingleton class and so output may be logged with a single line:

@code
pvrengine::ConsoleLog::inst().log("This line is added to the log\n");
@endcode

@subsection _options_ OptionsMenu

pvrengine::OptionsMenu provides a user interface to change settings that can be set up by
the client application to the engine. Simply instantiate some pvrengine::Option objects,
add them to an OptionsMenu object, tell it to render and query the OptionsMenu for feedback.

@subsection _time_ TimeController

The pvrengine::TimeController deals with delta time calculations so that applications can
be frame rate independent. Also does frames per second calculations and POD animation timing.
This is a CPVRTSingleton.

@subsection _camera_ SimpleCamera

pvrengine::SimpleCamera is an elevation and heading camera class provided to help
with visualisation and navigation of 3D scenes.

@section _datastructure_ Data and Mathematical Structures

- dynamicArray is an expanding array template used to simplify the acquisition and disposal
of memory as required by the engine.

- pvrengine::Plane defines an infinite surface in 3D space
- pvrengine::BoundingBox defines 8 points enclosing a volume in 3D space
- pvrengine::BoundingHex defines 6 planes enclosing a volume in 3D space


******************************************************************************/

/******************************************************************************
Includes
******************************************************************************/

#include "PVRShell.h"
#include "../PVRTools.h"
#include "Globals.h"
#include "BoundingBox.h"
#include "BoundingHex.h"
#include "ConsoleLog.h"
#include "dynamicArray.h"
#include "LightManager.h"
#include "MaterialManager.h"
#include "MeshManager.h"
#include "Option.h"
#include "OptionsMenu.h"
#include "Plane.h"
#include "SimpleCamera.h"
#include "TextureManager.h"
#include "TimeController.h"
#include "Uniform.h"

#include "ContextManager.h"
#include "Light.h"
#include "Material.h"
#include "Mesh.h"
#include "PVRESettings.h"
#include "UniformHandler.h"
#include "Uniform.h"
#include "PVRESettings.h"

/*!***************************************************************************
* @Namespace pvrengine
* @Brief The PVREngine namespace.
* @Description The PVREngine namespace.
*****************************************************************************/
namespace pvrengine
{

	/*!***************************************************************************
	* @Class PVREngine
	* @Brief 	Main include file for the PVREngine.
	* @Description 	Main include file for the PVREngine.
	*****************************************************************************/
	class PVREngine : public PVRShell
	{
	protected:
		PVRESettings	m_PVRESettings;			/*! Settings struct for the engine */

	public:
		/*!***********************************************************************
		@Function			PVREngine
		@Description		Constructor
		*************************************************************************/
		PVREngine(){}
	};

}
#endif // PVRENGINE_H

/******************************************************************************
End of file (PVREngine.h)
******************************************************************************/
