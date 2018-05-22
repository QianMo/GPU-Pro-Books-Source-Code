//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//  Scene Graph 3D                                                          //
//  Georgios Papaioannou, 2009                                              //
//                                                                          //
//  This is a free, extensible scene graph management library that works    //
//  along with the EaZD deferred renderer. Both libraries and their source  //
//  code are free. If you use this code as is or any part of it in any kind //
//  of project or product, please acknowledge the source and its author.    //
//                                                                          //
//  For manuals, help and instructions, please visit:                       //
//  http://graphics.cs.aueb.gr/graphics/                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

#include "SceneGraph.h"

//------------------- DECLARATION OF SCENE GRAPH NODE NAMES --------------------
//
// Add your own associations here
//

Node3D * buildNode(char *name)
{
#define DECLARE_NODE(STRING,TYPE) \
    if (STR_EQUAL(name,STRING)) return new TYPE();

	// node naming map -----------------------------
	DECLARE_NODE("group",Group3D)
	DECLARE_NODE("world",World3D)
	DECLARE_NODE("transformation",TransformNode3D)
	DECLARE_NODE("linearmotion",LinearMotion3D)
	DECLARE_NODE("spinner",Spinner3D)
	DECLARE_NODE("object",Geometry3D)
	DECLARE_NODE("camera",Camera3D)
	DECLARE_NODE("user",User3D)
	DECLARE_NODE("input",Input3D)
	DECLARE_NODE("light",Light3D)
	DECLARE_NODE("character",Cal3D)
	DECLARE_NODE("force",Force3D)

	return NULL;
#undef DECLARE_NODE
}

//----------------------- Message map initialization ---------------------------
//
// Add your own node itialization directives here
//

INIT_MESSAGE_MAP(Node3D)
INIT_MESSAGE_MAP(Geometry3D)
INIT_MESSAGE_MAP(Group3D)
INIT_MESSAGE_MAP(Camera3D)
INIT_MESSAGE_MAP(Light3D)
INIT_MESSAGE_MAP(User3D)
INIT_MESSAGE_MAP(World3D)
INIT_MESSAGE_MAP(TransformNode3D)
INIT_MESSAGE_MAP(LinearMotion3D)
INIT_MESSAGE_MAP(Spinner3D)
INIT_MESSAGE_MAP(Input3D)
INIT_MESSAGE_MAP(Cal3D)
INIT_MESSAGE_MAP(Force3D)

//----------------------- Event registration ---------------------------------
//
// Add your own node events. In the source code of your new node classes
// call EVENT_OCCURED( [eventname] ) to trigger the respective event
//

void SceneGraphRegisterEvents()
{
	static bool registered = false;
	if (registered)
		return;
	
	REGISTER_EVENT(World3D,"init");
	
	REGISTER_EVENT(Input3D,"button1down");
	REGISTER_EVENT(Input3D,"button1pressed");
	REGISTER_EVENT(Input3D,"button1released");
	REGISTER_EVENT(Input3D,"button2down");
	REGISTER_EVENT(Input3D,"button2pressed");
	REGISTER_EVENT(Input3D,"button2released");
	REGISTER_EVENT(Input3D,"button3down");
	REGISTER_EVENT(Input3D,"button3pressed");
	REGISTER_EVENT(Input3D,"button3released");
	REGISTER_EVENT(Input3D,"button4pressed");
	REGISTER_EVENT(Input3D,"button4released");
	REGISTER_EVENT(Input3D,"button5pressed");
	REGISTER_EVENT(Input3D,"button5released");
	REGISTER_EVENT(Input3D,"button6pressed");
	REGISTER_EVENT(Input3D,"button6released");
	REGISTER_EVENT(Input3D,"button7pressed");
	REGISTER_EVENT(Input3D,"button7released");
	REGISTER_EVENT(Input3D,"button8pressed");
	REGISTER_EVENT(Input3D,"button8released");
	REGISTER_EVENT(Input3D,"button9pressed");
	REGISTER_EVENT(Input3D,"button9released");
	REGISTER_EVENT(Input3D,"button10pressed");
	REGISTER_EVENT(Input3D,"button10released");
	REGISTER_EVENT(Input3D,"button11pressed");
	REGISTER_EVENT(Input3D,"button11released");
	REGISTER_EVENT(Input3D,"button12pressed");
	REGISTER_EVENT(Input3D,"button12released");
	
	REGISTER_EVENT(User3D,"moveforward");
	REGISTER_EVENT(User3D,"movebackward");
	REGISTER_EVENT(User3D,"moveleft");
	REGISTER_EVENT(User3D,"moveright");
	REGISTER_EVENT(User3D,"moveup");
	REGISTER_EVENT(User3D,"movedown");
	REGISTER_EVENT(User3D,"tiltup");
	REGISTER_EVENT(User3D,"tiltdown");
	REGISTER_EVENT(User3D,"turnleft");
	REGISTER_EVENT(User3D,"turnright");
	
	REGISTER_EVENT(Camera3D,"follow");
}
