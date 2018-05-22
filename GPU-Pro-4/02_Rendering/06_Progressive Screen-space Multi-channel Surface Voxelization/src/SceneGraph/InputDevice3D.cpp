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

InputDevice3D::InputDevice3D()
{
	int i;
	num_axes = num_buttons = 0;
	err_code = SCENE_GRAPH_ERROR_NONE;
	initialized = false;
	for (i=0; i<16; i++)
		axes[i].value = 0.0f;
	for (i=0; i<32; i++)
		buttons[i] = false;
}

InputDevice3D::~InputDevice3D()
{
}

void InputDevice3D::init()
{
	if (initialized)
		return;
	initialized = true;
}

void InputDevice3D::reset()
{
	int i;
	for (i=0; i<num_axes; i++)
		axes[i].value = 0.0f;
	for (i=0; i<num_buttons; i++)
		buttons[i] = false;
}

void InputDevice3D::passData(void *)
{
}

void InputDevice3D::setRanges(int axis, float rmin, float rmax)
{
	if (axis>=num_axes)
		return;
	axes[axis].range_min = rmin;
	axes[axis].range_max = rmax;
}

float InputDevice3D::getNormalizedValue(int axis)
{
	if (axis>=num_axes)
		return 0.0f;
	return 2.0f*(axes[axis].value-axes[axis].range_min)/
		     (axes[axis].range_max-axes[axis].range_min)-1.0f;
}

float InputDevice3D::getValue(int axis)
{
	if (axis>=num_axes)
		return 0.0f;
	return axes[axis].value;
}

bool InputDevice3D::getButton(int i)
{
	if (i>=num_buttons)
		return false;
	return buttons[i];
}

int InputDevice3D::getButtons()
{
	return num_buttons;
}

int InputDevice3D::getAxes()
{
	return num_axes;
}

