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

Input3D::Input3D()
{
	device = NULL;
	prev_button_state = NULL;
}

Input3D::~Input3D()
{
	if (device)
		delete device;
	SAFEFREE(prev_button_state);
}

void Input3D::parse(xmlNodePtr pXMLnode)
{
	Node3D::parse(pXMLnode);

	device = new MouseKeyboardDevice3D();
}

void Input3D::app()
{
	//delayed initialization
	device->init();
	device->update(world->getDeltaTime());

	if (device->getButtons()>0)
	{
		if (device->getButton(0))
		{
			if (prev_button_state[0]==false)
			{
				EVENT_OCCURED("button1pressed");
			}
			EVENT_OCCURED("button1down");
		}
		else
		{
			if (prev_button_state[0]==true)
			{
				EVENT_OCCURED("button1released");
			}
		}
		prev_button_state[0]=device->getButton(0);
	}
	if (device->getButtons()>1)
	{
		if (device->getButton(1))
		{
			if (prev_button_state[1]==false)
			{
				EVENT_OCCURED("button2pressed");
			}
			EVENT_OCCURED("button2down");
		}
		else
		{
			if (prev_button_state[1]==true)
			{
				EVENT_OCCURED("button2released");
			}
		}
		prev_button_state[1]=device->getButton(1);
	}
	if (device->getButtons()>2)
	{
		if (device->getButton(2))
		{
			if (prev_button_state[2]==false)
			{
				EVENT_OCCURED("button3pressed");
			}
			EVENT_OCCURED("button3down");
		}
		else
		{
			if (prev_button_state[2]==true)
			{
				EVENT_OCCURED("button3released");
			}
		}
		prev_button_state[2]=device->getButton(2);
	}
	if (device->getButtons()>3)
	{
		if (device->getButton(3))
		{
			if (prev_button_state[3]==false)
			{
				EVENT_OCCURED("button4pressed");
			}
			EVENT_OCCURED("button4down");
		}
		else
		{
			if (prev_button_state[3]==true)
			{
				EVENT_OCCURED("button4released");
			}
		}
		prev_button_state[3]=device->getButton(3);
	}
	if (device->getButtons()>4)
	{
		if (device->getButton(4))
		{
			if (prev_button_state[4]==false)
			{
				EVENT_OCCURED("button5pressed");
			}
			EVENT_OCCURED("button5down");
		}
		else
		{
			if (prev_button_state[4]==true)
			{
				EVENT_OCCURED("button5released");
			}
		}
		prev_button_state[4]=device->getButton(4);
	}
	if (device->getButtons()>5)
	{
		if (device->getButton(5))
		{
			if (prev_button_state[5]==false)
			{
				EVENT_OCCURED("button6pressed");
			}
			EVENT_OCCURED("button6down");
		}
		else
		{
			if (prev_button_state[5]==true)
			{
				EVENT_OCCURED("button6released");
			}
		}
		prev_button_state[5]=device->getButton(5);
	}
	if (device->getButtons()>6)
	{
		if (device->getButton(6))
		{
			if (prev_button_state[6]==false)
			{
				EVENT_OCCURED("button7pressed");
			}
			EVENT_OCCURED("button7down");
		}
		else
		{
			if (prev_button_state[6]==true)
			{
				EVENT_OCCURED("button7released");
			}
		}
		prev_button_state[6]=device->getButton(6);
	}
	if (device->getButtons()>7)
	{
		if (device->getButton(7))
		{
			if (prev_button_state[7]==false)
			{
				EVENT_OCCURED("button8pressed");
			}
			EVENT_OCCURED("button8down");
		}
		else
		{
			if (prev_button_state[7]==true)
			{
				EVENT_OCCURED("button8released");
			}
		}
		prev_button_state[7]=device->getButton(7);
	}
	if (device->getButtons()>8)
	{
		if (device->getButton(8))
		{
			if (prev_button_state[8]==false)
			{
				EVENT_OCCURED("button9pressed");
			}
			EVENT_OCCURED("button9down");
		}
		else
		{
			if (prev_button_state[8]==true)
			{
				EVENT_OCCURED("button9released");
			}
		}
		prev_button_state[8]=device->getButton(8);
	}
	if (device->getButtons()>9)
	{
		if (device->getButton(9))
		{
			if (prev_button_state[9]==false)
			{
				EVENT_OCCURED("button10pressed");
			}
			EVENT_OCCURED("button10down");
		}
		else
		{
			if (prev_button_state[9]==true)
			{
				EVENT_OCCURED("button10released");
			}
		}
		prev_button_state[9]=device->getButton(9);
	}
	if (device->getButtons()>10)
	{
		if (device->getButton(10))
		{
			if (prev_button_state[10]==false)
			{
				EVENT_OCCURED("button11pressed");
			}
			EVENT_OCCURED("button11down");
		}
		else
		{
			if (prev_button_state[10]==true)
			{
				EVENT_OCCURED("button11released");
			}
		}
		prev_button_state[10]=device->getButton(10);
	}
	if (device->getButtons()>11)
	{
		if (device->getButton(11))
		{
			if (prev_button_state[11]==false)
			{
				EVENT_OCCURED("button12pressed");
			}
			EVENT_OCCURED("button12down");
		}
		else
		{
			if (prev_button_state[11]==true)
			{
				EVENT_OCCURED("button12released");
			}
		}
		prev_button_state[11]=device->getButton(11);
	}
	Node3D::app();
}

void Input3D::init()
{
	prev_button_state = (bool*)malloc(device->getButtons()*sizeof(bool));
	
	Node3D::init();
}

void Input3D::reset()
{
	device->reset();
	
	Node3D::init();
}

void Input3D::processMessage(char * msg)
{
	Node3D::processMessage(msg);
}

