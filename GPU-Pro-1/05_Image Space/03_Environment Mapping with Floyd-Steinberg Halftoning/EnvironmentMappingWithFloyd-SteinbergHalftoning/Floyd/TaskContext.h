#pragma once

class Theatre;
class ResourceOwner;

class RenderContext;
class AnimationContext;
class ControlContext;
class MessageContext;

class TaskContext
{
public:
	Theatre* theatre;
	ResourceOwner* localResourceOwner;

	TaskContext(Theatre* theatre, ResourceOwner* localResourceOwner)
	{
		this->theatre = theatre;
		this->localResourceOwner = localResourceOwner;
	}

	virtual const RenderContext*		asRenderContext() const {return NULL;}
	virtual const AnimationContext*		asAnimationContext() const {return NULL;}
	virtual const ControlContext*		asControlContext() const {return NULL;}
	virtual const MessageContext*		asMessageContext() const {return NULL;}
};
