#pragma once
#include "ResourceOwner.h"
#include "TaskBlock.h"
#include "AnimationTask.h"
#include "ControlTask.h"
#include "MessageTask.h"

typedef std::vector<AnimationTask> AnimationTaskList;
typedef std::vector<ControlTask> ControlTaskList;
typedef std::vector<MessageTask> MessageTaskList;
class Play;
class XMLNode;
class ControlStatus;
class MessageContext;

class Act : public ResourceOwner
{
	Play* play;

	AnimationTaskList animationTaskList;
	ControlTaskList controlTaskList;
	MessageTaskList messageTaskList;

	void loadAnimationTasks(XMLNode& scriptNode);
	void loadControlTasks(XMLNode& scriptNode);
	void loadMessageTasks(XMLNode& scriptNode);

public:
	Act(Play* play, XMLNode& actNode);
	~Act(void);

	void processMessage( const MessageContext& context);
	void animate(double dt, double t);
	void render();
	void control(const ControlStatus& status, double dt);

	ResourceSet* getResourceSet();
};
