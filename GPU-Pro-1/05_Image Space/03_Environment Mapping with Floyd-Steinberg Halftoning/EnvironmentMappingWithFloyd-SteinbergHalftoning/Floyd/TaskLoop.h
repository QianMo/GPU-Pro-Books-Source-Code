#pragma once
#include "Task.h"

class Play;
class XMLNode;
class TaskBlock;
class ResourceOwner;

class TaskLoop :
	public Task
{
	TaskBlock* taskBlock;
	D3DXVECTOR3 start;
	D3DXVECTOR3 step;
	D3DXVECTOR3 end;
//	std::wstring effectName;
//	std::string variableName;
	ID3D10EffectVectorVariable* variable;
public:
//	TaskLoop(const std::wstring& effectName, const std::string& variableName, const D3DXVECTOR3& start, const D3DXVECTOR3& step, const D3DXVECTOR3& end);
//	void loadTasks(Play* play, XMLNode& loopNode);
	TaskLoop(Play* play, XMLNode& loopNode, ResourceOwner* localResourceOwner);
	void execute(const TaskContext& context);
	~TaskLoop();

};
