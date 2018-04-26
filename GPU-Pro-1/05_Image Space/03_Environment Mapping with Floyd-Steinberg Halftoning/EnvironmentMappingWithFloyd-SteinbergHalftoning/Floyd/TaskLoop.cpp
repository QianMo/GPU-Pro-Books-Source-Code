#include "DXUT.h"
#include "TaskLoop.h"
#include "TaskBlock.h"
#include "Theatre.h"
#include "xmlParser.h"

/*
TaskLoop::TaskLoop(const std::wstring& effectName, const std::string& variableName, const D3DXVECTOR3& start, const D3DXVECTOR3& step, const D3DXVECTOR3& end)
{
	this->effectName = effectName;
	this->variableName = variableName;
	this->start = start;
	this->step = step;
	this->end = end;
}
*/

TaskLoop::TaskLoop(Play* play, XMLNode& loopNode, ResourceOwner* localResourceOwner)
{
	const wchar_t* effectName = loopNode|L"effect";
	ID3D10Effect* effect;
	if(effectName)
	{
		effect = play->getTheatre()->getEffect(effectName);
		if(effect == NULL)
			EggXMLERR(loopNode, L"Rendering loop variable effect not found. [" << effectName << L"]");
	}
	else
		EggXMLERR(loopNode, L"Rendering loop variable effect not specified.");
	std::string variableName = loopNode.readString(L"variable");
	variable = effect->GetVariableByName(variableName.c_str())->AsVector();
	if(variable == NULL)
		EggXMLERR(loopNode, L"Rendering loop variable not found. [" << variableName.c_str() << L"]");

	start = loopNode.readVector(L"start");
	step = loopNode.readVector(L"step", D3DXVECTOR3(100, 100, 100));
	end = loopNode.readVector(L"end");

	taskBlock = new TaskBlock();
	taskBlock->loadTasks(play, loopNode, localResourceOwner);
}

TaskLoop::~TaskLoop(void)
{
	delete taskBlock;
}


void TaskLoop::execute(const TaskContext& context)
{
	for(int x = start.x; x <= end.x; x += step.x)
		for(int y = start.y; y <= end.y; y += step.y)
			for(int z = start.z; z <= end.z; z += step.z)
			{
				int d[4];
				d[0] = x;
				d[1] = y;
				d[2] = z;
				d[3] = 0;
				variable->SetIntVector(d);
				taskBlock->execute(context);
			}
}