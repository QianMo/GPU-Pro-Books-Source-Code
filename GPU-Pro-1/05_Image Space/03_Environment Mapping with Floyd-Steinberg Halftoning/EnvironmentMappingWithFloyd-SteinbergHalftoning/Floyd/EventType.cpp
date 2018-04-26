#include "DXUT.h"
#include "EventType.h"

EventType::EventType(unsigned int id, const wchar_t* typeName)
:id(id)
{
	names[typeName] = this;
}

EventType::EventTypeInstanceMap EventType::names;

bool EventType::operator==(const EventType& o) const
{
	return id == o.id;
}

bool EventType::operator<(const EventType& o) const
{
	return id < o.id;
}

const EventType& EventType::fromString(const std::wstring& typeName)
{
	EventTypeInstanceMap::iterator i = names.find(typeName);
	if(i != names.end())
	{
		return *(i->second);
	}
	else
		EggERR("Unknown event type.");
	return nop;
}

const EventType EventType::nop(0, L"nop");
const EventType EventType::createDevice(1, L"createDevice");
const EventType EventType::createSwapChain(2, L"createSwapChain");
const EventType EventType::createAct(3, L"createAct");
const EventType EventType::createMesh(4, L"createMesh");
const EventType EventType::createShadedMesh(5, L"createShadedMesh");
const EventType EventType::createEntity(6, L"createEntity");
const EventType EventType::renderFrame(7, L"renderFrame");