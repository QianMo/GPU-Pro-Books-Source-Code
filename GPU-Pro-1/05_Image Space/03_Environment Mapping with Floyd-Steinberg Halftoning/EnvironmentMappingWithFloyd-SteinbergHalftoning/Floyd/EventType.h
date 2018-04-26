#pragma once


class EventType
{
	const unsigned int id;
	EventType(unsigned int id, const wchar_t* typeName);
	void operator=(const EventType& o){}

	typedef std::map<const std::wstring, const EventType*> EventTypeInstanceMap;

	static EventTypeInstanceMap names;

public:
	bool operator==(const EventType& o) const;
	bool operator<(const EventType& o) const;

	static const EventType createDevice;
	static const EventType createSwapChain;
	static const EventType createAct;
	static const EventType createMesh;
	static const EventType createShadedMesh;
	static const EventType createEntity;
	static const EventType nop;
	static const EventType renderFrame;

	static const EventType& fromString(const std::wstring& typeName);
};