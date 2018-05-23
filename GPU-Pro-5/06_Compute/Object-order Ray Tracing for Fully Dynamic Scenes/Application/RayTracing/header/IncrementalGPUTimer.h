#pragma once

#include "Tracing.h"

#include <beGraphics/Any/beQuery.h>
#include <beGraphics/Any/beAPI.h>

namespace app
{

class IncrementalGPUTimer
{
private:
	static const uint4 BufferCount = 3;

	lean::com_ptr<beg::api::Query> startQuery[BufferCount];
	lean::com_ptr<beg::api::Query> endQuery[BufferCount];
	bool waiting[BufferCount];
	uint4 bufferIdx;

	uint8 ticks;
	float ms;

public:
	IncrementalGPUTimer() { }
	IncrementalGPUTimer(beg::api::Device *device) { Construct(device); }

	void Construct(beg::api::Device *device)
	{
		for (uint4 b = 0; b < BufferCount; ++b)
		{
			startQuery[b] = beg::Any::CreateTimestampQuery(device);
			endQuery[b] = beg::Any::CreateTimestampQuery(device);
		}
		Reset();
	}

	void Begin(beg::api::DeviceContext *context)
	{
		if (waiting[bufferIdx])
			ReadData(context, bufferIdx);

		context->End(startQuery[bufferIdx]);
	}
	void End(beg::api::DeviceContext *context)
	{
		context->End(endQuery[bufferIdx]);
		waiting[bufferIdx] = true;

		bufferIdx = (bufferIdx + 1) % BufferCount;
	}

	uint8 ReadData(beg::api::DeviceContext *context, uint4 bufferIdx)
	{
		if (waiting[bufferIdx])
		{
			uint8 beginStamp = beg::Any::GetTimestamp(context, startQuery[bufferIdx]);
			uint8 endStamp = beg::Any::GetTimestamp(context, endQuery[bufferIdx]);

			ticks += endStamp - beginStamp;

			waiting[bufferIdx] = false;
		}

		return ticks;
	}

	uint8 ReadData(beg::api::DeviceContext *context)
	{
		for (uint4 b = 0; b < BufferCount; ++b)
			ReadData(context, b);

		return ticks;
	}

	float ToMS(uint8 frequency)
	{
		ms = ticks * 1000000 / frequency / 1000.0f;
		return ms;
	}

	void Reset()
	{
		for (uint4 b = 0; b < BufferCount; ++b)
			waiting[b] = false;
		bufferIdx = 0;
		ticks = 0;
		ms = 0.0f;
	}

	const float* GetDataMS() const { return &ms; }
};

} // namespace