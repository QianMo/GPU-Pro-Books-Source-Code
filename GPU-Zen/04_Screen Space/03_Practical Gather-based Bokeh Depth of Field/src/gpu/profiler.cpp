#include <gpu/profiler.h>


NGPU::Profiler::Profiler()
{
	isProfiling = false;
}


bool NGPU::Profiler::Create()
{
	for (uint j = 0; j < BUFFERS_COUNT; j++)
	{
		D3D11_QUERY_DESC qd;
		qd.MiscFlags = 0;
		qd.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;

		if (FAILED(device->CreateQuery(&qd, &timestampDisjointQuery[j])))
			return false;
	}

	file.Open("profiler.txt", File::OpenMode::WriteText);

	return true;
};


void NGPU::Profiler::Destroy()
{
	for (uint i = 0; i < BUFFERS_COUNT; i++)
		SAFE_RELEASE(timestampDisjointQuery[i]);

	for (uint i = 0; i < queries.size(); i++)
	{
		for (uint j = 0; j < BUFFERS_COUNT; j++)
		{
			SAFE_RELEASE(queries[i].begin[j]);
			SAFE_RELEASE(queries[i].end[j]);
		}
	}

	file.Close();
};


void NGPU::Profiler::StartProfiling()
{
	isProfiling = true;
	currentFrameIndex = 0;

	for (uint i = 0; i < queries.size(); i++)
	{
		for (uint j = 0; j < queries[i].frames.size(); j++)
			queries[i].frames[j].used = false;
	}
}


bool NGPU::Profiler::StopProfiling()
{
	if (isProfiling && currentFrameIndex == FRAMES_COUNT_IN_ONE_MEASUREMENT + BUFFERS_COUNT - 1)
	{
		for (uint i = 0; i < queries.size(); i++)
		{
			bool allUsed = true;
			float avgTime = 0.0;

			for (uint j = 0; j < queries[i].frames.size(); j++)
			{
				if (queries[i].frames[j].used)
				{
					avgTime += queries[i].frames[j].time;
				}
				else
				{
					allUsed = false;
					break;
				}
			}

			if (allUsed)
			{
				avgTime /= (float)queries[i].frames.size();
				file.WriteText(queries[i].name + " - " + ToString(avgTime) + "\n");
			}
		}
		file.WriteTextNewline();

		isProfiling = false;

		return true;
	}
	else
	{
		return false;
	}
}


void NGPU::Profiler::StartFrame()
{
	if (isProfiling && currentFrameIndex >= 0 && currentFrameIndex < FRAMES_COUNT_IN_ONE_MEASUREMENT)
		deviceContext->Begin(timestampDisjointQuery[currentFrameIndex % BUFFERS_COUNT]);
}


void NGPU::Profiler::EndFrame()
{
	if (!isProfiling)
		return;

	//

	if (currentFrameIndex >= 0 && currentFrameIndex < FRAMES_COUNT_IN_ONE_MEASUREMENT)
		deviceContext->End(timestampDisjointQuery[currentFrameIndex % BUFFERS_COUNT]);

	//

	int frameBeingCollectedIndex = (currentFrameIndex - BUFFERS_COUNT + 1);
	int bufferIndex = frameBeingCollectedIndex % BUFFERS_COUNT;

	if (frameBeingCollectedIndex >= 0 && frameBeingCollectedIndex < FRAMES_COUNT_IN_ONE_MEASUREMENT)
	{
		while (deviceContext->GetData(timestampDisjointQuery[bufferIndex], NULL, 0, 0) == S_FALSE)
			Sleep(1);

		D3D11_QUERY_DATA_TIMESTAMP_DISJOINT qdtd;
		deviceContext->GetData(timestampDisjointQuery[bufferIndex], &qdtd, sizeof(qdtd), 0);

		if (!qdtd.Disjoint)
		{
			for (uint i = 0; i < queries.size(); i++)
			{
				uint64 begin, end;

				deviceContext->GetData(queries[i].begin[bufferIndex], &begin, sizeof(uint64), 0);
				deviceContext->GetData(queries[i].end[bufferIndex], &end, sizeof(uint64), 0);

				queries[i].frames[frameBeingCollectedIndex].time = float(end - begin) / float(qdtd.Frequency) * 1000.0f;
			}
		}
	}

	//

	currentFrameIndex++;
}


int NGPU::Profiler::Begin(const string& name)
{
	if (isProfiling && currentFrameIndex >= 0 && currentFrameIndex < FRAMES_COUNT_IN_ONE_MEASUREMENT)
	{
		int queryIndex = QueryIndex(name);

		if (queryIndex >= 0)
		{
			queries[queryIndex].frames[currentFrameIndex].used = true;
			deviceContext->End(queries[queryIndex].begin[currentFrameIndex % BUFFERS_COUNT]);
			return queryIndex;
		}
		else
		{
			AddQuery(name);
			queries.back().frames[currentFrameIndex].used = true;
			deviceContext->End(queries.back().begin[currentFrameIndex % BUFFERS_COUNT]);
			return queries.size() - 1;
		}
	}

	return -1;
}


void NGPU::Profiler::End(const string& name)
{
	if (isProfiling && currentFrameIndex >= 0 && currentFrameIndex < FRAMES_COUNT_IN_ONE_MEASUREMENT)
	{
		int queryIndex = QueryIndex(name);

		if (queryIndex >= 0 && queries[queryIndex].frames[currentFrameIndex].used)
			deviceContext->End(queries[queryIndex].end[currentFrameIndex % BUFFERS_COUNT]);
	}
}


void NGPU::Profiler::End(int index)
{
	if (isProfiling && currentFrameIndex >= 0 && currentFrameIndex < FRAMES_COUNT_IN_ONE_MEASUREMENT)
	{
		if (queries[index].frames[currentFrameIndex].used)
			deviceContext->End(queries[index].end[currentFrameIndex % BUFFERS_COUNT]);
	}
}


bool NGPU::Profiler::AddQuery(const string& name)
{
	Query query;

	query.name = name;

	D3D11_QUERY_DESC qd;
	qd.MiscFlags = 0;
	qd.Query = D3D11_QUERY_TIMESTAMP;

	for (int j = 0; j < BUFFERS_COUNT; j++)
	{
		if (FAILED(device->CreateQuery(&qd, &query.begin[j])))
			return false;
		if (FAILED(device->CreateQuery(&qd, &query.end[j])))
			return false;
	}

	queries.push_back(query);

	return true;
}


int NGPU::Profiler::QueryIndex(const string& name)
{
	for (uint i = 0; i < queries.size(); i++)
	{
		if (queries[i].name == name)
			return i;
	}

	return -1;
}
