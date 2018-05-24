#pragma once


#include "d3d11.h"

#include <essentials/stl.h>
#include <system/file.h>


using namespace NSystem;


#define FRAMES_COUNT_IN_ONE_MEASUREMENT		100
#define BUFFERS_COUNT						4


namespace NGPU
{
	class Profiler
	{
		struct Query
		{
			struct Frame
			{
				bool used;
				float time;

				Frame()
				{
					used = false;
				}
			};

			string name;
			vector<Frame> frames;

			ID3D11Query* begin[BUFFERS_COUNT];
			ID3D11Query* end[BUFFERS_COUNT];

			Query()
			{
				frames.resize(FRAMES_COUNT_IN_ONE_MEASUREMENT);
			}
		};

	public:
		Profiler();

		bool Create();
		void Destroy();

		void StartProfiling();
		bool StopProfiling();

		void StartFrame();
		void EndFrame();

		int Begin(const string& name);
		void End(const string& name);
		void End(int index);

	public: // readonly
		bool AddQuery(const string& name);
		int QueryIndex(const string& name);

	public: // readonly
		ID3D11Query* timestampDisjointQuery[BUFFERS_COUNT];

		bool isProfiling;
		int currentFrameIndex;
		vector<Query> queries;

		File file;
	};


	class ScopedProfilerQuery
	{
	public:
		ScopedProfilerQuery(Profiler& profiler, const string& name)
		{
			this->profiler = &profiler;
			queryIndex = profiler.Begin(name);
		}

		~ScopedProfilerQuery()
		{
			profiler->End(queryIndex);
		}

	public: // readonly
		Profiler* profiler;
		int queryIndex;
	};
}
