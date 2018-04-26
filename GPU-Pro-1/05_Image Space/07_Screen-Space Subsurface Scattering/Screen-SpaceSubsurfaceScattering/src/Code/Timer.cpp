/*
 * Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)
 * Copyright (C) 2009 Diego Gutierrez (diegog@unizar.es)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, 
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must display the names 'Jorge Jimenez'
 *    and 'Diego Gutierrez' as 'Real-Time Rendering R&D' in the credits of the
 *    application, if such credits exist. The authors of this work must be
 *    notified via email (jim@unizar.es) in this case of redistribution.
 * 
 * 3. Neither the name of copyright holders nor the names of its contributors 
 *    may be used to endorse or promote products derived from this software 
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS 
 * IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS 
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <DXUT.h>
#include "timer.h"
using namespace std;


Timer::Timer(ID3D10Device *device) : device(device), enabled(true), flushEnabled(true), windowSize(1) {
    D3D10_QUERY_DESC desc;
    desc.Query = D3D10_QUERY_EVENT;
    desc.MiscFlags = 0; 
    device->CreateQuery(&desc, &event);

    start();
}


Timer::~Timer() {
    SAFE_RELEASE(event);
}


void Timer::start(const wstring &msg) {
    if (enabled) {
        if (flushEnabled) {
            flush();
        }

        QueryPerformanceCounter((LARGE_INTEGER*) &t0);
        accum = 0.0f;
    }
}


float Timer::clock(const wstring &msg) {
    if (enabled) {
        if (flushEnabled) {
            flush();
        }

        __int64 t1, freq;
        QueryPerformanceCounter((LARGE_INTEGER*) &t1);
        QueryPerformanceFrequency((LARGE_INTEGER*) &freq);
        float t = float(double(t1 - t0) / double(freq));

        QueryPerformanceCounter((LARGE_INTEGER*) &t0);

        float m = mean(msg, t);
        accum += m;

        return m;
    } else {
        return 0.0f;
    }
}


float Timer::mean(const std::wstring &msg, float t) {
    Section &section = buffer[msg];
    if (windowSize > 1) {
        section.buffer.resize(windowSize);
        section.buffer[(section.pos++) % windowSize] = t;

        section.mean = 0.0;
        for (int i = 0; i < int(section.buffer.size()); i++) {
            section.mean += section.buffer[i];
        }
        section.mean /= float(windowSize);

        return section.mean;
    } else {
        section.mean = t;
        return section.mean;
    }
}


void Timer::flush() {
    event->End();

    BOOL queryData;
    while (event->GetData(&queryData, sizeof(BOOL), 0) != S_OK) {
        ;
    }
}


wostream &operator<<(wostream &out, const Timer &timer) { 
    for (std::map<std::wstring, Timer::Section>::const_iterator section = timer.buffer.begin();
         section != timer.buffer.end();
         section++) {
        const wstring &name = section->first;
        const float &sum = section->second.mean;
        out << name << L":" << sum << L"s:" << int(100.0f * sum / timer.accum) << L"%:" << int(1.0 / sum) << L"fps ";
    }
    return out;
}