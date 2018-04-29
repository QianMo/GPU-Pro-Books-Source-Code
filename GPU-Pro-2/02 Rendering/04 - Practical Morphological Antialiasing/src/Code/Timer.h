/*
 * Copyright (C) 2010 Jorge Jimenez (jim@unizar.es)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must display the name 'Jorge Jimenez' as
 *    'Real-Time Rendering R&D' in the credits of the application, if such
 *    credits exist. The author of this work must be notified via email
 *    (jim@unizar.es) in this case of redistribution.
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

#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <string>
#include <map>
#include <vector>

class Timer {
    public:
        Timer(ID3D10Device *device);
        ~Timer();

        void reset() { sections.clear(); }
        void start(const std::wstring &msg=L"");
        float clock(const std::wstring &msg=L"");
        float accumulated() const { return accum; }

        void setEnabled(bool enabled) { this->enabled = enabled; }
        bool isEnabled() const { return enabled; }

        void setFlushEnabled(bool flushEnabled) { this->flushEnabled = flushEnabled; }
        bool isFlushEnabled() const { return flushEnabled; }

        void setWindowSize(int windowSize) { this->windowSize = windowSize; }
        int getWindowSize() { return windowSize; }

        friend std::wostream &operator<<(std::wostream &out, const Timer &timer);

    private:
        float mean(const std::wstring &msg, float t);
        void flush();

        ID3D10Device *device;
        ID3D10Query *event;

        __int64 t0;
        float accum;

        bool enabled;
        bool flushEnabled;
        int windowSize;

        class Section {
            public:
                Section() : mean(0.0), pos(0), completed(0.0f) {}
                std::vector<std::pair<float, bool> > buffer;
                float mean;
                int pos;
                float completed;
        };
        std::map<std::wstring, Section> sections;
};

#endif