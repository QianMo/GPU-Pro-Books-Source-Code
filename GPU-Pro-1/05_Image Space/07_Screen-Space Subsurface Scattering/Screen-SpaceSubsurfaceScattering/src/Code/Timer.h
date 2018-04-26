// Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)

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

        void reset() { buffer.clear(); }
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
                Section() : mean(0.0), pos(0) {}
                std::vector<float> buffer;
                float mean;
                int pos;
        };
        std::map<std::wstring, Section> buffer;
};

#endif