#ifndef __LOG_H
#define __LOG_H

#include <stdarg.h>
#include <stdio.h>

class Log
{
public:
  static void Error(const char* fmt, ...)
  {
    char buf[4096] = { };
    va_list argptr;
    va_start(argptr, fmt);
    vsnprintf_s(buf, sizeof(buf) - 1, fmt, argptr);
    va_end(argptr);
    fputs(buf, stderr);
  }
  static void Info(const char* fmt, ...)
  {
    char buf[4096] = { };
    va_list argptr;
    va_start(argptr, fmt);
    vsnprintf_s(buf, sizeof(buf) - 1, fmt, argptr);
    va_end(argptr);
    fputs(buf, stdout);
  }
};

#endif //#ifndef __LOG_H
