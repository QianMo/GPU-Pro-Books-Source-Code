#include <stdafx.h>
#include <Demo.h>
#include <LogManager.h>

void LogManager::LogInfo(const char *msg, ...)
{
	char buffer[MAX_LOG_MSG_LENGTH]; 
	va_list va;
	va_start(va, msg);
	vsnprintf(buffer, MAX_LOG_MSG_LENGTH, msg, va);
	va_end(va);

  char str[MAX_LOG_MSG_LENGTH + 1];
  sprintf(str, "%s\n", buffer);
	OutputDebugStringA(str);
}

void LogManager::LogWarning(const char *msg, ...)
{ 
	char buffer[MAX_LOG_MSG_LENGTH]; 
	va_list va;
	va_start(va,msg);
	vsnprintf(buffer, MAX_LOG_MSG_LENGTH, msg, va);
	va_end(va);

	char str[MAX_LOG_MSG_LENGTH + 11];
  sprintf(str, "[WARNING] %s\n", buffer);
	OutputDebugStringA(str);
}

void LogManager::LogError(const char *msg, ...)
{
	char buffer[MAX_LOG_MSG_LENGTH]; 
	va_list va;
	va_start(va, msg);
	vsnprintf(buffer, MAX_LOG_MSG_LENGTH, msg, va);
	va_end(va);

	char str[MAX_LOG_MSG_LENGTH + 9];
  sprintf(str, "[ERROR] %s\n", buffer);
	OutputDebugStringA(str);
}