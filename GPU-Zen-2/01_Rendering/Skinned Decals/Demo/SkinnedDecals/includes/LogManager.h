#ifndef LOG_MANAGER_H
#define LOG_MANAGER_H

#define MAX_LOG_MSG_LENGTH 1024 //max length of log-message

#ifdef _DEBUG
  #define LOG_INFO(...) Demo::logManager->LogInfo(__VA_ARGS__) 

  #define LOG_WARNING(...) Demo::logManager->LogWarning(__VA_ARGS__) 

  #define LOG_ERROR(...) Demo::logManager->LogError(__VA_ARGS__) 
#else
  #define LOG_INFO(...)

  #define LOG_WARNING(...)

  #define LOG_ERROR(...)
#endif

// LogManager
//
class LogManager
{
public:
	void LogInfo(const char *msg, ...);

	void LogWarning(const char *msg, ...); 

	void LogError(const char *msg, ...);

};

#endif