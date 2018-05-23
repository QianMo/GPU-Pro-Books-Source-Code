/*****************************************************/
/* breeze Framework Launch Lib  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beLauncherInternal/stdafx.h"
#include "beLauncher/beInitEngine.h"

#include <lean/logging/log.h>
#include <lean/logging/log_file.h>

#include <beCore/beFileSystem.h>

// Default-initializes the engine.
void beLauncher::InitializeEngine(const lean::utf8_t *pLog, const lean::utf8_t *pFilesystem)
{
	InitializeLog(pLog);
	InitializeFilesystem(pFilesystem);

	LEAN_LOG_BREAK();
}

// Default-initializes the path environment
void beLauncher::InitializeLog(const lean::utf8_t *pPath)
{
	static const char *LogFileName = "Logs/breeze.log";

	static struct BreezeLog
	{
		lean::log_file logFile;

		BreezeLog(const lean::utf8_t *path)
			: logFile(path)
		{
			if (logFile.valid())
			{
				lean::error_log().add_target(&logFile);
				lean::info_log().add_target(&logFile);

				LEAN_LOG("Log file initialized: " << path);
			}
			else
				LEAN_LOG("Log file could not be initialized: " << path);
		}
		~BreezeLog()
		{
			lean::info_log().remove_target(&logFile);
			lean::error_log().remove_target(&logFile);
		}
	} breezeLog( (pPath) ? pPath : LogFileName );
}

// Default-initializes the path environment
void beLauncher::InitializeFilesystem(const lean::utf8_t *pPath)
{
	static const char *ConfigFileName = "filesystem.xml";

	static struct BreezeFS
	{
		lean::utf8_string path;

		BreezeFS(const lean::utf8_t *path)
			: path(path)
		{
			try
			{
				beCore::FileSystem::Get().LoadConfiguration(path);
				LEAN_LOG("Filesystem configured: " << path);
			}
			catch (std::runtime_error&)
			{
				LEAN_LOG("Filesystem could not be configured: " << path);
			}
		}
		~BreezeFS()
		{
			try
			{
				beCore::FileSystem::Get().SaveConfiguration(path);
				LEAN_LOG("Filesystem configuration written to disk: " << path);
			}
			catch (std::runtime_error&)
			{
				LEAN_LOG("Filesystem configuration could not be written to disk: " << path);
			}
		}
	} breezeFS( (pPath) ? pPath : ConfigFileName );
}
