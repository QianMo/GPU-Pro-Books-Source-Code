/*****************************************************/
/* breeze Framework Launch Lib  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_LAUNCHER_INITENGINE
#define BE_LAUNCHER_INITENGINE

#include "beLauncher.h"

namespace beLauncher
{

/// Default-initializes the engine.
BE_LAUNCHER_API void InitializeEngine(const lean::utf8_t *pLog = nullptr, const lean::utf8_t *pFilesystem = nullptr);

/// Default-initializes the log file.
BE_LAUNCHER_API void InitializeLog(const lean::utf8_t *pPath = nullptr);
/// Default-initializes the path environment
BE_LAUNCHER_API void InitializeFilesystem(const lean::utf8_t *pPath = nullptr);

}

#endif