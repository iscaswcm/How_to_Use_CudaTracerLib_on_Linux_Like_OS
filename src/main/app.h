#ifndef _CORE_APP_H_
#define _CORE_APP_H_

#include "core/precompiled.h"
#include <signal.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/settings.h"
#include "core/log.h"
#include "main/consolerunner.h"

class Log;
class Settings;
class ConsoleRunner;

class App
{
public:

	static int run(int argc, char** argv);
	static Log& getLog();
	static Settings& getSettings();
	static ConsoleRunner& getConsoleRunner();
};
#endif
