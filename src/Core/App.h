#ifndef _CORE_APP_H_
#define _CORE_APP_H_

#include "Core/Precompiled.h"
#include <signal.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "Utils/Settings.h"
#include "Utils/Log.h"
#include "Runners/ConsoleRunner.h"

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
