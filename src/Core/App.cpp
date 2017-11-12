#include "Core/App.h"
#include "Core/Common.h"
#include "Utils/CudaUtils.h"

void signalHandler(int signal)
{
	(void)signal;

	App::getLog().logInfo("Interrupted!");
	App::getConsoleRunner().interrupt();
}

int App::run(int argc, char** argv)
{
	signal(SIGINT, signalHandler); //Interrupt from keyboard, such as  CTRL+C CTRL+D, deal such events with signalHandler. 
	signal(SIGTERM, signalHandler);//Termination signal( default is kill),  deal such events with signalHandler.

	Log& log = getLog();//record log in cudatracerlib.log
	try
	{
		Settings&      settings      = getSettings();
		ConsoleRunner& consoleRunner = getConsoleRunner();

		if(!settings.load(argc, argv))//load settings from cmd line and confiurable file-- cudatracerlib.ini.
			return 0;
		printf("\033[32;49;2m");
		log.logInfo(std::string("CudaTracerLib v") + CUDATRACERLIB_VERSION);

		if(settings.general.maxCpuThreadCount == 0)
			settings.general.maxCpuThreadCount = std::thread::hardware_concurrency();

		log.logInfo("CPU thread count: %s", settings.general.maxCpuThreadCount);

#ifdef USE_CUDA

		int deviceCount;
		CudaUtils::checkError(cudaGetDeviceCount(&deviceCount), "Could not get device count");
		CudaUtils::checkError(cudaSetDevice(settings.general.cudaDeviceNumber), "Could not set device");

		log.logInfo("CUDA selected device: %d (device count: %d)", settings.general.cudaDeviceNumber, deviceCount);

		cudaDeviceProp deviceProp;
		CudaUtils::checkError(cudaGetDeviceProperties(&deviceProp, settings.general.cudaDeviceNumber), "Could not get device properties");

		int driverVersion;
		CudaUtils::checkError(cudaDriverGetVersion(&driverVersion), "Could not get driver version");

		int runtimeVersion;
		CudaUtils::checkError(cudaRuntimeGetVersion(&runtimeVersion), "Could not get runtime version");

		log.logInfo("CUDA device: %s | Compute capability: %d.%d | Driver version: %d | Runtime version: %d", 
				deviceProp.name, deviceProp.major, deviceProp.minor, driverVersion, runtimeVersion);

#endif
		printf("\033[0m");
        
		return consoleRunner.run();//run with console terminal
	}
	catch (...)
	{
		log.logException(std::current_exception());
		return -1;
	}
}

Log& App::getLog()
{
	static Log log("cudatracerlib.log");
	return log;
}

Settings& App::getSettings()
{
	static Settings settings;
	return settings;
}

ConsoleRunner& App::getConsoleRunner()
{
	static ConsoleRunner consoleRunner;
	return consoleRunner;
}
