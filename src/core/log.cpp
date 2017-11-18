#include "core/log.h"

using namespace std::chrono;
Log::Log()
{
}

Log::Log(const std::string& logFileName)
{
	setLogFile(logFileName);
}

void Log::setLogFile(const std::string& logFileName)
{
	if(logFile.is_open())
		logFile.close();

	logFile.open(logFileName);
}

void Log::setMinimumMessageLevel(LogMessageLevel value)
{
	minimumMessageLevel = value;
}

void Log::handleMessage(LogMessageLevel messageLevel, const std::string& message)
{
	if(messageLevel >= minimumMessageLevel)
	{
		std::string formattedMessage = formatMessage(messageLevel, message);
		outputMessage(messageLevel, formattedMessage);
	}
}

std::string Log::formatMessage(LogMessageLevel messageLevel, const std::string& message)
{
	std::string messageLevelName;

	switch (messageLevel)
	{
		case LogMessageLevel::DEBUG:
			messageLevelName = "Debug";
			break;

		case LogMessageLevel::INFO:
			messageLevelName = "Info";
			break;

		case LogMessageLevel::WARNING:
			messageLevelName = "Warning";
			break;

		case LogMessageLevel::ERROR:
			messageLevelName = "Error";
			break;

		default: messageLevelName = "Unknown"; break;
	}

	auto now = std::chrono::system_clock::now();
	auto epoch = now.time_since_epoch();
	time_t tt = system_clock::to_time_t(now);
	tm local_tm = *localtime(&tt);
	auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch).count();
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
	milliseconds = milliseconds - seconds * 1000;

	return tfm::format("%02d:%02d:%02d.%03d [%s] %s", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec, milliseconds, messageLevelName, message);
}

void Log::outputMessage(LogMessageLevel messageLevel, const std::string& message)
{
	std::lock_guard<std::mutex> lock(outputMutex);

	std::cout << message << std::endl;

	if(logFile.is_open())
	{
		logFile << message << std::endl;
		logFile.flush();
	}
}

void Log::logMessage(LogMessageLevel messageLevel, const std::string& message)
{
	handleMessage(messageLevel, message);
}

void Log::logDebug(const std::string& message)
{
	logMessage(LogMessageLevel::DEBUG, message);
}

void Log::logInfo(const std::string& message)
{
	logMessage(LogMessageLevel::INFO, message);
}

void Log::logWarning(const std::string& message)
{
	logMessage(LogMessageLevel::WARNING, message);
}

void Log::logError(const std::string& message)
{
	logMessage(LogMessageLevel::ERROR, message);
}

void Log::logException(const std::exception_ptr& exception)
{
	try
	{
		std::rethrow_exception(exception);
	}
	catch (const std::exception& ex)
	{
		logError("Exception: %s: %s", typeid(ex).name(), ex.what());
	}
	catch (const std::string& s)
	{
		logError("Exception: %s", s);
	}
	catch (...)
	{
		logError("Unknown exception!");
	}
}
