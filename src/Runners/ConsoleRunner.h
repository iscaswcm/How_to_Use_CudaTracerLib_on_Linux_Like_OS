#ifndef _CONSOLE_RUNNER_H_
#define _CONSOLE_RUNNER_H_
#include "Utils/Timer.h"
class ConsoleRunner
{
public:
	bool interrupted;
	int run();
	void interrupt();

private:

	void printProgress(float percentage, const TimerData& elapsed, const TimerData& remaining, uint32_t pixelSamples, const int currentpass);
	MovingAverage samplesPerSecondAverage;
};
#endif
