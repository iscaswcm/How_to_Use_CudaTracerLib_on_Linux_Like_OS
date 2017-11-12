// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.
#ifndef _TIMER_H_
#define _TIMER_H_


#include <chrono>

#include "Math/MovingAverage.h"

struct TimerData
{
	uint64_t totalNanoseconds = 0;
	uint64_t totalMilliseconds = 0;
	uint64_t totalSeconds = 0;
	uint64_t totalMinutes = 0;
	uint64_t totalHours = 0;

	uint64_t hours = 0;
	uint64_t minutes = 0;
	uint64_t seconds = 0;
	uint64_t milliseconds = 0;

	std::string getString(bool withMilliseconds = false) const;
};

class Timer
{
	public:

		Timer();

		void restart();

		float getElapsedMilliseconds() const;
		float getElapsedSeconds() const;
		TimerData getElapsed() const;

		void setTargetValue(float value);
		void updateCurrentValue(float value);
		void setAveragingAlpha(float alpha);
		TimerData getRemaining();
		float getPercentage() const;

	private:

		std::chrono::high_resolution_clock::time_point startTime;

		float currentValue = 0.0f;
		float targetValue = 0.0f;

		MovingAverage remainingMillisecondsAverage;
};
#endif
