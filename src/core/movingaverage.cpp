#include "core/precompiled.h"

#include "core/movingaverage.h"

MovingAverage::MovingAverage(float alpha_, float average_)
{
	setAlpha(alpha_);
	setAverage(average_);
}

void MovingAverage::setAlpha(float alpha_)
{
	assert(alpha_ > 0.0f && alpha_ <= 1.0f);

	alpha = alpha_;
}

void MovingAverage::setAverage(float average_)
{
	average = average_;
}

void MovingAverage::addMeasurement(float value)
{
	average = alpha * value + (1.0f - alpha) * average;
}

float MovingAverage::getAverage() const
{
	return average;
}
