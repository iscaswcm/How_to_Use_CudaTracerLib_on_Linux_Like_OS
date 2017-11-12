// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Math/MovingAverage.h"

CUDA_CALLABLE MovingAverage::MovingAverage(float alpha_, float average_)
{
	setAlpha(alpha_);
	setAverage(average_);
}

CUDA_CALLABLE void MovingAverage::setAlpha(float alpha_)
{
	assert(alpha_ > 0.0f && alpha_ <= 1.0f);

	alpha = alpha_;
}

CUDA_CALLABLE void MovingAverage::setAverage(float average_)
{
	average = average_;
}

CUDA_CALLABLE void MovingAverage::addMeasurement(float value)
{
	average = alpha * value + (1.0f - alpha) * average;
}

CUDA_CALLABLE float MovingAverage::getAverage() const
{
	return average;
}
