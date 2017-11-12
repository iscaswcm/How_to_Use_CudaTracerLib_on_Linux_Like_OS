#ifndef _MOVING_RANGE_H_
#define _MOVING_RANGE_H_


#include "Core/Common.h"

class MovingAverage
{
	public:

		CUDA_CALLABLE explicit MovingAverage(float alpha = 1.0f, float average = 0.0f);

		CUDA_CALLABLE void setAlpha(float alpha);
		CUDA_CALLABLE void setAverage(float average);
		CUDA_CALLABLE void addMeasurement(float value);
		CUDA_CALLABLE float getAverage() const;

	private:

		float alpha;
		float average;
};
#endif
