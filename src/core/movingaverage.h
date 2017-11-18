#ifndef _MOVING_RANGE_H_
#define _MOVING_RANGE_H_


#include "core/common.h"

class MovingAverage
{
	public:

		 explicit MovingAverage(float alpha = 1.0f, float average = 0.0f);

		 void setAlpha(float alpha);
		 void setAverage(float average);
		 void addMeasurement(float value);
		 float getAverage() const;

	private:

		float alpha;
		float average;
};
#endif
