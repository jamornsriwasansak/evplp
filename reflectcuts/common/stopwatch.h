#pragma once

#include <ctime>
#include <chrono>

class StopWatch {
public:

	StopWatch() {}
	inline void reset()
	{
		_mStartTime = std::chrono::high_resolution_clock::now();
	}

	inline long long timeMilliSec()
	{
		std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - _mStartTime).count();
	}

	inline long long timeNanoSec()
	{
		std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - _mStartTime).count();
	}

private:

	std::chrono::high_resolution_clock::time_point _mStartTime;
};