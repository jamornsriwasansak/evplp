#pragma once

#include "common/reflectcuts.h"

#include <iostream>
#include <vector>
#include <random>

class Rng32
{
public:
	Rng32(): _mGenerator32(std::random_device()()) {}
#ifdef USE_DETERMINISTIC_RESULT
	Rng32(const uint32_t seed): _mGenerator32(seed) {}
#else
	Rng32(const uint32_t seed): _mGenerator32(std::random_device()()) {}
#endif

	inline int32_t nextInt32(const int32_t start, const int32_t end) const
	{
		assert(end >= start);
		return std::uniform_int_distribution<int32_t>(start, end)(_mGenerator32);
	}
	
	inline uint32_t nextUint32(const uint32_t start, const uint32_t end) const
	{
		assert(end >= start);
		return std::uniform_int_distribution<uint32_t>(start, end)(_mGenerator32);
	}

	inline float nextFloat(const float start, const float end) const
	{
		assert(end >= start);
		return std::uniform_real_distribution<float>(start, end)(_mGenerator32);
	}

	inline float nextFloat() const
	{
		return this->nextFloat(0.0f, 1.0f);
	}

private:
	mutable std::mt19937 _mGenerator32;
};

class Rng64
{
public:
	Rng64(): _mGenerator64(std::random_device()()) {}
#ifdef USE_DETERMINISTIC_RESULT
	Rng64(const uint64_t seed): _mGenerator64(seed) {}
#else
	Rng64(const uint64_t seed): _mGenerator64(std::random_device()()) {}
#endif

	inline int32_t nextInt32(const int32_t start, int32_t end) const
	{
		assert(end >= start);
		return std::uniform_int_distribution<int32_t>(start, end)(_mGenerator64);
	}
	
	inline int64_t nextInt64(const int64_t start, int64_t end) const
	{
		assert(end >= start);
		return std::uniform_int_distribution<int64_t>(start, end)(_mGenerator64);
	}
	
	inline uint32_t nextUint32(const uint32_t start, const uint32_t end) const
	{
		assert(end >= start);
		return std::uniform_int_distribution<uint32_t>(start, end)(_mGenerator64);
	}

	inline uint64_t nextUint64(const uint64_t start, const uint64_t end) const
	{
		assert(end >= start);
		return std::uniform_int_distribution<uint64_t>(start, end)(_mGenerator64);
	}

	inline double nextFloat(const double start, const double end) const
	{
		assert(end >= start);
		return std::uniform_real_distribution<double>(start, end)(_mGenerator64);
	}

	inline double nextFloat() const
	{
		return this->nextFloat(0, 1);
	}

private:
	mutable std::mt19937_64 _mGenerator64;
};
