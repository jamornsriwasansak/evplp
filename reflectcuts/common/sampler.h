#pragma once

#include "reflectcuts.h"

class Sampler
{
public:
	virtual unique_ptr<Sampler> clone(const uint32_t seed) const = 0;

	virtual int32_t nextInt32(const int32_t start, const int32_t end) const = 0;
	virtual int32_t nextInt32() const
	{
		return nextInt32(std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max());
	}

	virtual uint32_t nextUint32(const uint32_t start, const uint32_t end) const = 0;
	virtual uint32_t nextUint32() const
	{
		return nextUint32(0, std::numeric_limits<uint32_t>::max());
	}

	virtual Float nextFloat(const Float start, const Float end) const = 0;
	virtual Float nextFloat() const = 0;
	virtual Vec2 nextVec2() const = 0;
};