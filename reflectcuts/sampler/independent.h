#pragma once

#include "common/sampler.h"
#include "common/rng.h"

class IndependentSampler : public Sampler
{
public:
	IndependentSampler() {}
	IndependentSampler(const uint32_t seed): _mRng(seed) {}

	inline unique_ptr<Sampler> clone(const uint32_t seed) const override
	{
		return make_unique<IndependentSampler>(seed);
	}

	inline int32_t nextInt32(const int32_t start, const int32_t end) const override
	{
		return _mRng.nextInt32(start, end);
	}

	inline uint32_t nextUint32(const uint32_t start, const uint32_t end) const override
	{
		return _mRng.nextUint32(start, end);
	}

	inline Float nextFloat(const Float start, const Float end) const override
	{
		return _mRng.nextFloat(start, end);
	}

	inline Float nextFloat() const override
	{
		return _mRng.nextFloat();
	}

	inline Vec2 nextVec2() const override
	{
		return Vec2(_mRng.nextFloat(), _mRng.nextFloat());
	}

private:
	Rng _mRng;
};