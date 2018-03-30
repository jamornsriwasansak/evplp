#pragma once

#include <vector>
#include <cassert>

#include "common/reflectcuts.h"

#include "math/math.h"
#include "math/ray.h"
#include "math/aabb.h"

class Shape
{
public:
	Shape() {}
	virtual bool intersectP(const Ray & r) const { assert(false && "unimplemented"); return false; }
	virtual bool intersect(Intersection * isectPtr, const Ray & r) const { assert(false && "unimplemented"); return false; }
	virtual bool canIntersect() const { return false; }
	virtual void refine(std::vector<shared_ptr<Shape>> * refine) const { assert(false && "unimplemented"); }
	inline virtual bool clipAabb(Aabb * resultPtr, Float split1, Float split2, uint8_t dim) const { assert(false && "unimplemented"); return false; }
	virtual void samplePosition(Vec3 * position, Vec3 * direction, const Sampler & sampler) const { assert(false && ("unimplmented")); }

	virtual Aabb computeBbox() const = 0;
	virtual Float mArea() const
	{
		assert(false && "unimplemented");
		return Float(0.0);
	}
	shared_ptr<Material> mMaterialPtr = nullptr;
};