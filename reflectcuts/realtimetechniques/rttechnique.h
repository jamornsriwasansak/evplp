#pragma once

#include "json/json.hpp"
#include "rtcommon.h"

class RtTechnique
{
public:
	virtual void render(shared_ptr<RtScene> & scene, const glm::vec2 & resolution, const nlohmann::json & json) = 0;
};