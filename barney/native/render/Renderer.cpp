// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/render/Renderer.h"
#include "native/Context.h"

namespace BARNEY_NS {
  namespace native {
    
    /*! the base class for _any_ other type of object/actor in the
      barney class hierarchy */
    Renderer::Renderer(Context *context)
      : Object(context)
    {}

    /*! pretty-printer for printf-debugging */
    std::string Renderer::toString() const
    {
      return "barney::Renderer";
    }

    Renderer::SP Renderer::create(Context *context)
    { return std::make_shared<Renderer>(context); }

    void Renderer::commit()
    {
      bgColor         = staged.bgColor;
      ambientRadiance = staged.ambientRadiance;
      pathsPerPixel   = staged.pathsPerPixel;
      bgTexture       = staged.bgTexture;
      cutPlane        = staged.cutPlane;
      maxVolumeBounces = staged.maxVolumeBounces;
      sampleLimit     = staged.sampleLimit;
      maxBounce       = staged.maxBounce;
      maxDiffuseBounce = staged.maxDiffuseBounce;
      maxGlossyBounce = staged.maxGlossyBounce;
      maxTransmissionBounce = staged.maxTransmissionBounce;
      clampDirect     = staged.clampDirect;
      clampIndirect   = staged.clampIndirect;
      // volumeMultiScatter = staged.volumeMultiScatter;
    }
  
    bool Renderer::setObject(const std::string &member,
                             const std::shared_ptr<Object> &value) 
    {
      if (Object::setObject(member,value))
        return true;
      if (member == "bgTexture") {
        staged.bgTexture = value ? value->as<Texture>() : 0;
        return true;
      }
      return false;
    }
  
    bool Renderer::set1f(const std::string &member, const float &value)
    {
      if (Object::set1f(member,value))
        return true;
      if (member == "ambientRadiance") {
        staged.ambientRadiance = value;
        return true;
      }
      if (member == "clampDirect") {
        staged.clampDirect = value;
        return true;
      }
      if (member == "clampIndirect") {
        staged.clampIndirect = value;
        return true;
      }
      return false;
    }
  
    bool Renderer::set1i(const std::string &member, const int &value)
    {
      if (Object::set1i(member,value))
        return true;
      if (member == "pathsPerPixel") {
        staged.pathsPerPixel = value;
        return true;
      }
      if (member == "crosshairs") {
        staged.crosshairs = value;
        return true;
      }
      if (member == "maxVolumeBounces") {
        staged.maxVolumeBounces = value;
        return true;
      }
      if (member == "sampleLimit") {
        staged.sampleLimit = value;
        return true;
      }
      if (member == "maxBounce") {
        staged.maxBounce = value;
        return true;
      }
      if (member == "maxDiffuseBounce") {
        staged.maxDiffuseBounce = value;
        return true;
      }
      if (member == "maxGlossyBounce") {
        staged.maxGlossyBounce = value;
        return true;
      }
      if (member == "maxTransmissionBounce") {
        staged.maxTransmissionBounce = value;
        return true;
      }
      // if (member == "volumeMultiScatter") {
      //   staged.volumeMultiScatter = value;
      //   return true;
      // }
      return false;
    }
  
    bool Renderer::set4f(const std::string &member, const vec4f &value)
    {
      if (Object::set4f(member,value))
        return true;
      if (member == "bgColor") {
        staged.bgColor = value;
        return true;
      }
      if (member == "cutPlane") {
        staged.cutPlane = value;
        return true;
      }
      return false;
    }

    Renderer::DD Renderer::getDD(Device *device) const
    {
      Renderer::DD dd;
      dd.bgColor = bgColor;
      dd.bgTexture
        = bgTexture ? bgTexture->getTextureObject(device)
        : 0;
      dd.ambientRadiance = ambientRadiance;
      dd.pathsPerPixel = pathsPerPixel;
      dd.cutPlane = cutPlane;
      dd.maxVolumeBounces = maxVolumeBounces;
      dd.maxBounce = maxBounce;
      dd.maxDiffuseBounce = maxDiffuseBounce;
      dd.maxGlossyBounce = maxGlossyBounce;
      dd.maxTransmissionBounce = maxTransmissionBounce;
      // as in cycles the value limits the mean of the channels
      dd.clampDirectSum = 3.f * std::max(clampDirect,0.f);
      dd.clampIndirectSum = 3.f * std::max(clampIndirect,0.f);
      // dd.volumeMultiScatter = volumeMultiScatter;
      return dd;
    }

  }
}
