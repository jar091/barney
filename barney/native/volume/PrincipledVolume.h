// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "barney/native/common/barney-common.h"
#include "native/volume/TransferFunction.h"

namespace BARNEY_NS {
  namespace native {
    
    struct Device;

    /*! Direct two-transfer-function volume model. The regular transfer
      function supplies scattering tint and extinction; the emission transfer
      function supplies emission color and mask. */
    struct PrincipledVolumeParams {
      int     enabled = 0;
      /*! Blender ColorRamp mode: if no emission transfer function is set,
        the regular (color/opacity) transfer function also drives the
        emission color, i.e. the emitted radiance per unit length is
        Le = emissionStrength * tf.rgb * tf.w (tf.w already holds
        1/unitDistance). An explicit emission transfer function takes
        precedence. */
      int     colorRamp = 0;
      float   emissionStrength = 0.f;

      struct DD {
        int     enabled;
        int     colorRamp;
        float   emissionStrength;
      };

      inline DD getDD(Device *device) const
      {
        DD dd;
        dd.enabled = enabled;
        dd.colorRamp = colorRamp;
        dd.emissionStrength = emissionStrength;
        (void)device;
        return dd;
      }
    };

    /*! Radiance added at a real collision. The emission transfer function
        independently controls emission color and mask. 'xfSample' is the
        regular transfer function sampled at the collision point (rgb =
        tint, w = extinction sigma_t); it is used in colorRamp mode when no
        emission transfer function is set. */
    inline __rtc_device
    vec3f principledCollisionEmission(float scalar,
                                      const vec4f &xfSample,
                                      const TransferFunction::DD &emissionXf,
                                      int emissionXfActive,
                                      const PrincipledVolumeParams::DD &p)
    {
      if (p.emissionStrength <= 0.f)
        return vec3f(0.f);

      if (emissionXfActive) {
        const vec4f e = emissionXf.map(scalar);
        if (e.w <= 0.f)
          return vec3f(0.f);
        return p.emissionStrength * e.w * getPos(e);
      }

      if (p.colorRamp) {
        /* collision estimator of an emitting medium (as in Cycles): a
           real collision happens with density sigma_t = tf.w, so the
           emission added there is Le/sigma_t = emissionStrength*tf.rgb */
        if (xfSample.w <= 0.f)
          return vec3f(0.f);
        return p.emissionStrength * getPos(xfSample);
      }

      return vec3f(0.f);
    }

  }
}
