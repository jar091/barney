// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/common/barney-common.h"
#include "native/common/math.h"
#include "native/Object.h"

namespace BARNEY_NS {
  namespace native {

    /*! the camera model we use in barney */
    struct Camera : public Object {
      typedef std::shared_ptr<Camera> SP;
      
      typedef enum {
        UNDEFINED=0,
        PERSPECTIVE,
        ORTHOGRAPHIC,
        OMNIDIRECTIONAL
      } Type;
      
      /*! device-data for the camera object; to avoid virtual functions
        this currently uses a 'type'-switch, so the camera code on the
        device will have to 'interpret' what the actual fields
        mean. on the host side, all derived cameras will have to fill
        in this one shared struct */
      struct DD {
        inline DD() : type(UNDEFINED) {};
        inline DD(const DD &) = default;
        inline ~DD() = default;
      
        Type  type = UNDEFINED;
        vec4f imageRegion {0.f, 0.f, 1.f, 1.f};
        
        /* Motion-vector reprojection matrices (world -> clip). When
           haveMotionMatrices is false, BN_FB_MOTION writes are skipped
           regardless of whether per-instance deltas were supplied. */
        bool     haveMotionMatrices = false;
        float    currViewProj[16]   = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        float    prevViewProj[16]   = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

        /*! near/far clip planes (perspective and orthographic cameras
          only). The planes are perpendicular to the camera direction
          'clipAxis' (unit length), the distances are measured along
          that axis, not along the ray. Only camera rays get clipped;
          near=0 and far=inf mean 'no clipping' */
        float clipNear = 0.f;
        float clipFar  = BARNEY_INF;
        vec3f clipAxis { 0.f, 0.f, 1.f };

#if RTC_DEVICE_CODE
        /*! true if camera rays need to get clipped at all */
        inline __rtc_device bool hasClipPlanes() const
        { return clipNear > 0.f || clipFar < 1e30f; }

        /*! computes the distances along the camera ray with (unit)
          direction 'dir' at which that ray crosses the near and far
          clip planes. Returns false if the ray does not point into
          the clipped half space at all */
        inline __rtc_device bool getClipRange(vec3f dir,
                                              float &tNear,
                                              float &tFar) const
        {
          const float cosAxis = dot(dir,clipAxis);
          tNear = 0.f;
          tFar  = 1e30f;
          if (cosAxis <= 1e-6f) return false;
          if (clipNear > 0.f)  tNear = clipNear / cosAxis;
          if (clipFar < 1e30f) tFar  = clipFar / cosAxis;
          return true;
        }
#endif

        union {
          /* vector along u direction, for ONE pixel */
          struct {
            /*! vector from camera center to to lower-left pixel (i.e., pixel
              (0,0)) on the focal plane */
            vec3f dir_00;
            vec3f dir_du;
            /* vector along v direction, for ONE pixel */
            vec3f dir_dv;
            /*! lens center ... */
            vec3f lens_00;
            /* radius of lens, for DOF */
            float apertureRadius;
            /* distance to focal plane, for DOF */
            float focusDistance;
          } perspective;
          struct {
            vec3f org_00;
            vec3f org_du;
            vec3f org_dv;
            vec3f dir;
            float height;
            float aspect;
          } orthographic;
          struct {
            affine3f toWorld;
          } omni;
        };
      };
      DD dd;

      Camera(Context *owner);
      virtual ~Camera() = default;

      /*! subtypes call from their commit() so base camera state propagates
        into DD after subtype-specific fields are populated */
      void commitBaseDeviceData();
      
      bool set1i(const std::string &member, const int &value) override;
      bool set4f(const std::string &member, const vec4f &value) override;
      bool set4x4f(const std::string &member, const vec4f *value) override;
      
      static Camera::SP create(Context *owner, const std::string &type);

      DD getDD() { return dd; }
    protected:
      bool  motionEnabled = false;
      vec4f imageRegion {0.f, 0.f, 1.f, 1.f};
      float motionCurrViewProj[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
      float motionPrevViewProj[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
      
    };

  }
}
