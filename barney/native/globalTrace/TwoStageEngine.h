// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file TwoStageEngine.h Optional, run-time switchable optimizations
  of the two-stage ray-queue exchange (see TwoStage.h). This header is
  self-contained on purpose (only MPI + owl math), and parameterized
  over an 'Env' that provides memory, kernel launches, fences, and the
  local trace. barney instantiates it with a CUDA/OptiX environment
  (TwoStage.cu); a CPU-only test harness instantiates it with a host
  environment and a mock tracer, so the whole routing / exchange /
  reduction logic (including the MPI state machine) can be validated
  without GPUs.

  Features (all independent, all off by default; the default TwoStage
  code path is not touched when none of them is enabled):

  - cull        : per-rank world-space bounds are all-gathered; a ray
                  is only sent to a host (cross-node stage) / GPU
                  (intra-node stage) whose (inflated) box its segment
                  [0,tMax] overlaps. Rays are bucketed per destination
                  on the device, and a per-destination index list is
                  kept so hits can be routed back and reduced into the
                  originating ray.
  - compactHits : only rays whose tMax decreased during local tracing
                  (i.e., that actually hit something) produce a hit
                  record; hit records are sent as (index,hit) pairs.
  - smallPayload: 24-byte ray records (octahedral direction, 27-bit RNG
                  seed hash + 5 flag bits) and 96-byte hit records
                  (octahedral normal + type packed into 32 bits),
                  instead of 40 / 100 bytes.
  - numChunks   : the owner's rays are split into k chunks that are
                  pushed through the whole two-stage pipeline
                  independently by a small non-blocking state machine,
                  so communication of one chunk overlaps with local
                  tracing of another.

  Reduction order (and thus tie-breaking between equal tHit values) is
  the same as in the default code path: first over the GPUs of a host
  (in gpu order), then over hosts (in host order), always 'strictly
  closer wins'.
*/

#pragma once

#include <mpi.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <owl/common/math/vec.h>
#include <owl/common/math/box.h>

#ifndef __rtc_both
# error "TwoStageEngine.h requires __rtc_both (include the barney rtc headers, or define it for a host-only build)"
#endif

namespace BARNEY_NS {
  namespace native {
    namespace ts_opt {

      using owl::common::vec3f;
      using owl::common::box3f;

      // ==================================================================
      // small device/host helpers
      // ==================================================================

      /*! atomically increments the given counter and returns the old
        value (i.e., a unique slot index). On CUDA this is
        warp-aggregated (threads of a warp that hit the same counter
        issue a single atomic), which matters because in the cull
        kernels all threads of a launch increment the same few
        counters. */
      inline __rtc_both int tsAtomicInc(int *counter)
      {
#if defined(__CUDA_ARCH__)
# if __CUDA_ARCH__ >= 700
        const unsigned active = __activemask();
        const unsigned peers
          = __match_any_sync(active,(unsigned long long)counter);
        unsigned lane;
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
        const int leader = __ffs(peers) - 1;
        int base = 0;
        if ((int)lane == leader)
          base = atomicAdd(counter,__popc(peers));
        base = __shfl_sync(peers,base,leader);
        return base + __popc(peers & ((1u << lane) - 1u));
# else
        return atomicAdd(counter,1);
# endif
#elif defined(__HIP_DEVICE_COMPILE__)
        return atomicAdd(counter,1);
#else
        // host environments run 'kernels' serially
        return (*counter)++;
#endif
      }

      inline __rtc_both float tsSignNotZero(float f)
      { return (f >= 0.f) ? 1.f : -1.f; }

      inline __rtc_both vec3f tsNormalize(vec3f v)
      {
        const float l = sqrtf(v.x*v.x+v.y*v.y+v.z*v.z);
        return (l > 0.f) ? v * (1.f/l) : vec3f(0.f,0.f,1.f);
      }

      /*! octahedral projection of a (not necessarily unit) direction
        onto [-1,1]^2 */
      inline __rtc_both void tsOctProject(vec3f n, float &u, float &v)
      {
        const float l1 = fabsf(n.x)+fabsf(n.y)+fabsf(n.z);
        if (!(l1 > 0.f)) { u = 0.f; v = 0.f; return; }
        float x = n.x / l1;
        float y = n.y / l1;
        if (n.z < 0.f) {
          const float ox = (1.f - fabsf(y)) * tsSignNotZero(x);
          const float oy = (1.f - fabsf(x)) * tsSignNotZero(y);
          x = ox; y = oy;
        }
        u = x; v = y;
      }

      inline __rtc_both vec3f tsOctUnproject(float x, float y)
      {
        vec3f n(x, y, 1.f - fabsf(x) - fabsf(y));
        if (n.z < 0.f) {
          const float ox = (1.f - fabsf(y)) * tsSignNotZero(x);
          const float oy = (1.f - fabsf(x)) * tsSignNotZero(y);
          n.x = ox; n.y = oy;
        }
        return tsNormalize(n);
      }

      inline __rtc_both float tsOctDequant(uint32_t q, uint32_t maxQ)
      { return (float)q * (2.f/(float)maxQ) - 1.f; }

      /*! decode a 2x'bits' octahedral code (u in the low bits). Each
        component uses the codes 0..2^bits-2 only (an odd number of
        codes, so that 0 and +-1 are exactly representable); the
        all-ones code is never produced by the encoder and can be
        used as a marker. */
      inline __rtc_both vec3f tsOctDecode(uint32_t code, int bits)
      {
        const uint32_t mask = (1u<<bits)-1u;
        const uint32_t maxQ = mask-1u;
        const uint32_t qu = code & mask;
        const uint32_t qv = (code >> bits) & mask;
        return tsOctUnproject(tsOctDequant(qu,maxQ),tsOctDequant(qv,maxQ));
      }

      /*! 'precise' octahedral encoding with 2x'bits' bits: of the four
        quantized neighbors of the projected point, pick the one whose
        decoded direction is closest to the input (roughly halves the
        worst-case error of plain rounding) */
      inline __rtc_both uint32_t tsOctEncode(vec3f n, int bits)
      {
        const uint32_t maxQ = (1u<<bits)-2u;
        float u, v;
        tsOctProject(n,u,v);
        const float fu = (fminf(fmaxf(u,-1.f),1.f)*.5f+.5f)*(float)maxQ;
        const float fv = (fminf(fmaxf(v,-1.f),1.f)*.5f+.5f)*(float)maxQ;
        const uint32_t u0 = (uint32_t)fminf(floorf(fu),(float)maxQ);
        const uint32_t v0 = (uint32_t)fminf(floorf(fv),(float)maxQ);
        const vec3f nn = tsNormalize(n);
        uint32_t best = u0 | (v0 << bits);
        // note: compare squared distances, not dot products - the
        // candidates differ by ~1e-5 rad, far below the float
        // resolution of a dot product close to 1
        float bestDist = 1e30f;
        for (int du=0;du<2;du++)
          for (int dv=0;dv<2;dv++) {
            const uint32_t qu = (u0+du > maxQ) ? maxQ : u0+du;
            const uint32_t qv = (v0+dv > maxQ) ? maxQ : v0+dv;
            const uint32_t code = qu | (qv << bits);
            const vec3f d = tsOctDecode(code,bits) - nn;
            const float dist = d.x*d.x+d.y*d.y+d.z*d.z;
            if (dist < bestDist) { bestDist = dist; best = code; }
          }
        return best;
      }

      /*! 27-bit hash of a 64-bit RNG state (murmur3 finalizer) */
      inline __rtc_both uint32_t tsSeedHash27(uint64_t s)
      {
        s ^= s >> 33; s *= 0xff51afd7ed558ccdull;
        s ^= s >> 33; s *= 0xc4ceb9fe1a85ec53ull;
        s ^= s >> 33;
        return (uint32_t)(s >> 37);
      }

      /*! deterministic, injective expansion of a 27-bit seed hash back
        into a 64-bit RNG state (splitmix64 of an odd-multiplier
        affine map, both bijective) */
      inline __rtc_both uint64_t tsSeedExpand(uint32_t h)
      {
        uint64_t z = (uint64_t)h * 0x9e3779b97f4a7c15ull + 0xcafef00dd15ea5e5ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
      }

      /*! plain world-space box that can be passed to device code */
      struct TSBox { vec3f lower, upper; };

      /*! conservative ray-segment vs box test. mirrors what the trace
        kernel does (zero direction components are replaced by 1e-6,
        tMin is 0); boxes are expected to be inflated already. Empty
        boxes and segments with tMax <= 0 never overlap. */
      inline __rtc_both bool tsSegmentOverlapsBox(vec3f org, vec3f dir,
                                                  float tMax,
                                                  const TSBox &box)
      {
        if (!(tMax > 0.f)) return false;
        if (box.lower.x > box.upper.x ||
            box.lower.y > box.upper.y ||
            box.lower.z > box.upper.z) return false;
        if (dir.x == 0.f) dir.x = 1e-6f;
        if (dir.y == 0.f) dir.y = 1e-6f;
        if (dir.z == 0.f) dir.z = 1e-6f;
        float t0 = 0.f, t1 = tMax;
        {
          const float inv = 1.f/dir.x;
          float tn = (box.lower.x - org.x)*inv, tf = (box.upper.x - org.x)*inv;
          if (tn > tf) { float t = tn; tn = tf; tf = t; }
          t0 = fmaxf(t0,tn); t1 = fminf(t1,tf);
        }
        {
          const float inv = 1.f/dir.y;
          float tn = (box.lower.y - org.y)*inv, tf = (box.upper.y - org.y)*inv;
          if (tn > tf) { float t = tn; tn = tf; tf = t; }
          t0 = fmaxf(t0,tn); t1 = fminf(t1,tf);
        }
        {
          const float inv = 1.f/dir.z;
          float tn = (box.lower.z - org.z)*inv, tf = (box.upper.z - org.z)*inv;
          if (tn > tf) { float t = tn; tn = tf; tf = t; }
          t0 = fmaxf(t0,tn); t1 = fminf(t1,tf);
        }
        // relative slack against rounding in the slab computation
        return t0 <= t1 * (1.f + 1e-5f) + 1e-20f;
      }

      /*! inflate a box by a relative epsilon so that the float slab
        test above is conservative w.r.t. whatever the actual
        intersection programs report (and w.r.t. rounding in the
        instance-transformed bounds). Empty boxes stay empty. */
      inline TSBox tsInflate(const box3f &b)
      {
        TSBox r;
        if (b.lower.x > b.upper.x ||
            b.lower.y > b.upper.y ||
            b.lower.z > b.upper.z) {
          r.lower = vec3f(+FLT_MAX); r.upper = vec3f(-FLT_MAX);
          return r;
        }
        const vec3f ext = b.upper - b.lower;
        const float maxExt = std::max(ext.x,std::max(ext.y,ext.z));
        const float maxAbs
          = std::max(std::max(std::max(fabsf(b.lower.x),fabsf(b.lower.y)),
                              std::max(fabsf(b.lower.z),fabsf(b.upper.x))),
                     std::max(fabsf(b.upper.y),fabsf(b.upper.z)));
        const float eps = 1e-4f*(maxExt+maxAbs) + 1e-6f;
        r.lower = b.lower - vec3f(eps);
        r.upper = b.upper + vec3f(eps);
        // clamp to finite values (an 'infinite' box is +-FLT_MAX)
        r.lower = max(r.lower,vec3f(-FLT_MAX));
        r.upper = min(r.upper,vec3f(+FLT_MAX));
        return r;
      }

      inline TSBox tsUnion(TSBox a, const TSBox &b)
      {
        a.lower = min(a.lower,b.lower);
        a.upper = max(a.upper,b.upper);
        return a;
      }

      // ==================================================================
      // wire formats
      // ==================================================================

      enum {
        TS_FLAG_IN_MEDIUM = 1,
        TS_FLAG_SPECULAR  = 2,
        TS_FLAG_SHADOW    = 4,
        TS_FLAG_DBG       = 8,
        TS_FLAG_CROSSHAIR = 16
      };

      /*! same content (and size) as native::RayOnly */
      template<typename Seed>
      struct RayWireFull {
        vec3f    org;
        vec3f    dir;
        float    tMax;
        uint32_t flags;
        Seed     rngSeed;
      };

      /*! 24-byte ray record: direction octahedral 2x16 bit, RNG state
        hashed to 27 bits (+5 flag bits) */
      struct RayWireSmall {
        vec3f    org;
        uint32_t dirOct;
        float    tMax;
        uint32_t seedFlags;
      };

      /*! same content (and size) as native::HitOnly */
      template<typename Data, typename NormalT>
      struct HitWireFull {
        float    tHit;
        vec3f    P;
        NormalT  N;
        uint16_t bsdfType;
        Data     hitBSDF;
      };

      /*! normal (2x14 bit octahedral, or 'zero normal' code) and bsdf
        type (4 bits) packed into a single 32-bit word */
      template<typename Data>
      struct HitWireSmall {
        float    tHit;
        vec3f    P;
        uint32_t nt;
        Data     hitBSDF;
      };

      /*! sparse hit record: 'idx' is the position of the ray in the
        list it was received with (i.e., relative to the source
        segment) */
      template<typename Hit>
      struct HitEntry {
        uint32_t idx;
        Hit      hit;
      };

      enum { TS_N_BITS = 14, TS_N_MASK = (1u<<28)-1u, TS_N_ZERO = TS_N_MASK };

      template<typename Env>
      struct WireFull {
        typedef typename Env::Ray      Ray;
        typedef RayWireFull<typename Env::Seed> RayW;
        typedef HitWireFull<typename Env::BSDFData,typename Env::NormalT> HitW;
        static const char *name() { return "full"; }

        static inline __rtc_both void encodeRay(RayW &w, const Ray &r)
        {
          w.org  = r.org;
          w.dir  = r.dir;
          w.tMax = r.tMax;
          w.flags
            = (r.isInMedium  ? TS_FLAG_IN_MEDIUM : 0)
            | (r.isSpecular  ? TS_FLAG_SPECULAR  : 0)
            | (r.isShadowRay ? TS_FLAG_SHADOW    : 0)
            | (r._dbg        ? TS_FLAG_DBG       : 0)
            | (r.crosshair   ? TS_FLAG_CROSSHAIR : 0);
          w.rngSeed = r.rngSeed;
        }
        static inline __rtc_both void rayGeom(const RayW &w,
                                              vec3f &org, vec3f &dir, float &tMax)
        { org = w.org; dir = w.dir; tMax = w.tMax; }
        static inline __rtc_both float tMaxOf(const RayW &w) { return w.tMax; }
        static inline __rtc_both void decodeRay(Ray &r, const RayW &w)
        {
          r.org         = w.org;
          r.dir         = w.dir;
          r.tMax        = w.tMax;
          r.isInMedium  = (w.flags & TS_FLAG_IN_MEDIUM) ? 1 : 0;
          r.isSpecular  = (w.flags & TS_FLAG_SPECULAR)  ? 1 : 0;
          r.isShadowRay = (w.flags & TS_FLAG_SHADOW)    ? 1 : 0;
          r._dbg        = (w.flags & TS_FLAG_DBG)       ? 1 : 0;
          r.crosshair   = (w.flags & TS_FLAG_CROSSHAIR) ? 1 : 0;
          r.rngSeed     = w.rngSeed;
          r.bsdfType    = 0; /* == PackedBSDF::NONE */
        }
        static inline __rtc_both void makeHit(HitW &h, const Ray &r)
        {
          h.tHit     = r.tMax;
          h.P        = r.P;
          h.N        = r.N;
          h.bsdfType = r.bsdfType;
          h.hitBSDF  = r.hitBSDF;
        }
        static inline __rtc_both void makeNoHit(HitW &h, float tMax)
        {
          h.tHit     = tMax;
          h.P        = vec3f(0.f);
          h.N        = vec3f(0.f);
          h.bsdfType = 0;
        }
        static inline __rtc_both float tHitOf(const HitW &h) { return h.tHit; }
        static inline __rtc_both void applyHit(Ray &r, const HitW &h)
        {
          r.tMax     = h.tHit;
          r.bsdfType = h.bsdfType;
          r.hitBSDF  = h.hitBSDF;
          r.P        = h.P;
          r.N        = h.N;
        }
      };

      template<typename Env>
      struct WireSmall {
        typedef typename Env::Ray      Ray;
        typedef RayWireSmall           RayW;
        typedef HitWireSmall<typename Env::BSDFData> HitW;
        static const char *name() { return "small"; }

        static inline __rtc_both void encodeRay(RayW &w, const Ray &r)
        {
          w.org    = r.org;
          w.dirOct = tsOctEncode(r.dir,16);
          w.tMax   = r.tMax;
          const uint32_t flags
            = (r.isInMedium  ? TS_FLAG_IN_MEDIUM : 0)
            | (r.isShadowRay ? TS_FLAG_SHADOW    : 0)
            | (r.isSpecular  ? TS_FLAG_SPECULAR  : 0)
            | (r._dbg        ? TS_FLAG_DBG       : 0)
            | (r.crosshair   ? TS_FLAG_CROSSHAIR : 0);
          w.seedFlags = (tsSeedHash27(r.rngSeed.state) << 5) | flags;
        }
        static inline __rtc_both void rayGeom(const RayW &w,
                                              vec3f &org, vec3f &dir, float &tMax)
        { org = w.org; dir = tsOctDecode(w.dirOct,16); tMax = w.tMax; }
        static inline __rtc_both float tMaxOf(const RayW &w) { return w.tMax; }
        static inline __rtc_both void decodeRay(Ray &r, const RayW &w)
        {
          r.org         = w.org;
          r.dir         = tsOctDecode(w.dirOct,16);
          r.tMax        = w.tMax;
          r.isInMedium  = (w.seedFlags & TS_FLAG_IN_MEDIUM) ? 1 : 0;
          r.isSpecular  = (w.seedFlags & TS_FLAG_SPECULAR)  ? 1 : 0;
          r.isShadowRay = (w.seedFlags & TS_FLAG_SHADOW)    ? 1 : 0;
          r._dbg        = (w.seedFlags & TS_FLAG_DBG)       ? 1 : 0;
          r.crosshair   = (w.seedFlags & TS_FLAG_CROSSHAIR) ? 1 : 0;
          r.rngSeed.state = tsSeedExpand(w.seedFlags >> 5);
          r.bsdfType    = 0; /* == PackedBSDF::NONE */
        }
        static inline __rtc_both uint32_t packNT(vec3f N, int type)
        {
          uint32_t n;
          if (N.x == 0.f && N.y == 0.f && N.z == 0.f)
            n = TS_N_ZERO;
          else
            // never produces the all-ones code (see tsOctDecode)
            n = tsOctEncode(N,TS_N_BITS);
          return n | ((uint32_t)(type & 0xf) << 28);
        }
        static inline __rtc_both vec3f unpackN(uint32_t nt)
        {
          const uint32_t n = nt & TS_N_MASK;
          return (n == TS_N_ZERO) ? vec3f(0.f) : tsOctDecode(n,TS_N_BITS);
        }
        static inline __rtc_both void makeHit(HitW &h, const Ray &r)
        {
          h.tHit    = r.tMax;
          h.P       = r.P;
          h.nt      = packNT((vec3f)r.N,(int)r.bsdfType);
          h.hitBSDF = r.hitBSDF;
        }
        static inline __rtc_both void makeNoHit(HitW &h, float tMax)
        {
          h.tHit = tMax;
          h.P    = vec3f(0.f);
          h.nt   = TS_N_ZERO;
        }
        static inline __rtc_both float tHitOf(const HitW &h) { return h.tHit; }
        static inline __rtc_both void applyHit(Ray &r, const HitW &h)
        {
          r.tMax     = h.tHit;
          r.bsdfType = (h.nt >> 28);
          r.hitBSDF  = h.hitBSDF;
          r.P        = h.P;
          r.N        = unpackN(h.nt);
        }
      };

      // ==================================================================
      // configuration / topology / statistics
      // ==================================================================

      struct Topology {
        int numHosts    = 1;
        int gpusPerHost = 1;
        int hostIdx     = 0;
        int gpuIdx      = 0;
        /*! communicator over all ranks (one GPU per rank) */
        MPI_Comm world  = MPI_COMM_NULL;
        /*! world rank of logical GPU (h,g), at [h*gpusPerHost+g] */
        std::vector<int> rankOf;
      };

      struct Config {
        bool cull         = false;
        bool compactHits  = false;
        bool smallPayload = false;
        /*! use MPI collectives (I)alltoall(v) instead of point-to-point */
        bool collectives  = false;
        int  numChunks    = 1;
        /*! print aggregated statistics every N traceRays calls (0=off) */
        int  statsInterval = 0;
      };

      struct Stats {
        enum { OWNED=0, SENT_X, RECV_X, SENT_I, TRACED, HITS_I, HITS_X,
               BYTES_SENT, CALLS, NUM };
        double v[NUM] = {0};
      };

      enum { TS_MAX_CHUNKS = 64 };

      // ==================================================================
      // the engine
      // ==================================================================

      /*! Env must provide:
          typedefs Ray, Seed (with uint64_t 'state'), BSDFData, NormalT, Fence
          void *alloc(size_t); void free(void*);
          void *allocHost(size_t); void freeHost(void*);
          void copyAsync(void *dst, const void *src, size_t);
          void memsetAsync(void *, int, size_t);
          template<typename F> void forEach(int n, const F &f);  // async
          Fence *createFence(); void destroyFence(Fence*);
          void record(Fence*); bool query(Fence*); void wait(Fence*);
          void traceAsync(Ray *rays, int n, Fence *doneFence);
          void syncAll();
      */
      template<typename Env>
      struct Engine {
        typedef typename Env::Ray   Ray;
        typedef typename Env::Fence Fence;

        enum Step {
          S_A_PREP=0, S_A_CNT, S_A_DATA,
          S_B_PREP, S_B_CNT, S_B_DATA,
          S_TRACE, S_HITS,
          S_HB_CNT, S_HB_DATA, S_HB_REDUCE,
          S_HA_CNT, S_HA_DATA, S_HA_REDUCE,
          S_DONE
        };
        enum Wait { W_IDLE=0, W_GPU, W_MPI };

        struct DevBuf {
          void  *ptr   = nullptr;
          size_t bytes = 0;
        };

        struct Chunk {
          int   step = S_DONE;
          int   wait = W_IDLE;
          Fence *fence = nullptr;
          std::vector<MPI_Request> reqs;
          // owner range
          int b0 = 0, n0 = 0;
          int nX = 0, nY = 0;
          // host-side (pinned) counters, see layout in hostCounters()
          int *hc = nullptr;
          // prefix offsets (host)
          std::vector<int> rOff1, rOff2, off2, off1;
          // scratch for collectives (must stay alive until completion)
          std::vector<int> sc, sd, rc, rd;
        };

        Engine(Env &env, const Topology &topo, const Config &cfg)
          : env(env), topo(topo), cfg(cfg),
            H(topo.numHosts), G(topo.gpusPerHost),
            hI(topo.hostIdx), gI(topo.gpuIdx),
            K(std::max(1,std::min((int)TS_MAX_CHUNKS,cfg.numChunks)))
        {
          if ((int)topo.rankOf.size() != H*G)
            throw std::runtime_error("ts_opt: invalid topology");
          MPI_Comm_rank(topo.world,&worldRank);
          MPI_Comm_size(topo.world,&worldSize);
          // cross-node comm: all GPUs with same local index, ordered by host
          MPI_Comm_split(topo.world,gI,hI,&xComm);
          // intra-node comm: all GPUs of the same host, ordered by gpu
          MPI_Comm_split(topo.world,hI,gI,&iComm);
          int r;
          MPI_Comm_rank(xComm,&r);
          if (r != hI) throw std::runtime_error("ts_opt: unexpected cross-node comm rank");
          MPI_Comm_rank(iComm,&r);
          if (r != gI) throw std::runtime_error("ts_opt: unexpected intra-node comm rank");
          if (cfg.collectives) {
            // one communicator per exchange step, so that
            // non-blocking collectives of different chunks/steps can
            // never be matched incorrectly (per comm they are always
            // posted in chunk order).
            for (int s=0;s<S_DONE;s++) {
              stepComm[s] = MPI_COMM_NULL;
              if (isMpiStep(s))
                MPI_Comm_dup(isIntraStep(s) ? iComm : xComm,&stepComm[s]);
            }
          } else {
            MPI_Comm_dup(xComm,&xP2P);
            MPI_Comm_dup(iComm,&iP2P);
          }
          chunks.resize(K);
          for (auto &c : chunks) {
            c.fence = env.createFence();
            c.hc    = (int*)env.allocHost(numHostCounters()*sizeof(int));
          }
          boxesD.bytes = 0;
        }

        ~Engine()
        {
          for (auto &c : chunks) {
            env.destroyFence(c.fence);
            env.freeHost(c.hc);
          }
          for (auto b : allBufs()) if (b->ptr) env.free(b->ptr);
          for (auto &t : mpiTypes) MPI_Type_free(&t.second);
          if (cfg.collectives) {
            for (int s=0;s<S_DONE;s++)
              if (stepComm[s] != MPI_COMM_NULL) MPI_Comm_free(&stepComm[s]);
          } else {
            MPI_Comm_free(&xP2P);
            MPI_Comm_free(&iP2P);
          }
          MPI_Comm_free(&xComm);
          MPI_Comm_free(&iComm);
        }

        static bool isMpiStep(int s)
        {
          return s == S_A_CNT || s == S_A_DATA || s == S_B_CNT || s == S_B_DATA
            || s == S_HB_CNT || s == S_HB_DATA || s == S_HA_CNT || s == S_HA_DATA;
        }
        static bool isIntraStep(int s)
        { return s == S_B_CNT || s == S_B_DATA || s == S_HB_CNT || s == S_HB_DATA; }

        // host counter layout per chunk:
        // [cnt1:H][rcnt1:H][cnt2:G][rcnt2:G][hcB:G][rhcB:G][hcA:H][rhcA:H]
        int numHostCounters() const { return 4*H+4*G; }
        int *cnt1 (Chunk &c) { return c.hc; }
        int *rcnt1(Chunk &c) { return c.hc+H; }
        int *cnt2 (Chunk &c) { return c.hc+2*H; }
        int *rcnt2(Chunk &c) { return c.hc+2*H+G; }
        int *hcB  (Chunk &c) { return c.hc+2*H+2*G; }
        int *rhcB (Chunk &c) { return c.hc+2*H+3*G; }
        int *hcA  (Chunk &c) { return c.hc+2*H+4*G; }
        int *rhcA (Chunk &c) { return c.hc+3*H+4*G; }
        // device counter layout per chunk: [cnt1:H][cnt2:G][hcB:G][hcA:H]
        int numDevCounters() const { return 2*H+2*G; }

        static int chunkBegin(int n, int c, int K)
        { return (int)(((int64_t)n * c) / K); }
        int chunkSize(int n, int c) const
        { return chunkBegin(n,c+1,K) - chunkBegin(n,c,K); }

        /*! trace all 'numActive' rays in 'queue' globally; results are
            reduced into the queue in place (exactly like
            TwoStage::traceRays does). 'localBounds' must cover all
            data this rank's traceAsync() can hit; it is only used if
            culling is enabled (an empty box means 'no data'). */
        void traceRays(Ray *queue, int numActive, const box3f &localBounds)
        {
          if (cfg.smallPayload) run<WireSmall<Env>>(queue,numActive,localBounds);
          else                  run<WireFull<Env>>(queue,numActive,localBounds);
        }

        // ------------------------------------------------------------------
        // everything below is 'public' only because CUDA extended
        // lambdas require public enclosing functions
        // ------------------------------------------------------------------

        struct RankInfo {
          TSBox box;
          int   numActive;
          int   pad;
        };

        template<typename W>
        void run(Ray *queue, int numActive, const box3f &localBounds)
        {
          this->queue = queue;
          // --- per-call all-gather of bounds and ray counts ---
          RankInfo mine;
          mine.box = tsInflate(localBounds);
          mine.numActive = numActive;
          mine.pad = 0;
          allInfo.resize(worldSize);
          MPI_Allgather(&mine,sizeof(RankInfo),MPI_BYTE,
                        allInfo.data(),sizeof(RankInfo),MPI_BYTE,
                        topo.world);
          int maxActive = 0;
          for (auto &ri : allInfo) maxActive = std::max(maxActive,ri.numActive);
          cap0 = std::max(1,(maxActive+K-1)/K);
          capX = H*cap0;
          capY = G*capX;

          // per-destination boxes: [0..H) host boxes, [H..H+G) gpu boxes
          // of our own host
          hostBoxes.resize(H+G);
          for (int h=0;h<H;h++) {
            TSBox u; u.lower = vec3f(+FLT_MAX); u.upper = vec3f(-FLT_MAX);
            for (int g=0;g<G;g++)
              u = tsUnion(u,allInfo[topo.rankOf[h*G+g]].box);
            hostBoxes[h] = u;
          }
          for (int g=0;g<G;g++)
            hostBoxes[H+g] = allInfo[topo.rankOf[hI*G+g]].box;

          allocBuffers<W>();
          if (cfg.cull)
            env.copyAsync(boxesD.ptr,hostBoxes.data(),(H+G)*sizeof(TSBox));

          for (int c=0;c<K;c++) {
            Chunk &ch = chunks[c];
            ch.step = S_A_PREP;
            ch.wait = W_IDLE;
            ch.reqs.clear();
            ch.b0 = chunkBegin(numActive,c,K);
            ch.n0 = chunkSize(numActive,c);
            ch.nX = ch.nY = 0;
          }
          for (int s=0;s<S_DONE;s++) postedCount[s] = 0;

          int numDone = 0;
          while (numDone < K) {
            bool progress = false;
            for (int c=0;c<K;c++) {
              Chunk &ch = chunks[c];
              if (ch.step == S_DONE) continue;
              if (ch.wait == W_GPU) {
                if (!env.query(ch.fence)) continue;
                ch.wait = W_IDLE; ch.step++; progress = true;
              } else if (ch.wait == W_MPI) {
                int flag = 0;
                if (!ch.reqs.empty())
                  MPI_Testall((int)ch.reqs.size(),ch.reqs.data(),&flag,
                              MPI_STATUSES_IGNORE);
                else
                  flag = 1;
                if (!flag) continue;
                ch.reqs.clear();
                ch.wait = W_IDLE; ch.step++; progress = true;
              }
              // start as many steps as possible without waiting
              while (ch.step != S_DONE && ch.wait == W_IDLE && canStart(c)) {
                // every mpi step counts as 'posted' (in chunk order),
                // even if in the current configuration it does not
                // need to exchange anything
                if (isMpiStep(ch.step)) postedCount[ch.step]++;
                startStep<W>(c);
                progress = true;
                if (ch.wait == W_IDLE) ch.step++;
              }
              if (ch.step == S_DONE) numDone++;
            }
            (void)progress;
          }
          env.syncAll();
          updateStats<W>();
        }

        bool canStart(int c)
        {
          Chunk &ch = chunks[c];
          if (ch.step == S_TRACE && c > 0 && chunks[c-1].step <= S_TRACE)
            // traces are launched strictly in chunk order, and only
            // once the previous one completed (the optix backend
            // re-uses its launch-param buffer)
            return false;
          if (cfg.collectives && isMpiStep(ch.step) && postedCount[ch.step] != c)
            // collectives on a given step's communicator must be
            // posted in the same (chunk) order on all ranks
            return false;
          return true;
        }

        // ------------------------------------------------------------------
        // buffers
        // ------------------------------------------------------------------
        DevBuf bkt1, idx1, W0, X, bkt2, idx2, Y, staged, hitsY,
          recvB, accX, entA, recvA, cntD, boxesD;
        std::vector<DevBuf*> allBufs()
        { return { &bkt1,&idx1,&W0,&X,&bkt2,&idx2,&Y,&staged,&hitsY,
                   &recvB,&accX,&entA,&recvA,&cntD,&boxesD }; }

        void ensure(DevBuf &b, size_t bytes)
        {
          if (bytes <= b.bytes) return;
          if (b.ptr) env.free(b.ptr);
          b.ptr   = env.alloc(bytes);
          b.bytes = bytes;
        }

        template<typename W>
        void allocBuffers()
        {
          typedef typename W::RayW RayW;
          typedef typename W::HitW HitW;
          typedef HitEntry<HitW>   Entry;
          const size_t k = K;
          const bool cull = cfg.cull, cmp = cfg.compactHits;
          if (cull) {
            ensure(bkt1,  k*H*cap0*sizeof(RayW));
            ensure(idx1,  k*H*cap0*sizeof(int));
            ensure(bkt2,  k*G*(size_t)capX*sizeof(RayW));
            ensure(idx2,  k*G*(size_t)capX*sizeof(int));
            ensure(boxesD,(H+G)*sizeof(TSBox));
          } else
            ensure(W0,    k*cap0*sizeof(RayW));
          ensure(X,       k*capX*sizeof(RayW));
          ensure(Y,       k*capY*sizeof(RayW));
          ensure(staged,  k*capY*sizeof(Ray));
          ensure(hitsY,   k*capY*(cmp ? sizeof(Entry) : sizeof(HitW)));
          ensure(recvB,   k*(size_t)G*capX*(cmp ? sizeof(Entry) : sizeof(HitW)));
          ensure(accX,    k*capX*sizeof(HitW));
          if (cmp) ensure(entA, k*capX*sizeof(Entry));
          ensure(recvA,   k*(size_t)H*cap0*(cmp ? sizeof(Entry) : sizeof(HitW)));
          ensure(cntD,    k*numDevCounters()*sizeof(int));
        }

        template<typename T> T *bufPtr(DevBuf &b, int c, size_t perChunk)
        { return ((T*)b.ptr) + (size_t)c*perChunk; }

        MPI_Datatype mpiTypeOfSize(size_t size)
        {
          for (auto &t : mpiTypes) if (t.first == size) return t.second;
          MPI_Datatype t;
          MPI_Type_contiguous((int)size,MPI_BYTE,&t);
          MPI_Type_commit(&t);
          mpiTypes.push_back({size,t});
          return t;
        }

        // ------------------------------------------------------------------
        // mpi exchange helper: 'sendBase'+'sDispl[d]' (in elements)
        // with 'sCnt[d]' elements goes to peer d; likewise for recv.
        // ------------------------------------------------------------------
        void postExchange(Chunk &ch, int c, int step, size_t elemSize,
                          const void *sendBase, const int *sCnt, const int *sDispl,
                          void *recvBase, const int *rCnt, const int *rDispl)
        {
          const bool intra = isIntraStep(step);
          const int n = intra ? G : H;
          MPI_Datatype t = (elemSize == sizeof(int)) ? MPI_INT : mpiTypeOfSize(elemSize);
          double bytes = 0.;
          for (int d=0;d<n;d++) bytes += (double)sCnt[d]*elemSize;
          stats.v[Stats::BYTES_SENT] += bytes;
          if (cfg.collectives) {
            MPI_Comm comm = stepComm[step];
            ch.sc.assign(sCnt,sCnt+n);  ch.sd.assign(sDispl,sDispl+n);
            ch.rc.assign(rCnt,rCnt+n);  ch.rd.assign(rDispl,rDispl+n);
            MPI_Request req = MPI_REQUEST_NULL;
            if (K == 1) {
              // single chunk: plain blocking collective (same as the
              // existing opt_mpi code path uses)
              MPI_Alltoallv(sendBase,ch.sc.data(),ch.sd.data(),t,
                            recvBase,ch.rc.data(),ch.rd.data(),t,comm);
            } else {
              MPI_Ialltoallv(sendBase,ch.sc.data(),ch.sd.data(),t,
                             recvBase,ch.rc.data(),ch.rd.data(),t,comm,&req);
              ch.reqs.push_back(req);
            }
          } else {
            MPI_Comm comm = intra ? iP2P : xP2P;
            const int tag = step*TS_MAX_CHUNKS + c;
            for (int d=0;d<n;d++) {
              if (rCnt[d] <= 0) continue;
              MPI_Request req;
              MPI_Irecv((char*)recvBase + (size_t)rDispl[d]*elemSize,
                        rCnt[d],t,d,tag,comm,&req);
              ch.reqs.push_back(req);
            }
            for (int d=0;d<n;d++) {
              if (sCnt[d] <= 0) continue;
              MPI_Request req;
              MPI_Isend((const char*)sendBase + (size_t)sDispl[d]*elemSize,
                        sCnt[d],t,d,tag,comm,&req);
              ch.reqs.push_back(req);
            }
          }
          ch.wait = W_MPI;
        }

        /*! exchange one int per peer: send[d] -> peer d, recv[d] <- peer d */
        void postCounts(Chunk &ch, int c, int step, const int *send, int *recv)
        {
          const int n = isIntraStep(step) ? G : H;
          std::vector<int> ones(n,1), displ(n);
          for (int d=0;d<n;d++) displ[d] = d;
          postExchange(ch,c,step,sizeof(int),send,ones.data(),displ.data(),
                       recv,ones.data(),displ.data());
          stats.v[Stats::BYTES_SENT] -= n*sizeof(int);
        }

        static void prefix(std::vector<int> &off, const int *cnt, int n, int &total)
        {
          off.resize(n);
          int s = 0;
          for (int d=0;d<n;d++) { off[d] = s; s += cnt[d]; }
          total = s;
        }

        // ------------------------------------------------------------------
        // the steps
        // ------------------------------------------------------------------
        template<typename W>
        void startStep(int c)
        {
          typedef typename W::RayW RayW;
          typedef typename W::HitW HitW;
          typedef HitEntry<HitW>   Entry;
          Chunk &ch = chunks[c];
          const bool cull = cfg.cull, cmp = cfg.compactHits;
          const int  H = this->H, G = this->G;
          const int  cap0 = this->cap0, capX = this->capX;
          const size_t capY = this->capY;
          int *dCnt = bufPtr<int>(cntD,c,numDevCounters());
          int *dCnt1 = dCnt, *dCnt2 = dCnt+H, *dHcB = dCnt+H+G, *dHcA = dCnt+H+2*G;
          RayW *X  = bufPtr<RayW>(this->X,c,capX);
          RayW *Y  = bufPtr<RayW>(this->Y,c,capY);
          Ray  *staged = bufPtr<Ray>(this->staged,c,capY);
          HitW *accX = bufPtr<HitW>(this->accX,c,capX);
          int  *idx1 = cull ? bufPtr<int>(this->idx1,c,(size_t)H*cap0) : nullptr;
          int  *idx2 = cull ? bufPtr<int>(this->idx2,c,(size_t)G*capX) : nullptr;
          const TSBox *boxes = (const TSBox *)boxesD.ptr;

          switch (ch.step) {
          // ================================================================
          // stage A: owner -> same-gpu-index peer on every host
          // ================================================================
          case S_A_PREP: {
            Ray *src = queue + ch.b0;
            const int n0 = ch.n0;
            if (cull) {
              RayW *bkt = bufPtr<RayW>(bkt1,c,(size_t)H*cap0);
              env.memsetAsync(dCnt1,0,H*sizeof(int));
              env.forEach(n0,[=] __rtc_both (int i) {
                  RayW w;
                  W::encodeRay(w,src[i]);
                  vec3f org, dir; float tMax;
                  W::rayGeom(w,org,dir,tMax);
                  for (int h=0;h<H;h++) {
                    if (!tsSegmentOverlapsBox(org,dir,tMax,boxes[h])) continue;
                    const int slot = tsAtomicInc(&dCnt1[h]);
                    bkt [(size_t)h*cap0+slot] = w;
                    idx1[(size_t)h*cap0+slot] = i;
                  }
                });
              env.copyAsync(cnt1(ch),dCnt1,H*sizeof(int));
            } else {
              RayW *w0 = bufPtr<RayW>(W0,c,cap0);
              env.forEach(n0,[=] __rtc_both (int i) {
                  W::encodeRay(w0[i],src[i]);
                });
              for (int h=0;h<H;h++) cnt1(ch)[h] = n0;
            }
            env.record(ch.fence);
            ch.wait = W_GPU;
          } break;
          case S_A_CNT: {
            if (cull)
              postCounts(ch,c,S_A_CNT,cnt1(ch),rcnt1(ch));
            else
              for (int h=0;h<H;h++)
                rcnt1(ch)[h] = chunkSize(allInfo[topo.rankOf[h*G+gI]].numActive,c);
          } break;
          case S_A_DATA: {
            prefix(ch.rOff1,rcnt1(ch),H,ch.nX);
            std::vector<int> sd(H);
            for (int h=0;h<H;h++) sd[h] = cull ? h*cap0 : 0;
            postExchange(ch,c,S_A_DATA,sizeof(RayW),
                         cull ? (void*)bufPtr<RayW>(bkt1,c,(size_t)H*cap0)
                         : (void*)bufPtr<RayW>(W0,c,cap0),
                         cnt1(ch),sd.data(),
                         X,rcnt1(ch),ch.rOff1.data());
          } break;
          // ================================================================
          // stage B: forward to the gpus of our own host
          // ================================================================
          case S_B_PREP: {
            const int nX = ch.nX;
            if (cull) {
              RayW *bkt = bufPtr<RayW>(bkt2,c,(size_t)G*capX);
              env.memsetAsync(dCnt2,0,G*sizeof(int));
              env.forEach(nX,[=] __rtc_both (int x) {
                  const RayW w = X[x];
                  vec3f org, dir; float tMax;
                  W::rayGeom(w,org,dir,tMax);
                  for (int g=0;g<G;g++) {
                    if (!tsSegmentOverlapsBox(org,dir,tMax,boxes[H+g])) continue;
                    const int slot = tsAtomicInc(&dCnt2[g]);
                    bkt [(size_t)g*capX+slot] = w;
                    idx2[(size_t)g*capX+slot] = x;
                  }
                });
              env.copyAsync(cnt2(ch),dCnt2,G*sizeof(int));
              env.record(ch.fence);
              ch.wait = W_GPU;
            } else {
              for (int g=0;g<G;g++) cnt2(ch)[g] = nX;
            }
          } break;
          case S_B_CNT: {
            if (cull)
              postCounts(ch,c,S_B_CNT,cnt2(ch),rcnt2(ch));
            else
              for (int g=0;g<G;g++) {
                int s = 0;
                for (int h=0;h<H;h++)
                  s += chunkSize(allInfo[topo.rankOf[h*G+g]].numActive,c);
                rcnt2(ch)[g] = s;
              }
          } break;
          case S_B_DATA: {
            prefix(ch.rOff2,rcnt2(ch),G,ch.nY);
            std::vector<int> sd(G);
            for (int g=0;g<G;g++) sd[g] = cull ? g*capX : 0;
            postExchange(ch,c,S_B_DATA,sizeof(RayW),
                         cull ? (void*)bufPtr<RayW>(bkt2,c,(size_t)G*capX) : (void*)X,
                         cnt2(ch),sd.data(),
                         Y,rcnt2(ch),ch.rOff2.data());
          } break;
          // ================================================================
          // local trace
          // ================================================================
          case S_TRACE: {
            const int nY = ch.nY;
            env.forEach(nY,[=] __rtc_both (int y) {
                W::decodeRay(staged[y],Y[y]);
              });
            env.traceAsync(staged,nY,ch.fence);
            ch.wait = W_GPU;
          } break;
          case S_HITS: {
            if (cmp) {
              Entry *ent = bufPtr<Entry>(hitsY,c,capY);
              env.memsetAsync(dHcB,0,G*sizeof(int));
              for (int g=0;g<G;g++) {
                const int base = ch.rOff2[g];
                int *ctr = dHcB+g;
                env.forEach(rcnt2(ch)[g],[=] __rtc_both (int j) {
                    const int y = base+j;
                    if (!(staged[y].tMax < W::tMaxOf(Y[y]))) return;
                    const int slot = tsAtomicInc(ctr);
                    Entry &e = ent[base+slot];
                    e.idx = j;
                    W::makeHit(e.hit,staged[y]);
                  });
              }
              env.copyAsync(hcB(ch),dHcB,G*sizeof(int));
            } else {
              HitW *hits = bufPtr<HitW>(hitsY,c,capY);
              env.forEach(ch.nY,[=] __rtc_both (int y) {
                  W::makeHit(hits[y],staged[y]);
                });
            }
            env.record(ch.fence);
            ch.wait = W_GPU;
          } break;
          // ================================================================
          // hits back to the gpus of our own host, reduce per X-ray
          // ================================================================
          case S_HB_CNT: {
            if (cmp) postCounts(ch,c,S_HB_CNT,hcB(ch),rhcB(ch));
          } break;
          case S_HB_DATA: {
            int total;
            if (cmp) {
              prefix(ch.off2,rhcB(ch),G,total);
              postExchange(ch,c,S_HB_DATA,sizeof(Entry),
                           bufPtr<Entry>(hitsY,c,capY),hcB(ch),ch.rOff2.data(),
                           bufPtr<Entry>(recvB,c,(size_t)G*capX),rhcB(ch),ch.off2.data());
            } else {
              prefix(ch.off2,cnt2(ch),G,total);
              postExchange(ch,c,S_HB_DATA,sizeof(HitW),
                           bufPtr<HitW>(hitsY,c,capY),rcnt2(ch),ch.rOff2.data(),
                           bufPtr<HitW>(recvB,c,(size_t)G*capX),cnt2(ch),ch.off2.data());
            }
          } break;
          case S_HB_REDUCE: {
            env.forEach(ch.nX,[=] __rtc_both (int x) {
                W::makeNoHit(accX[x],W::tMaxOf(X[x]));
              });
            for (int g=0;g<G;g++) {
              // one launch per source gpu, in gpu order: within a
              // launch every target is hit at most once, and launches
              // are ordered on the stream -> deterministic, same
              // tie-breaking as the default path
              const int off = ch.off2[g];
              int  *ilist = cull ? idx2 + (size_t)g*capX : nullptr;
              if (cmp) {
                const Entry *in = bufPtr<Entry>(recvB,c,(size_t)G*capX) + off;
                env.forEach(rhcB(ch)[g],[=] __rtc_both (int k) {
                    const Entry &e = in[k];
                    const int x = ilist ? ilist[e.idx] : (int)e.idx;
                    if (W::tHitOf(e.hit) < W::tHitOf(accX[x])) accX[x] = e.hit;
                  });
              } else {
                const HitW *in = bufPtr<HitW>(recvB,c,(size_t)G*capX) + off;
                env.forEach(cnt2(ch)[g],[=] __rtc_both (int j) {
                    const int x = ilist ? ilist[j] : j;
                    if (W::tHitOf(in[j]) < W::tHitOf(accX[x])) accX[x] = in[j];
                  });
              }
            }
            if (cmp) {
              Entry *ent = bufPtr<Entry>(entA,c,capX);
              env.memsetAsync(dHcA,0,H*sizeof(int));
              for (int h=0;h<H;h++) {
                const int base = ch.rOff1[h];
                int *ctr = dHcA+h;
                env.forEach(rcnt1(ch)[h],[=] __rtc_both (int j) {
                    const int x = base+j;
                    if (!(W::tHitOf(accX[x]) < W::tMaxOf(X[x]))) return;
                    const int slot = tsAtomicInc(ctr);
                    Entry &e = ent[base+slot];
                    e.idx = j;
                    e.hit = accX[x];
                  });
              }
              env.copyAsync(hcA(ch),dHcA,H*sizeof(int));
            }
            env.record(ch.fence);
            ch.wait = W_GPU;
          } break;
          // ================================================================
          // hits back to the owners, reduce into the ray queue
          // ================================================================
          case S_HA_CNT: {
            if (cmp) postCounts(ch,c,S_HA_CNT,hcA(ch),rhcA(ch));
          } break;
          case S_HA_DATA: {
            int total;
            if (cmp) {
              prefix(ch.off1,rhcA(ch),H,total);
              postExchange(ch,c,S_HA_DATA,sizeof(Entry),
                           bufPtr<Entry>(entA,c,capX),hcA(ch),ch.rOff1.data(),
                           bufPtr<Entry>(recvA,c,(size_t)H*cap0),rhcA(ch),ch.off1.data());
            } else {
              prefix(ch.off1,cnt1(ch),H,total);
              postExchange(ch,c,S_HA_DATA,sizeof(HitW),
                           accX,rcnt1(ch),ch.rOff1.data(),
                           bufPtr<HitW>(recvA,c,(size_t)H*cap0),cnt1(ch),ch.off1.data());
            }
          } break;
          case S_HA_REDUCE: {
            Ray *dst = queue + ch.b0;
            for (int h=0;h<H;h++) {
              const int off = ch.off1[h];
              int  *ilist = cull ? idx1 + (size_t)h*cap0 : nullptr;
              if (cmp) {
                const Entry *in = bufPtr<Entry>(recvA,c,(size_t)H*cap0) + off;
                env.forEach(rhcA(ch)[h],[=] __rtc_both (int k) {
                    const Entry &e = in[k];
                    const int i = ilist ? ilist[e.idx] : (int)e.idx;
                    if (W::tHitOf(e.hit) < dst[i].tMax) W::applyHit(dst[i],e.hit);
                  });
              } else {
                const HitW *in = bufPtr<HitW>(recvA,c,(size_t)H*cap0) + off;
                env.forEach(cnt1(ch)[h],[=] __rtc_both (int j) {
                    const int i = ilist ? ilist[j] : j;
                    if (W::tHitOf(in[j]) < dst[i].tMax) W::applyHit(dst[i],in[j]);
                  });
              }
            }
            env.record(ch.fence);
            ch.wait = W_GPU;
          } break;
          default:
            throw std::runtime_error("ts_opt: invalid step");
          }
        }

        // ------------------------------------------------------------------
        // statistics
        // ------------------------------------------------------------------
        template<typename W>
        void updateStats()
        {
          stats.v[Stats::CALLS] += 1;
          for (auto &ch : chunks) {
            stats.v[Stats::OWNED] += ch.n0;
            for (int h=0;h<H;h++) stats.v[Stats::SENT_X] += cnt1(ch)[h];
            stats.v[Stats::RECV_X] += ch.nX;
            for (int g=0;g<G;g++) stats.v[Stats::SENT_I] += cnt2(ch)[g];
            stats.v[Stats::TRACED] += ch.nY;
            if (cfg.compactHits) {
              for (int g=0;g<G;g++) stats.v[Stats::HITS_I] += hcB(ch)[g];
              for (int h=0;h<H;h++) stats.v[Stats::HITS_X] += hcA(ch)[h];
            } else {
              stats.v[Stats::HITS_I] += ch.nY;
              stats.v[Stats::HITS_X] += ch.nX;
            }
          }
          if (cfg.statsInterval <= 0) return;
          if (((long long)stats.v[Stats::CALLS]) % cfg.statsInterval != 0) return;
          Stats sum;
          MPI_Reduce(stats.v,sum.v,Stats::NUM,MPI_DOUBLE,MPI_SUM,0,topo.world);
          if (worldRank == 0) {
            const double calls = std::max(1.,stats.v[Stats::CALLS]);
            const double owned = std::max(1.,sum.v[Stats::OWNED]);
            std::stringstream ss;
            ss << "#bn.ts_opt stats (all ranks, per call; wire=" << W::name()
               << " cull=" << cfg.cull << " compactHits=" << cfg.compactHits
               << " chunks=" << K << " collectives=" << cfg.collectives << "):"
               << " owned=" << sum.v[Stats::OWNED]/calls
               << " sentX=" << sum.v[Stats::SENT_X]/calls
               << " (x" << sum.v[Stats::SENT_X]/owned << ")"
               << " sentI=" << sum.v[Stats::SENT_I]/calls
               << " traced=" << sum.v[Stats::TRACED]/calls
               << " (x" << sum.v[Stats::TRACED]/owned << ")"
               << " hitRecsI=" << sum.v[Stats::HITS_I]/calls
               << " hitRecsX=" << sum.v[Stats::HITS_X]/calls
               << " MB/call=" << sum.v[Stats::BYTES_SENT]/calls/(1024.*1024.)
               << std::endl;
            std::cout << ss.str();
          }
          stats = Stats();
        }

        Env              &env;
        const Topology    topo;
        const Config      cfg;
        const int H, G, hI, gI, K;
        int worldRank = 0, worldSize = 0;
        MPI_Comm xComm = MPI_COMM_NULL, iComm = MPI_COMM_NULL;
        MPI_Comm xP2P = MPI_COMM_NULL, iP2P = MPI_COMM_NULL;
        MPI_Comm stepComm[S_DONE];
        int      postedCount[S_DONE];
        std::vector<std::pair<size_t,MPI_Datatype>> mpiTypes;
        std::vector<Chunk>    chunks;
        std::vector<RankInfo> allInfo;
        std::vector<TSBox>    hostBoxes;
        Ray   *queue = nullptr;
        int    cap0 = 0, capX = 0;
        size_t capY = 0;
        Stats  stats;
      };

    } // ::ts_opt
  }
}
