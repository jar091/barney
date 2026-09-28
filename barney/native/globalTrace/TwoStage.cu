// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/native/MPIContext.h"
#include "barney/native/globalTrace/TwoStage.h"
#include "barney/native/DeviceGroup.h"
#include "barney/native/render/RayQueue.h"
#include "barney/native/FromEnv.h"
// for the optional exchange optimizations (ts_* flags):
#include "barney/native/globalTrace/TwoStageEngine.h"
#include "barney/native/render/OptixGlobals.h"
#include "barney/native/GlobalModel.h"
#include "barney/native/ModelSlot.h"
#include "barney/native/geometry/Triangles.h"
#include "barney/native/geometry/Spheres.h"
#include "barney/native/geometry/Cylinders.h"
#include "barney/native/geometry/Capsules.h"
#include "barney/native/geometry/Cones.h"
#include "barney/native/geometry/IsoSurface.h"
#include "barney/native/volume/Volume.h"

namespace BARNEY_NS {
  namespace native {

    extern void (*profHook)();
  
    std::vector<std::tuple<double,double,int,const char *>> kernelTimes;

#define TWO_STAGE_PROFILE 0
  
#if TWO_STAGE_PROFILE
# define ENTER() const double prof_t0 = getCurrentTime();
# define LEAVE(count,name)                                              \
    const double prof_t1 = getCurrentTime();                            \
    kernelTimes.push_back(std::make_tuple<double,double,int,const char *>((double)prof_t0,(double)prof_t1,(int)count,(const char *)name));
#else
# define ENTER() /* nothing */
# define LEAVE(count,name) /* nothing */
#endif
  
    int prof_rank;

    void twoStageProfHook()
    {
      std::stringstream ss;
      static double t00 = std::get<0>(kernelTimes[0]);
      for (auto kernel : kernelTimes) {
        double t0 = std::get<0>(kernel)-t00;
        double t1 = std::get<1>(kernel)-t00;
        int numItems = std::get<2>(kernel);
        const char *name = std::get<3>(kernel);

        ss << "r" << prof_rank << " [" << prettyDouble(t0) << "s.."
           << prettyDouble(t1) << "s = "
           << prettyDouble(t1-t0) << "s]: "
           << prettyNumber(numItems) << " items in "
           << name
           << " -> " << prettyDouble(1000000.f*(t1-t0)/numItems) << "s per mio items"
           << std::endl;
      }
      kernelTimes.clear();
      std::cout << ss.str();
    }
  
    __rtc_global
    void buildHitsOnly(const rtc::ComputeInterface &ci,
                       HitOnly *hitOnly,
                       Ray *rayQueue,
                       int N);
  
    __rtc_global
    void reduceReceivedHitsKernel_intraNode(const rtc::ComputeInterface &ci,
                                            HitOnly *hitOnly,
                                            int nRays,
                                            int reduceFactor)
    {
#if RTC_DEVICE_CODE
      int tid = ci.launchIndex().x;
      if (tid >= nRays) return;

      HitOnly reduced = hitOnly[tid];
      for (int peer=1;peer<reduceFactor;peer++) {
        HitOnly *hit = hitOnly+peer*nRays+tid;
      
        if (hit->tHit >= reduced.tHit) continue;
      
        reduced = *hit;
      }
      hitOnly[tid] = reduced;
#endif
    }
  
    __rtc_global
    void reduceReceivedHitsKernel_crossNodes(const rtc::ComputeInterface &ci,
                                             Ray *rayQueueThisRank,
                                             HitOnly *hitOnlyAllRanks,
                                             int nRays,
                                             int reduceFactor)
    {
#if RTC_DEVICE_CODE
      int tid = ci.launchIndex().x;
      if (tid >= nRays) return;
    
      Ray ray = rayQueueThisRank[tid];
      for (int peer=0;peer<reduceFactor;peer++) {
        HitOnly *hit = hitOnlyAllRanks+peer*nRays+tid;

        if (hit->tHit >= ray.tMax) continue;

        ray.tMax     = hit->tHit;
        ray.bsdfType = hit->bsdfType;
        ray.hitBSDF  =  hit->hitBSDF;
        ray.P        =  hit->P;
        ray.N        =  hit->N;
      }
      rayQueueThisRank[tid] = ray;
#endif
    }
  
  
    __rtc_global
    void createRayOnly(const rtc::ComputeInterface &ci,
                       RayOnly *rayOnly,
                       Ray *rayQueue,
                       int N);
    __rtc_global
    void buildStagedRayQueue(const rtc::ComputeInterface &ci,
                             Ray *rayQueue,
                             RayOnly *rayOnly,
                             int N);
  
    // ==================================================================
    // optional exchange optimizations (ts_cull, ts_compact_hits,
    // ts_small_payload, ts_pipeline); see TwoStageEngine.h. None of
    // the code below is used unless at least one of the flags is set.
    // ==================================================================

#if defined(__CUDACC__)
    template<typename F>
    __global__ void twoStageForEachKernel(int n, F f)
    {
      const int i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < n) f(i);
    }
#endif

    /*! device 'environment' for ts_opt::Engine: memory, kernel
        launches (all on the device's compute stream), fences (cuda
        events), and the local trace (same launch as
        Context::traceRaysLocally(), but asynchronous) */
    struct TwoStageEnv {
      typedef native::Ray      Ray;
      typedef RNGSeed          Seed;
      typedef PackedBSDF::Data BSDFData;
      typedef vec3h            NormalT;
      struct Fence {
#if defined(__CUDACC__)
        cudaEvent_t ev[2] = { 0, 0 };
#endif
        int n = 0;
      };

      TwoStageEnv(MPIContext *context, Device *device)
        : context(context), device(device), rtc(device->rtc)
      {}

      void *alloc(size_t n)     { return rtc->allocMem(n); }
      void  free(void *p)       { rtc->freeMem(p); }
      void *allocHost(size_t n) { return rtc->allocHost(n); }
      void  freeHost(void *p)   { rtc->freeHost(p); }
      void copyAsync(void *d, const void *s, size_t n) { rtc->copyAsync(d,s,n); }
      void memsetAsync(void *p, int v, size_t n) { rtc->memsetAsync(p,v,n); }
      void syncAll() { rtc->sync(); }

      template<typename F>
      void forEach(int n, const F &f)
      {
        if (n <= 0) return;
#if defined(__CUDACC__)
        SetActiveGPU forDuration(device);
        const int bs = 128;
        twoStageForEachKernel<<<divRoundUp(n,bs),bs,0,rtc->stream>>>(n,f);
#else
        for (int i=0;i<n;i++) f(i);
#endif
      }

      Fence *createFence()
      {
        Fence *f = new Fence;
#if defined(__CUDACC__)
        SetActiveGPU forDuration(device);
        for (int i=0;i<2;i++)
          BARNEY_CUDA_CALL(EventCreateWithFlags(&f->ev[i],cudaEventDisableTiming));
#endif
        return f;
      }
      void destroyFence(Fence *f)
      {
#if defined(__CUDACC__)
        SetActiveGPU forDuration(device);
        for (int i=0;i<2;i++)
          if (f->ev[i]) cudaEventDestroy(f->ev[i]);
#endif
        delete f;
      }
      /*! fence covering all work issued so far on the compute stream */
      void record(Fence *f)
      {
#if defined(__CUDACC__)
        SetActiveGPU forDuration(device);
        BARNEY_CUDA_CALL(EventRecord(f->ev[0],rtc->stream));
        f->n = 1;
#else
        rtc->sync();
        f->n = 0;
#endif
      }
      bool query(Fence *f)
      {
#if defined(__CUDACC__)
        SetActiveGPU forDuration(device);
        for (int i=0;i<f->n;i++) {
          cudaError_t rc = cudaEventQuery(f->ev[i]);
          if (rc == cudaErrorNotReady) return false;
          if (rc != cudaSuccess)
            throw std::runtime_error(std::string("#bn.ts_opt: cuda error ")
                                     +cudaGetErrorString(rc));
        }
#endif
        return true;
      }
      void wait(Fence *f)
      {
#if defined(__CUDACC__)
        SetActiveGPU forDuration(device);
        for (int i=0;i<f->n;i++)
          BARNEY_CUDA_CALL(EventSynchronize(f->ev[i]));
#endif
      }

      /*! trace 'n' (staged) rays against all local model slots,
          without waiting for completion; 'f' gets recorded such that
          it completes once the trace is done */
      void traceAsync(Ray *rays, int n, Fence *f)
      {
        SetActiveGPU forDuration(device);
        int numLaunches = 0;
        if (n > 0 && model) {
          for (auto slot : model->modelSlots) {
            for (auto dev : *slot->devices) {
              if (dev != device) continue;
              OptixGlobals dd;
              dd.rays     = rays;
              dd.hitIDs   = 0;
              dd.numRays  = n;
              dd.world    = slot->world->getDD(dev);
              dd.accel    = slot->getInstanceAccel(dev);
              dd.cutPlane = context->activeCutPlane;
              if (dd.accel == 0) continue;
              // the optix backend re-uses its (pinned) launch-param
              // buffer, so never have two launches in flight
              if (numLaunches > 0) rtc->sync();
              const int bs = 256;
              const int nb = divRoundUp(n,bs);
              dev->traceRays->launch(/* same (inverted) layout as in
                                        traceRaysLocally */
                                     vec2i(bs,nb),&dd);
              numLaunches++;
            }
          }
        }
        record(f);
#if defined(__CUDACC__) && defined(BARNEY_RTC_OPTIX)
        // optix trace launches run on their own (launch-param)
        // stream, not on the compute stream
        if (numLaunches > 0 && !rtc->activeTraceStreams.empty()) {
          BARNEY_CUDA_CALL(EventRecord(f->ev[1],rtc->activeTraceStreams.back()));
          f->n = 2;
        }
#endif
      }

      MPIContext  *const context;
      Device      *const device;
      rtc::Device *const rtc;
      /*! model of the current traceRays() call */
      GlobalModel *model = nullptr;
    };

    /*! copies a device data array to the host; returns false if the
        array's element size does not match T */
    template<typename T>
    static bool tsDownload(const PODData::SP &data, std::vector<T> &out)
    {
      out.clear();
      if (!data || data->count == 0) return true;
      if (owlSizeOf(data->type) != sizeof(T)) return false;
      out.resize(data->count);
      data->download(data->devices->get(0),out.data());
      return true;
    }

    /*! object-space bounds of everything in the group that the trace
        kernel can hit (surfaces AND volumes); returns false if the
        bounds of some object can not be determined (the caller then
        has to assume 'infinite' bounds). Note these are computed from
        the raw input arrays (all vertices, plus maximum radius), so
        they are conservative. */
    static bool tsGroupBounds(Group *group, box3f &bounds)
    {
      bounds = box3f();
      for (auto &vol : group->volumes) {
        if (!vol) continue;
        if (!vol->sf) return false;
        if (!vol->sf->worldBounds.empty())
          bounds.extend(vol->sf->worldBounds);
      }
      for (auto &geom : group->geoms) {
        if (!geom) continue;
        Geometry *g = geom.get();
        if (auto t = dynamic_cast<Triangles*>(g)) {
          std::vector<vec3f> v;
          if (!tsDownload(t->vertices,v)) return false;
          for (auto p : v) bounds.extend(p);
        } else if (auto sp = dynamic_cast<Spheres*>(g)) {
          std::vector<vec3f> v;
          std::vector<float> r;
          if (!tsDownload(sp->origins,v)) return false;
          if (v.empty()) continue;
          if (!tsDownload(sp->radii,r)) return false;
          float maxR = sp->radii ? 0.f : sp->defaultRadius;
          for (auto ri : r) maxR = std::max(maxR,ri);
          box3f b;
          for (auto p : v) b.extend(p);
          bounds.extend(box3f(b.lower-vec3f(maxR),b.upper+vec3f(maxR)));
        } else if (auto cy = dynamic_cast<Cylinders*>(g)) {
          std::vector<vec3f> v;
          std::vector<float> r;
          if (!tsDownload(cy->vertices,v)) return false;
          if (v.empty()) continue;
          if (!cy->radii || !tsDownload(cy->radii,r)) return false;
          float maxR = 0.f;
          for (auto ri : r) maxR = std::max(maxR,ri);
          box3f b;
          for (auto p : v) b.extend(p);
          bounds.extend(box3f(b.lower-vec3f(maxR),b.upper+vec3f(maxR)));
        } else if (auto co = dynamic_cast<Cones*>(g)) {
          std::vector<vec3f> v;
          std::vector<float> r;
          if (!tsDownload(co->vertices,v)) return false;
          if (v.empty()) continue;
          if (!co->radii || !tsDownload(co->radii,r)) return false;
          float maxR = 0.f;
          for (auto ri : r) maxR = std::max(maxR,ri);
          box3f b;
          for (auto p : v) b.extend(p);
          bounds.extend(box3f(b.lower-vec3f(maxR),b.upper+vec3f(maxR)));
        } else if (auto ca = dynamic_cast<Capsules*>(g)) {
          std::vector<vec4f> v;
          if (!tsDownload(ca->vertices,v)) return false;
          for (auto p : v) {
            const vec3f c(p.x,p.y,p.z);
            const float r = fabsf(p.w);
            bounds.extend(box3f(c-vec3f(r),c+vec3f(r)));
          }
        } else if (auto iso = dynamic_cast<IsoSurface*>(g)) {
          if (!iso->sf) return false;
          if (!iso->sf->worldBounds.empty())
            bounds.extend(iso->sf->worldBounds);
        } else
          // unknown geometry type
          return false;
      }
      return true;
    }

    struct TwoStageOpt {
      TwoStageOpt(MPIContext *context, Device *device,
                  const ts_opt::Topology &topo, const ts_opt::Config &cfg)
        : env(context,device), engine(env,topo,cfg), cfg(cfg), device(device)
      {}

      /*! world-space bounds of all data this rank's local trace can
          hit, i.e., the union over all instances (of all model slots
          on our device) of the instance-transformed group bounds.
          Cached until the model or any of its slots changes. */
      box3f localBounds(GlobalModel *model)
      {
        std::vector<std::pair<const void*,uint64_t>> sig;
        sig.push_back({model,0});
        for (auto slot : model->modelSlots)
          sig.push_back({slot.get(),slot->contentEpoch});
        if (sig == boundsSignature) return cachedBounds;

        box3f bounds;
        bool infinite = false;
        for (auto slot : model->modelSlots) {
          bool onOurDevice = false;
          for (auto dev : *slot->devices) if (dev == device) onOurDevice = true;
          if (!onOurDevice) continue;
          for (size_t i=0;i<slot->instances.groups.size() && !infinite;i++) {
            Group *group = slot->instances.groups[i].get();
            if (!group) continue;
            box3f gb;
            if (!tsGroupBounds(group,gb)) { infinite = true; break; }
            if (gb.empty()) continue;
            bounds.extend(xfmBounds(slot->instances.xfms[i],gb));
          }
        }
        if (infinite)
          bounds = box3f(vec3f(-FLT_MAX),vec3f(+FLT_MAX));
        if (FromEnv::logQueues || cfg.statsInterval > 0) {
          std::stringstream ss;
          ss << "#bn.ts_opt(r" << device->globalRank() << "): local bounds "
             << bounds << (infinite ? " (infinite: unknown object type)" : "")
             << std::endl;
          std::cout << ss.str();
        }
        boundsSignature = sig;
        cachedBounds = bounds;
        return bounds;
      }

      TwoStageEnv  env;
      ts_opt::Engine<TwoStageEnv> engine;
      const ts_opt::Config cfg;
      Device *const device;
      std::vector<std::pair<const void*,uint64_t>> boundsSignature;
      box3f cachedBounds;
    };

    TwoStage::~TwoStage()
    {
      int finalized = 0;
      MPI_Finalized(&finalized);
      // the engine frees mpi communicators; if mpi is already gone,
      // just leak it (like the default path leaks its buffers)
      if (opt && !finalized)
        delete opt;
      opt = nullptr;
    }

    void TwoStage::traceRaysOpt(GlobalModel *model,
                                uint32_t rngSeed,
                                bool needHitIDs)
    {
      assert(needHitIDs == false); // not implemented (same as default path)
      SetActiveGPU forDuration(device);
      opt->env.model = model;
      box3f bounds = opt->cfg.cull ? opt->localBounds(model) : box3f();
      opt->engine.traceRays(device->rayQueue->traceAndShadeReadQueue.rays,
                            device->rayQueue->numActive,
                            bounds);
      opt->env.model = nullptr;
    }

    TwoStage::TwoStage(MPIContext *context)
      : GlobalTraceImpl(context),
        context(context),
        world(context->world),
        topo(context->topo.get()),
        logTopo(FromEnv::logTopo),
        logQueues(FromEnv::logQueues),
        opt_mpi(FromEnv::enabled("opt_mpi"))
    {
      prof_rank = world.rank;
#if TWO_STAGE_PROFILE
      profHook = twoStageProfHook;
#endif

      if (context->devices->size() != 1)
        throw std::runtime_error
          ("twostage all2all currently only works for one device per rank");
      this->device = context->devices->get(0);
    
      if (topo->islands.size() != 1)
        throw std::runtime_error
          ("twostage all2all currently only works for a single island");

      myGID = device->globalRank();
      numGlobal = topo->allDevices.size();
      global.rayCounts.resize(numGlobal);
      // sanity check that all physical nodes have same number of GPUs
      std::map<size_t,int> gpuCountInHost;
      int numHosts = 0;
      for (int gid=0; gid<context->topo->allDevices.size(); gid++) {
        auto &dev = context->topo->allDevices[gid];
        gpuCountInHost[dev.hostNameHash]++;
        numHosts = std::max(numHosts,topo->physicalHostIndexOf[gid]+1);
      }
      gpusPerHost = gpuCountInHost.begin()->second;
      for (auto count : gpuCountInHost)
        if (count.second != gpusPerHost)
          throw std::runtime_error
            ("twostage all2all currently requires same number of GPUs on all ranks");
      assert(numHosts * gpusPerHost == context->topo->allDevices.size());

      this->hostIdx = topo->physicalHostIndexOf[myGID];
      // allows oversubscription - we enumerate based on (host:process)
      // instead of (host.physialGPU)
      this->gpuIdx = topo->rankOnHost[myGID];
      _rankOf.resize(numGlobal);

      std::vector<int> logicalGidOfRank(numGlobal);
      int myLogicalGID = this->hostIdx * gpusPerHost + this->gpuIdx;
      world.allGather(logicalGidOfRank.data(),&myLogicalGID,1);
      for (int r=0;r<numGlobal;r++)
        _rankOf[logicalGidOfRank[r]] = r;
      this->numHosts = numGlobal / gpusPerHost;

      if (opt_mpi) {
        crossNodes.comm = world.split(this->gpuIdx);
        crossNodes.rayCounts.resize(crossNodes.comm.size);
        intraNode.comm = world.split(this->hostIdx);
        intraNode.rayCounts.resize(intraNode.comm.size);
      }
    
      if (logTopo) {
        world.barrier();
        if (context->myRank() == 0) {
          std::cout << "=========== TwoStage All2all ===========" << std::endl;
          std::cout << "- num MPI ranks (w/ one gpu each) " << numGlobal << std::endl;
          std::cout << "- detected num physical hosts " << numHosts << std::endl;
          std::cout << "- detected num (active) GPUs per host " << gpusPerHost << std::endl;
          PRINT(numHosts);
          PRINT(gpusPerHost);
          for (int h=0;h<numHosts;h++)
            for (int g=0;g<gpusPerHost;g++) {
              std::cout << "- gpu on rank " << (rankOf(h,g))
                        << " is logical h" << h << "g" << g << " {"
                        << topo->toString(rankOf(h,g)) << "}" << std::endl;
            }
        }
        world.barrier();
      }

      // ------------------------------------------------------------------
      // optional exchange optimizations; if none is set, 'opt' stays
      // null and traceRays() runs the unchanged default code path
      // ------------------------------------------------------------------
      ts_opt::Config cfg;
      cfg.cull          = FromEnv::enabled("ts_cull");
      cfg.compactHits   = FromEnv::enabled("ts_compact_hits");
      cfg.smallPayload  = FromEnv::enabled("ts_small_payload");
      cfg.numChunks     = std::max(0,FromEnv::intValue("ts_pipeline",0));
      cfg.statsInterval = std::max(0,FromEnv::intValue("ts_stats",0));
      cfg.collectives   = opt_mpi;
      if (cfg.cull || cfg.compactHits || cfg.smallPayload || cfg.numChunks > 0) {
        cfg.numChunks = std::min(std::max(1,cfg.numChunks),(int)ts_opt::TS_MAX_CHUNKS);
        ts_opt::Topology tt;
        tt.numHosts    = this->numHosts;
        tt.gpusPerHost = gpusPerHost;
        tt.hostIdx     = hostIdx;
        tt.gpuIdx      = gpuIdx;
        tt.world       = world.comm;
        tt.rankOf      = _rankOf;
        opt = new TwoStageOpt(context,(Device*)device,tt,cfg);
        if (world.rank == 0)
          std::cout << "#bn.two-stage: exchange optimizations enabled:"
                    << " ts_cull=" << cfg.cull
                    << " ts_compact_hits=" << cfg.compactHits
                    << " ts_small_payload=" << cfg.smallPayload
                    << " ts_pipeline=" << cfg.numChunks << " (chunks)"
                    << " mpi=" << (cfg.collectives ? "collectives (opt_mpi)" : "point-to-point")
                    << " ts_stats=" << cfg.statsInterval
                    << std::endl;
      }
    }


    void TwoStage::ensureAllOurQueuesAreLargeEnough()
    {
      auto rtc = device->rtc;
      size_t ourRequiredQueueSize
        = device->rayQueue->size * numGlobal;
      if (ourRequiredQueueSize > currentReservedSize) {
        if (logQueues) {
          std::cout << "resizing ray queues from " << currentReservedSize
                    << " to " << ourRequiredQueueSize << std::endl;
        }
        for (int i=0;i<2;i++)
          if (raysOnly[i]) rtc->freeMem(raysOnly[i]);
        for (int i=0;i<2;i++)
          if (hitsOnly[i]) rtc->freeMem(hitsOnly[i]);
      
        if (stagedRayQueue) rtc->freeMem(stagedRayQueue);
      
        size_t N = ourRequiredQueueSize+1024;
        for (int i=0;i<2;i++)
          raysOnly[i] = (RayOnly*)rtc->allocMem(N*sizeof(RayOnly));
        for (int i=0;i<2;i++)
          hitsOnly[i] = (HitOnly*)rtc->allocMem(N*sizeof(HitOnly));
        stagedRayQueue = (Ray *)rtc->allocMem(N*sizeof(Ray));
      
        currentReservedSize = N;
      }
    }

    // step 1: have all ranks exchange which (global) device has how
    // many rays (needed to set up the send/receives)
    void TwoStage::exchangeHowManyRaysEachDeviceHas()
    {
      ENTER();

      if (opt_mpi) {
      } else {
        int myRayCount = device->rayQueue->numActive;
        world.allGather(global.rayCounts.data(),&myRayCount,1);
    
        if (logQueues)  {
          if (myGID == 0) {
            std::cout << "ray counts (" << global.rayCounts.size() << "):";
            for (auto rc : global.rayCounts) std::cout << " " << rc;
            std::cout << std::endl;
          }
        }
      }
      LEAVE(1,"exchangeHowManyRaysEachDeviceHas");
    }
  
  
    /*! in this stage we have all each GPU exchange its rays with
      all GPUs that have same phsycail ID in all OTHER ranks, but NOT
      with other GPUs in same rank
    */
    void TwoStage::sendAndReceiveRays_crossNodes()
    {
      ENTER();
      // -----------------------------------------------------------------------------
      // first, create 'raysOnly[]' array, for each local device
      // -----------------------------------------------------------------------------
      int myRayCount = device->rayQueue->numActive;
      {
        SetActiveGPU forDuration(device);
        int bs = 128;
        int nb = divRoundUp(myRayCount,bs);
        __rtc_launch(device->rtc,
                     createRayOnly,
                     nb,bs,
                     // args
                     raysOnly[0],
                     device->rayQueue->traceAndShadeReadQueue.rays,
                     myRayCount);
      }


      if (opt_mpi) {
        crossNodes.comm.allGather(crossNodes.rayCounts.data(),&myRayCount,1);

        void *sendBuf = raysOnly[0];
        int sendCount = myRayCount*sizeof(RayOnly);
        void *recvBuf = raysOnly[1];
        std::vector<int> recvCounts(crossNodes.comm.size);
        std::vector<int> recvOffsets(crossNodes.comm.size);
        int sumCounts = 0;
        for (int i=0;i<crossNodes.comm.size;i++) {
          recvOffsets[i] = sumCounts;
          recvCounts[i] = crossNodes.rayCounts[i]*sizeof(RayOnly);
          sumCounts += recvCounts[i];
        }
        crossNodes.sumRaysReceived = sumCounts / sizeof(RayOnly);
        if (logQueues) 
          printf("xchg-rays-cross r%i we have %i sumrecv %i first two counts %i %i\n",
                 myGID,myRayCount,crossNodes.sumRaysReceived,
                 crossNodes.rayCounts[0],
                 crossNodes.rayCounts[1]);
      
        device->rtc->sync();
        BN_MPI_CALL(Allgatherv(sendBuf,sendCount,
                               MPI_BYTE,
                               recvBuf,(int*)recvCounts.data(),(int*)recvOffsets.data(),
                               MPI_BYTE,
                               crossNodes.comm));
      } else {
        device->rtc->sync();
        std::vector<MPI_Request> requests;
        int recvOfs = 0;
        for (int h=0;h<numHosts;h++) {
          MPI_Request req;
          int recvCount = global.rayCounts[rankOf(h,gpuIdx)];
          if (logQueues) 
            printf("splat-cross r%i receiving %i from %i (q 0->1)\n",
                   myGID,recvCount,rankOf(h,gpuIdx));
          world.recv(rankOf(h,gpuIdx),0,raysOnly[1]+recvOfs,
                     recvCount,req);
          recvOfs += recvCount;
          requests.push_back(req);
        }
        crossNodes.sumRaysReceived = recvOfs;
        if (logQueues) 
          printf("splat-cross r%i total received %i\n",
                 myGID,crossNodes.sumRaysReceived);
    
        for (int h=0;h<numHosts;h++) {
          MPI_Request req;
          if (logQueues) 
            printf("splat-cross r%i sending %i to %i (q 0->1)\n",
                   myGID,myRayCount,rankOf(h,gpuIdx));
          world.send(rankOf(h,gpuIdx),0,raysOnly[0],myRayCount,req);
          requests.push_back(req);
        }
    
        BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
      }
      LEAVE(recvOfs,"sendAndReceiveRays_crossNodes");
    }
  

    /*! in this stage we have all each GPU exchange its rays with
      all GPUs that have same phsycail ID in all OTHER ranks, but NOT
      with other GPUs in same rank
    */
    void TwoStage::sendAndReceiveRays_intraNode()
    {
      ENTER();

      if (opt_mpi) {
        int numRaysWeHave = crossNodes.sumRaysReceived;
        intraNode.comm.allGather(intraNode.rayCounts.data(),&numRaysWeHave,1);

        void *sendBuf = raysOnly[1];
        void *recvBuf = raysOnly[0];

        int sendCount = numRaysWeHave*sizeof(RayOnly);
        std::vector<int> recvCounts(intraNode.comm.size);
        std::vector<int> recvOffsets(intraNode.comm.size);
        int sumCounts = 0;
        for (int i=0;i<intraNode.comm.size;i++) {
          recvOffsets[i] = sumCounts;
          recvCounts[i] = intraNode.rayCounts[i]*sizeof(RayOnly);
          sumCounts += recvCounts[i];
        }
        intraNode.sumRaysReceived = sumCounts / sizeof(RayOnly);

        if (logQueues) 
          printf("xchg-rays-intra r%i we have %i sumrecv %i\n",
                 myGID,numRaysWeHave,intraNode.sumRaysReceived);
        device->rtc->sync();
        BN_MPI_CALL(Allgatherv(sendBuf,sendCount,
                               MPI_BYTE,
                               recvBuf,(int*)recvCounts.data(),(int*)recvOffsets.data(),
                               MPI_BYTE,
                               intraNode.comm));
      } else {
        std::vector<MPI_Request> requests;
        int recvOfs = 0;
        for (int g=0;g<gpusPerHost;g++) {
          MPI_Request req;
          int raysOnPeer = 0;
          for (int h=0;h<numHosts;h++)
            raysOnPeer += global.rayCounts[rankOf(h,g)];
          if (logQueues) 
            printf("splat-intra r%i receiving %i from %i (q 1->0)\n",
                   myGID,raysOnPeer,rankOf(hostIdx,g));
          world.recv(rankOf(hostIdx,g),0,
                     raysOnly[0]+recvOfs,raysOnPeer,
                     req);
          recvOfs += raysOnPeer;
          requests.push_back(req);
        }
        intraNode.sumRaysReceived = recvOfs;
        if (logQueues) 
          printf("splat-intra r%i total received %i\n",
                 myGID,intraNode.sumRaysReceived);

        int numRaysWeHave = 0;
        for (int h=0;h<numHosts;h++)
          numRaysWeHave += global.rayCounts[rankOf(h,gpuIdx)];
        for (int g=0;g<gpusPerHost;g++) {
          MPI_Request req;
          if (logQueues) 
            printf("splat-intra r%i sending %i to %i (q 1->0)\n",
                   myGID,numRaysWeHave,rankOf(hostIdx,g));
          world.send(rankOf(hostIdx,g),0,
                     raysOnly[1],numRaysWeHave,
                     req);
          requests.push_back(req);
        }
        BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
      }
    
      LEAVE(recvOfs,"sendAndReceiveRays_intraNode");
    }

  

    void TwoStage::traceRays(GlobalModel *model,
                             uint32_t rngSeed,
                             bool needHitIDs) 
    {
      if (opt) {
        // any of the ts_* optimizations enabled
        traceRaysOpt(model,rngSeed,needHitIDs);
        return;
      }
      assert(needHitIDs == false); // not implemented right now
      ensureAllOurQueuesAreLargeEnough();
      exchangeHowManyRaysEachDeviceHas();
      sendAndReceiveRays_crossNodes();
      sendAndReceiveRays_intraNode();

      traceAllReceivedRays(model,rngSeed,needHitIDs);

      exchangeHits_intraNode();
      reduceHits_intraNode();
      exchangeHits_crossNodes();
      reduceHits_crossNodes();
    }


    void TwoStage::traceAllReceivedRays(GlobalModel *model,
                                        uint32_t rngSeed,
                                        bool needHitIDs)
    {
    
      SetActiveGPU forDuration(device);
      int numRaysWeHaveTotal = intraNode.sumRaysReceived;
      {
        ENTER();
        if (logQueues) 
          printf("buildlocalrays r%i total rays %i (q0)\n",
                 myGID,numRaysWeHaveTotal);
        __rtc_launch(device->rtc,
                     buildStagedRayQueue,
                     divRoundUp(numRaysWeHaveTotal,1024),1024,
                     // args
                     stagedRayQueue,
                     raysOnly[0],
                     numRaysWeHaveTotal);
      
        device->rtc->sync();
        LEAVE(numRaysWeHaveTotal,"buildStagedRayQueue");
      }
    
      auto savedOriginalRayCount = device->rayQueue->numActive;
      auto savedOriginalRayQueue = device->rayQueue->traceAndShadeReadQueue.rays;
      device->rayQueue->traceAndShadeReadQueue.rays = stagedRayQueue;
      device->rayQueue->numActive = numRaysWeHaveTotal;

      {
        ENTER()
          if (logQueues) 
            printf("localtrace r%i total rays %i\n",
                   myGID,numRaysWeHaveTotal);
        context->traceRaysLocally(model,rngSeed,needHitIDs);
        device->rtc->sync();
        LEAVE(numRaysWeHaveTotal,"localTrace");
      }
    
      if (logQueues) 
        printf("buildhits r%i total rays %i (q0)\n",
               myGID,numRaysWeHaveTotal);
      {
        ENTER();
        __rtc_launch(device->rtc,
                     buildHitsOnly,
                     divRoundUp(numRaysWeHaveTotal,1024),1024,
                     // args
                     hitsOnly[0],
                     stagedRayQueue,
                     numRaysWeHaveTotal);
        device->rtc->sync();
        LEAVE(numRaysWeHaveTotal,"buildHitsOnly");
      }
      device->rayQueue->numActive = savedOriginalRayCount;
      device->rayQueue->traceAndShadeReadQueue.rays = savedOriginalRayQueue;
    }
  
    void TwoStage::exchangeHits_intraNode()
    {
      ENTER();
      if (opt_mpi) {
        void *sendBuf = hitsOnly[0];
        void *recvBuf = hitsOnly[1];
        std::vector<int> sendOffsets(intraNode.comm.size);
        std::vector<int> sendCounts(intraNode.comm.size);
        std::vector<int> recvOffsets(intraNode.comm.size);
        std::vector<int> recvCounts(intraNode.comm.size);
        int recvSum = 0;
        int sendSum = 0;
        for (int i=0;i<intraNode.comm.size;i++) {
          int recvCount = crossNodes.sumRaysReceived;
          int sendCount = intraNode.rayCounts[i];
          recvOffsets[i] = recvSum*sizeof(HitOnly);
          sendOffsets[i] = sendSum*sizeof(HitOnly);
          recvCounts[i] = recvCount*sizeof(HitOnly);
          sendCounts[i] = sendCount*sizeof(HitOnly);
          sendSum += sendCount;
          recvSum += recvCount;
        }
        BN_MPI_CALL(Alltoallv(sendBuf,
                              (const int*)sendCounts.data(),
                              (const int*)sendOffsets.data(),
                              MPI_BYTE,
                              recvBuf,
                              (const int*)recvCounts.data(),
                              (const int*)recvOffsets.data(),
                              MPI_BYTE,
                              intraNode.comm));
      } else {
        std::vector<MPI_Request> requests;
        int recvOfs = 0;
        for (int g=0;g<gpusPerHost;g++) {
          MPI_Request req;
          int recvCount = 0;
          for (int h=0;h<numHosts;h++)
            recvCount += global.rayCounts[rankOf(h,gpuIdx)];
          world.recv(rankOf(hostIdx,g),0,
                     hitsOnly[1]+recvOfs,recvCount,req);
          if (logQueues) 
            printf("xchg-intra r%i receiving %i from %i (q0->1)\n",
                   myGID,recvCount,rankOf(hostIdx,g));
          requests.push_back(req);
          recvOfs += recvCount;
        }

        // and matching sends
        int sendOfs = 0;
        for (int g=0;g<gpusPerHost;g++) {
          int sendCount = 0;
          for (int h=0;h<numHosts;h++)
            sendCount += global.rayCounts[rankOf(h,g)];
          MPI_Request req;
          world.send(rankOf(hostIdx,g),0,
                     hitsOnly[0]+sendOfs,sendCount,req);
          if (logQueues) 
            printf("xchg-intra r%i sending %i to %i (q0->1)\n",
                   myGID,sendCount,rankOf(hostIdx,g));
          requests.push_back(req);
          sendOfs += sendCount;
        }
        BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
      }
      LEAVE(recvOfs,"exchangeHits_intraNode");
    }


    void TwoStage::reduceHits_intraNode()
    {
      ENTER();
      SetActiveGPU forDuration(device);
      int numUniqueRaysThisGPU = crossNodes.sumRaysReceived;
      // int g = gpuIdx;
      // for (int h=0;h<numHosts;h++)
      //   numUniqueRaysThisGPU += rayCounts[rankOf(h,g)];

      if (logQueues) 
        printf("r%i intra-reducing %i sets of %i hits (q1)\n",
               myGID,
               gpusPerHost,
               numUniqueRaysThisGPU);
      __rtc_launch(device->rtc,
                   reduceReceivedHitsKernel_intraNode,
                   divRoundUp(numUniqueRaysThisGPU,128),128,
                   // args
                   hitsOnly[1],
                   numUniqueRaysThisGPU,
                   gpusPerHost);
      device->rtc->sync();
      LEAVE(numUniqueRaysThisGPU*gpusPerHost,"reduceHits_intraNode");
    }
  
    void TwoStage::exchangeHits_crossNodes()
    {
      ENTER();
      if (opt_mpi) {
        int myRayCount = device->rayQueue->numActive;
        void *sendBuf = hitsOnly[1];
        void *recvBuf = hitsOnly[0];
        std::vector<int> sendOffsets(crossNodes.comm.size);
        std::vector<int> sendCounts(crossNodes.comm.size);
        std::vector<int> recvOffsets(crossNodes.comm.size);
        std::vector<int> recvCounts(crossNodes.comm.size);
        int recvSum = 0;
        int sendSum = 0;
        for (int i=0;i<crossNodes.comm.size;i++) {
          int recvCount = myRayCount;
          int sendCount = crossNodes.rayCounts[i];
          recvOffsets[i] = recvSum*sizeof(HitOnly);
          sendOffsets[i] = sendSum*sizeof(HitOnly);
          recvCounts[i] = recvCount*sizeof(HitOnly);
          sendCounts[i] = sendCount*sizeof(HitOnly);
          sendSum += sendCount;
          recvSum += recvCount;
        }
        if (logQueues) 
          printf("xchg-hits-cross r%i myRayCount %i sendSum %i recvSum %i\n",
                 myGID,myRayCount,sendSum,recvSum);
        BN_MPI_CALL(Alltoallv(sendBuf,
                              (const int*)sendCounts.data(),
                              (const int*)sendOffsets.data(),
                              MPI_BYTE,
                              recvBuf,
                              (const int*)recvCounts.data(),
                              (const int*)recvOffsets.data(),
                              MPI_BYTE,
                              crossNodes.comm));
      } else {
        std::vector<MPI_Request> requests;
        int recvOfs = 0;
        int recvCount = global.rayCounts[rankOf(hostIdx,gpuIdx)];
        for (int h=0;h<numHosts;h++) {
          MPI_Request req;

          if (logQueues) 
            printf("xchg-intra r%i receiving %i from %i (q1->0)\n",
                   myGID,recvCount,rankOf(h,gpuIdx));
          world.recv(rankOf(h,gpuIdx),0,
                     hitsOnly[0]+recvOfs,recvCount,req);
          requests.push_back(req);
          recvOfs += recvCount;
        }

        // and matching sends
        int sendOfs = 0;
        for (int h=0;h<numHosts;h++) {
          MPI_Request req;
          int sendCount = global.rayCounts[rankOf(h,gpuIdx)];
          if (logQueues) 
            printf("xchg-intra r%i sending %i to %i (q1->0)\n",
                   myGID,sendCount,rankOf(h,gpuIdx));
          world.send(rankOf(h,gpuIdx),0,
                     hitsOnly[1]+sendOfs,sendCount,req);
          requests.push_back(req);
          sendOfs += sendCount;
        }
    
        BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
      }
      LEAVE(recvOfs,"exchangeHits_crossNodes");
    }
  
    void TwoStage::reduceHits_crossNodes()
    {
      ENTER();
      SetActiveGPU forDuration(device);
      if (opt_mpi) {
        int numUniqueRaysThisGPU = device->rayQueue->numActive;
        if (logQueues) 
          printf("r%i cross-reducing %i sets of %i hits (q0)\n",
                 myGID,numHosts,numUniqueRaysThisGPU);
        __rtc_launch(device->rtc,
                     reduceReceivedHitsKernel_crossNodes,
                     divRoundUp(numUniqueRaysThisGPU,128),128,
                     // args
                     device->rayQueue->traceAndShadeReadQueue.rays,
                     hitsOnly[0],
                     numUniqueRaysThisGPU,
                     numHosts);
      } else {
        int numUniqueRaysThisGPU = global.rayCounts[rankOf(hostIdx,gpuIdx)];
        if (logQueues) 
          printf("r%i cross-reducing %i sets of %i hits (q0)\n",
                 myGID,numHosts,numUniqueRaysThisGPU);
        __rtc_launch(device->rtc,
                     reduceReceivedHitsKernel_crossNodes,
                     divRoundUp(numUniqueRaysThisGPU,128),128,
                     // args
                     device->rayQueue->traceAndShadeReadQueue.rays,
                     hitsOnly[0],
                     numUniqueRaysThisGPU,
                     numHosts);
      }
      device->rtc->sync();
      LEAVE(numUniqueRaysThisGPU*numHosts,"reduceHits_crossNodes");
    }
  
  }
}
