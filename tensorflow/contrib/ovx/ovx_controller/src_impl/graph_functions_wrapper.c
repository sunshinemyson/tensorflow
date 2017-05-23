/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// to demonstrate the performance difference between ION and HLOS memory
// for sharing with ADSP.
#define USE_ION_MEMORY

#include <limits.h>
#include <stdio.h>

#include "ovx_controller.h"
#include "ovx_nn.h"
#include "tfm_log.h"

static const uint32_t MAX_NODES = 2048;
static const uint32_t MAX_EVENT_COUNT = 256;

static const bool DUMP_OUTPUT = false;
static const bool DBG_EXECUTION = true;

static const int OUT_RANKING_SIZE = 5;

// static only for this file.
// TODO(satok): allocate dynamically
static float s_output_values[300 * 300 * 3 * 4];

extern void init_graph(uint32_t id);
extern void init_graph_v1(uint32_t id);
extern uint8_t inception_dummy_int_data_299x299[];
extern uint8_t inception_dummy_int_data_224x224[];
extern float inception_dummy_float_data_299x299[];

enum InceptionVersion {
  INCEPTION_V1,
  INCEPTION_V3,
};

static enum InceptionVersion s_inception_version = INCEPTION_V3;

/////////////////////////////////////////////////
// file local functions

static const char *ConvertGraphInfoIdToName(unsigned int id) {
  // TODO(satok): implement
  return "?";
}

static const char *ConvertGraphInfoIdToOpName(unsigned int id) {
  // TODO(satok): implement
  return "?";
}

/////////////////////////////////////////////////
// file local utilities
static uint32_t FindMaxIdxWithExcludeList(
    const float *data, uint32_t entries, const int exclude_size,
    const int* exclude_idx) {
  int i;
  float maxval = data[0];
  int maxidx = 0;
  for (i = 0; i < entries; i++) {
    bool exclude = false;
    for (int j = 0; j < exclude_size; ++j) {
      if (exclude_idx[j] == i) {
        exclude = true;
        break;
      }
    }
    if (exclude) {
      continue;
    }
    if (maxval < data[i]) {
      maxval = data[i];
      maxidx = i;
    }
  }
  return maxidx;
}

static uint32_t FindMaxIdx(const float* data, uint32_t entries) {
  return FindMaxIdxWithExcludeList(data, entries, 0, NULL);
}

void ovx_controller_PrintMaxNIdx(const float *data, const uint32_t entries,
                         const int n, int* out_ranking) {
  if (DUMP_OUTPUT) {
    for (int i = 0; i < entries; ++i) {
      TFMLogDOvx("%d: val = %f", i, data[i]);
    }
  }
  for (int i = 0; i < n; ++i) {
    out_ranking[i] = INT_MAX;
  }
  for (int i = 0; i < n; ++i) {
    out_ranking[i] = FindMaxIdxWithExcludeList(data, entries, n, out_ranking);
  }
  TFMLogDOvx("=== RANKING ===");
  for (int i = 0; i < n; ++i) {
    TFMLogDOvx("%d: id = %d, val = %f", i, out_ranking[i], data[out_ranking[i]]);
  }
}

static inline unsigned long long int GetCounter(ovx_nn_perfinfo s) {
  unsigned long long int ret;
  ret = s.counter_hi;
  ret <<= 32;
  ret |= s.counter_lo;
  return ret;
}

static int CompareCycle(const void *va, const void *vb) {
  const ovx_nn_perfinfo *a = va;
  const ovx_nn_perfinfo *b = vb;
  unsigned long long int acount = GetCounter(*a);
  unsigned long long int bcount = GetCounter(*b);
  if (acount < bcount) {
    return -1;
  } else if (acount > bcount) {
    return 1;
  } else {
    return 0;
  }
}

/////////////////////////////////////////////////
// Graph functions

uint32_t ovx_controller_InstantiateGraph() {
  const uint32_t nn_id = ovx_nn_init();
  // set debug level to 99 for now
  //ovx_nn_set_debug_level(nn_id, 99);
  // TODO(satok): make this as argument
  ovx_nn_set_debug_level(nn_id, 0);
  return nn_id;
}

void ovx_controller_InitGraph(int version, uint32_t nn_id) {
  if (version == 1) {
    s_inception_version = INCEPTION_V1;
  } else if (version == 3) {
    s_inception_version = INCEPTION_V3;
  } else {
    TFMLOGE("Unsupported inception version %d", version);
    return;
  }
  if (s_inception_version == INCEPTION_V3) {
    init_graph(nn_id);
  } else if (s_inception_version == INCEPTION_V1) {
    init_graph_v1(nn_id);
  }
  TFMLogDOvx("Init graph (inception version = %d) done.", version);
}

bool ovx_controller_ConstructGraph(uint32_t nn_id) {
  int err;
  if ((err = ovx_nn_prepare(nn_id)) != 0) {
    TFMLOGE("Prepare failed! returned 0x%x\n", err);
    return false;
  } else {
    TFMLogDOvx("Prepare success!\n");
    return true;
  }
}

uint32_t ovx_controller_SetupGraph(int version)  {
  const uint32_t nn_id = ovx_controller_InstantiateGraph();
  ovx_controller_InitGraph(version, nn_id);
  ovx_controller_ConstructGraph(nn_id);
  return nn_id;
}

bool ovx_controller_ExecuteGraph(
    const uint32_t nn_id,
    const uint32_t batches,
    const uint32_t height,
    const uint32_t width,
    const uint32_t depth,
    uint8_t* int_data,
    const uint32_t int_data_size,
    uint32_t* out_batches,
    uint32_t* out_height,
    uint32_t* out_width,
    uint32_t* out_depth,
    uint8_t* out_vals,
    const uint32_t output_val_byte_size,
    uint32_t* out_data_byte_size) {
  int err;
  if (DBG_EXECUTION) {
    TFMLogDOvx("Preparing to execute...");
    TFMLogDOvx("Input: %d, %d, %d, %d, %d, %d",
            batches, height, width, depth, int_data[0], int_data_size);
    TFMLogDOvx("Output: %d, %p", output_val_byte_size, out_vals);
    LogDOvx("Execute graph!");
  }
  
  if ((err = ovx_nn_execute(nn_id,
                                batches,
                                height,
                                width,
                                depth,
                                int_data,
                                int_data_size,
                                out_batches,
                                out_height,
                                out_width,
                                out_depth,
                                out_vals,
                                output_val_byte_size,
                                out_data_byte_size)) != 0) {
    if (DBG_EXECUTION) {
      LogDOvx("Execution failed!");
      TFMLOGE("execute got err: %d\n",err);
    }
    return false;
  } else {
    if (DBG_EXECUTION) {
      LogDOvx("Execution succeeded!");
      TFMLogDOvx("%d x %d x %d x %d, byte size = %d\n",
              *out_batches,
              *out_height,
              *out_width,
              *out_depth,
              *out_data_byte_size);
    }
    return true;
  }
}

bool ovx_controller_ExecuteInceptionDummyData(uint32_t nn_id) {
  uint32_t out_batches, out_height, out_width, out_depth;
  uint32_t out_data_size;
  // s_output_values = 300 * 300 * 3 * 4 * 4
  const bool success = ovx_controller_ExecuteGraph(
      nn_id, INCEPTION_PARAM_BATCHES, INCEPTION_PARAM_HEIGHT_V3,
      INCEPTION_PARAM_WIDTH_V3, INCEPTION_PARAM_DEPTH,
      (uint8_t *)inception_dummy_int_data_299x299,
      INCEPTION_PARAM_HEIGHT_V3 * INCEPTION_PARAM_WIDTH_V3 *
      INCEPTION_PARAM_DEPTH,
      &out_batches, &out_height, &out_width, &out_depth,
      (uint8_t *)s_output_values, sizeof(s_output_values),
      &out_data_size);
  if (success) {
    int out_ranking[OUT_RANKING_SIZE];
    ovx_controller_PrintMaxNIdx(
        s_output_values,
        out_batches * out_height * out_width * out_depth,
        OUT_RANKING_SIZE, out_ranking);
    TFMLogDOvx("%d x %d x %d x %d, size = %d\n",
            out_batches,
            out_height,
            out_width,
            out_depth,
            out_data_size);
    TFMLogDOvx("max idx: %d\n", FindMaxIdx(
        s_output_values,
        out_batches * out_height * out_width * out_depth));
    if (out_ranking[0] == 169 && out_ranking[1] == 7) {
      return true;
    } else {
      TFMLogDOvx("Result is wrong! %d, %d", out_ranking[0], out_ranking[1]);
      return false;
    }
  } else {
    return false;
  }
}

void ovx_controller_DumpPerf(uint32_t nn_id) {
  ovx_nn_perfinfo info[MAX_NODES];
  unsigned long long int total_cycles = 0;
  unsigned long long int cum_cycles = 0;
  unsigned long long int counter = 0;
  unsigned int n_nodes;
  int i;
  TFMLogDOvx("Perf dump follows:");
  if (ovx_nn_get_perfinfo(nn_id, info, MAX_NODES, &n_nodes) != 0) {
    TFMLOGE("perf info failure");
    return;
  }
  TFMLogDOvx("Total %d nodes.",n_nodes);
  qsort(info,n_nodes,sizeof(info[0]), CompareCycle);
  for (i = 0; i < n_nodes; i++) {
    total_cycles += GetCounter(info[i]);
  }
  TFMLogDOvx("Total %lld cycles.",total_cycles);
  for (i = 0; i < n_nodes; i++) {
    counter = GetCounter(info[i]);
    cum_cycles += counter;
    TFMLogDOvx("node,0x%x,%s,%s,executions,%d,cycles,%lld,%f %%,"
            "cum_cycles,%lld,%f %%\n",
           info[i].node_id,
           ConvertGraphInfoIdToName(info[i].node_id),
           ConvertGraphInfoIdToOpName(info[i].node_id),
           info[i].executions,
           counter,
           100*((double)counter)/total_cycles,
           cum_cycles,
           100*((double)cum_cycles)/total_cycles);
  }
#ifdef ENABLE_HVX_FULL_DEBUG
  DumpAllPerf(nn_id);
#endif
}

void ovx_controller_DumpNodeName(uint32_t nn_id) {
  TFMLogDOvx("Show node name");
  const uint32_t id = nn_id;
  ovx_nn_perfinfo info[MAX_NODES];
  unsigned long long int total_cycles = 0;
  unsigned long long int cum_cycles = 0;
  unsigned long long int counter = 0;
  unsigned int node_count;
  int i;
  TFMLogDOvx("Perf dump follows:");
  if (ovx_nn_get_perfinfo(id, info, MAX_NODES, &node_count) != 0) {
    TFMLogDOvx("perf info failure");
    return;
  }
  TFMLogDOvx("Total %d nodes.",node_count);
  qsort(info, node_count, sizeof(info[0]), CompareCycle);
  for (i = 0; i < node_count; i++) {
    total_cycles += GetCounter(info[i]);
  }
  TFMLogDOvx("Total %lld cycles.", total_cycles);
  for (i = 0; i < node_count; i++) {
    counter = GetCounter(info[i]);
    cum_cycles += counter;
    TFMLogDOvx("node,0x%x,%s,%s,executions,%d,cycles,%lld,%f %%,"
            "cum_cycles,%lld,%f %%",
            info[i].node_id,
            ConvertGraphInfoIdToName(info[i].node_id),
            ConvertGraphInfoIdToOpName(info[i].node_id),
            info[i].executions,
            counter,
            100*((double)counter)/total_cycles,
            cum_cycles,
            100*((double)cum_cycles)/total_cycles);
  }
}

void ovx_controller_Teardown(uint32_t nn_id) {
  ovx_nn_teardown(nn_id);
}
