/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <stdio.h>
#include <string.h>

#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_tensor_util.h"
#include "ovx_controller.h"
#include "ovx_node_attr_template.h"
#include "ovx_log.h"


// if true, show id for each node
//static const bool DBG_SHOW_ID = false;

//static const uint32_t OUTPUT_PARAM_MAX_LINE_SIZE = 1000;

#define OVX_CONTROLLER_VERSION 0

// allocate print bufsize in advance @MB
//#define PRINT_BUFSIZE (2 * 1024 * 1024)

//static unsigned char s_print_buf[PRINT_BUFSIZE];

static vsi_nn_context_t s_context = NULL;
static vsi_nn_graph_t * s_graph = NULL;
static vsi_nn_tensor_id_t * s_tensor_node = NULL;

bool ovx_controller_ExecuteGraph(uint32_t graph_id) {
  bool ret;
  OVXLOGI("Execute graph.");

  ret = vsi_nn_RunGraph(s_graph);

  if (!ret) {
    OVXLOGE("Execution failed");
  }

  return ret;
}

uint32_t ovx_controller_GetTargetGraphId() {
  return 1;
}

void ovx_controller_SetTargetGraphId(uint32_t graph_id) {
    //TODO
}

void ovx_controller_PrintGraph(uint32_t id) {
}

int ovx_controller_GetWrapperVersion() {
  return OVX_CONTROLLER_VERSION;
}

int ovx_controller_GetOvxBinaryVersion() {
  int retval = 0;
  return retval;
}

bool ovx_controller_InitOvx() {
  return true;
}

bool ovx_controller_DeInitOvx() {
  OVXLOGI("Finalize ovx");
  return true;
}

// Append const tensor to the graph
bool ovx_controller_AppendConstTensor(
        const char* const name, int node_id,
        const uint32_t * const shape, uint32_t dim_num,
        uint8_t* data, int data_type) {
  vsi_nn_tensor_id_t tensor_id;
  vsi_nn_tensor_attr_t attr;
  OVXLOGI("Append const tensor %s.", name);
  memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
  memcpy(attr.size, shape, dim_num * sizeof(uint32_t));
  //TODO: Transpose
  //TODO: Data type
  attr.vtl = vx_false_e;
  attr.is_const = vx_true_e;
  attr.dtype.vx_type = VX_TYPE_FLOAT32;
  tensor_id = vsi_nn_AddTensor(s_graph, VSI_NN_TENSOR_ID_AUTO, &attr, data);
  if (VSI_NN_TENSOR_ID_NA != tensor_id) {
    OVXLOGE("Copy data to tensor %s fail.", name);
  } else {
    s_tensor_node[node_id] = tensor_id;
  }
  return true;
}

static bool _read_attr(vsi_nn_node_t* node,
        const uint8_t* const attr, int len) {
  return false;
}

void ovx_controller_parse_attrs(uint32_t node_id,
        const uint8_t* const data, int data_length) {
  vsi_nn_node_t * node = NULL;
  if (node_id < s_graph->node_num) {
    node = s_graph->nodes[node_id];
    if (false == _read_attr(node, data, data_length)) {
        OVXLOGE("Read attr fail.");
    }
  }
}

// Append node to the graph
uint32_t ovx_controller_AppendNode(
    const char* const name, int graph_id, int node_id, int op_id,
    const uint8_t* const inputs, int inputs_count,
    const uint8_t* const outputs, int outputs_count) {
  vsi_nn_node_id_t ovxnode_id = 0;
  vsi_nn_node_t * node = vsi_nn_AppendNode(s_graph, op_id, &ovxnode_id);

  if (VSI_NN_NODE_ID_NA == ovxnode_id) {
    OVXLOGE("Failed to append node %s(%d)", name, op_id);
    return ovxnode_id;
  }
  ovx_apply_node_attr_template(node);

  return ovxnode_id;
}

bool ovx_controller_ConstructGraph(uint32_t graph_id) {
  return true;
}

uint32_t ovx_controller_InstantiateGraph(
        uint32_t tensor_num, uint32_t node_num) {
  s_graph = vsi_nn_CreateGraph(s_context, tensor_num, node_num);
  if (NULL == s_graph) {
    OVXLOGE("Create graph(%d, %d) fail.", tensor_num, node_num);
  }
  return 1;
}

uint64_t ovx_controller_GetOutputNodeData(const char* node_name,
        uint8_t** buf, uint64_t* bytes) {

  OVXLOGI("Read output of %s.", node_name);
  //TODO: Find graph by name.
  //TODO: Find tensor by name.
  *bytes = vsi_nn_CopyTensorToBuffer(s_graph, s_graph->tensors[0], *buf);
  return *bytes;
}

bool ovx_controller_FillInputNode(
        const char* const name, int node_id,
        const uint32_t * const shape, uint32_t dim_num,
        const uint8_t* const buf, uint64_t buf_size) {
  return true;
}

