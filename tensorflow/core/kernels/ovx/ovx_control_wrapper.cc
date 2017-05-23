/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
vcyou may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/ovx/ovx_control_wrapper.h"

#include "tensorflow/core/kernels/ovx/ovx_ops_definitions.h"

#ifdef USE_OVX_LIBS
#include "tensorflow/core/platform/ovx/soc_interface.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#endif

namespace tensorflow {

const bool DBG_DUMP_VERIFICATION_STRING = false;
const bool SHOW_DBG_IN_SOC = false;
const bool DBG_USE_DUMMY_INPUT = false;
const bool DBG_USE_SAMPLE_INPUT = false;
const int64 FLAG_ENABLE_PANDA_BINARY_INPUT = 0x01;
const bool DBG_DUMP_INPUT_TENSOR_AS_FLOAT_DATA = false;

/* static */ GraphTransferInfo::NodeInfo* OvxControlWrapper::FindNodeInfo(
    const string& name, GraphTransferInfo* graph_transfer_info) {
  for (GraphTransferInfo::NodeInfo& node_info :
       *graph_transfer_info->mutable_node_info()) {
    if (node_info.name() == name) {
      return &node_info;
    }
  }
  return nullptr;
}

#ifdef USE_OVX_LIBS
int OvxControlWrapper::GetVersion() {
  return soc_interface_GetSocControllerVersion();
}

bool OvxControlWrapper::Init(const RemoteFusedGraphExecuteInfo& info) {
  soc_interface_SetLogLevel(SHOW_DBG_IN_SOC ? -1 /* debug */ : 0 /* info */);
  if (DBG_USE_SAMPLE_INPUT) {
    soc_interface_SetDebugFlag(FLAG_ENABLE_PANDA_BINARY_INPUT);
  }
  graph_transferer_.SetSerializedGraphTransferInfo(
      info.serialized_executor_parameters());
  execute_info_ = &info;
  return soc_interface_Init();
}

bool OvxControlWrapper::Finalize() { return soc_interface_Finalize(); }
bool OvxControlWrapper::SetupGraph() {
  // Copy graph transfer info to modify to adapt ovx nn library
  GraphTransferInfo& graph_transfer_info =
      graph_transferer_.GetMutableGraphTransferInfo();
  // Setup node
  // Setup tensor
  // Setup graph inputs & outputs

#if 0
  // Overwrite op type of input nodes for ovx
  for (const GraphTransferInfo::GraphInputNodeInfo& graph_input :
       graph_transfer_info.graph_input_node_info()) {
    GraphTransferInfo::NodeInfo* node_info =
        FindNodeInfo(graph_input.name(), &graph_transfer_info);
    CHECK_NE(node_info, nullptr);
    node_info->set_type_name(INPUT_OP_NAME);
    node_info->set_soc_op_id(
        OvxOpsDefinitions::getInstance().GetOpIdFor(INPUT_OP_NAME));
  }

  // Generate a new output node which is connected to graph output node
  // TODO: Support multiple output nodes
  CHECK_EQ(graph_transfer_info.graph_output_node_info_size(), 1);
  for (const GraphTransferInfo::GraphOutputNodeInfo& graph_output :
       graph_transfer_info.graph_output_node_info()) {
    const int new_output_node_id = graph_transfer_info.node_info_size() +
                                   graph_transfer_info.const_node_info_size() +
                                   2 /* offset for ids */;
    // Register a new output node
    GraphTransferInfo::NodeInfo& new_output_node_info =
        *graph_transfer_info.add_node_info();
    new_output_node_info.set_name(OUTPUT_OP_NAME);
    new_output_node_info.set_node_id(new_output_node_id);
    new_output_node_info.set_type_name(OUTPUT_OP_NAME);
    new_output_node_info.set_soc_op_id(
        OvxOpsDefinitions::getInstance().GetOpIdFor(OUTPUT_OP_NAME));
    new_output_node_info.set_padding_id(0 /* PADDING_NA_ID */);
    new_output_node_info.set_input_count(1);
    new_output_node_info.set_output_count(0);

    // Register node input for the new output node
    const GraphTransferInfo::NodeInfo* node_info =
        FindNodeInfo(graph_output.name(), &graph_transfer_info);
    CHECK_NE(node_info, nullptr);
    GraphTransferInfo::NodeInputInfo& node_input_info =
        *graph_transfer_info.add_node_input_info();
    node_input_info.set_node_id(new_output_node_id);
    GraphTransferInfo::NodeInput& node_input =
        *node_input_info.add_node_input();
    node_input.set_node_id(node_info->node_id());
    node_input.set_output_port(0);
  }
#endif

  if (DBG_DUMP_VERIFICATION_STRING) {
    GraphTransferer gt;
    gt.SetSerializedGraphTransferInfo(graph_transfer_info.SerializeAsString());
    gt.DumpVerificationStringOfNodeTransferParams();
  }

#if 0
  int inputs_count = 0;
  int outputs_count = 0;
  for (const GraphTransferInfo::NodeInputInfo& input_params :
       graph_transfer_info.node_input_info()) {
    inputs_count += input_params.node_input_size();
  }

  for (const GraphTransferInfo::NodeOutputInfo& output_params :
       graph_transfer_info.node_output_info()) {
    outputs_count += output_params.max_byte_size_size();
  }
  // Allocate memory for node inputs and node outputs
  soc_interface_AllocateNodeInputAndNodeOutputArray(inputs_count,
                                                    outputs_count);
#endif

  // Construct node input parameters
#if 0
  std::unordered_map<int, std::tuple<void*, int>> inputs_map;
  for (const GraphTransferInfo::NodeInputInfo& input_params :
       graph_transfer_info.node_input_info()) {
    const int count = input_params.node_input_size();
    int node_ids[count];
    int ports[count];
    for (int i = 0; i < count; ++i) {
      const GraphTransferInfo::NodeInput& node_input =
          input_params.node_input(i);
      node_ids[i] = node_input.node_id() + NODE_ID_OFFSET;
      ports[i] = node_input.output_port();
    }
    void* inputs_ptr = soc_interface_SetOneNodeInputs(count, node_ids, ports);
    const int node_id = input_params.node_id();
    CHECK(inputs_map.count(node_id) == 0);
    inputs_map.emplace(node_id, std::make_tuple(inputs_ptr, count));
  }

  // Construct node output parameters
  std::unordered_map<int, std::tuple<void*, int>> outputs_map;
  for (const GraphTransferInfo::NodeOutputInfo& output_params :
       graph_transfer_info.node_output_info()) {
    const int count = output_params.max_byte_size_size();
    int sizes[count];
    for (int i = 0; i < count; ++i) {
      const int size = output_params.max_byte_size(i);
      sizes[i] = size;
    }
    void* outputs_ptr = soc_interface_SetOneNodeOutputs(count, sizes);
    const int node_id = output_params.node_id();
    CHECK(outputs_map.count(node_id) == 0);
    outputs_map.emplace(node_id, std::make_tuple(outputs_ptr, count));
  }
#endif

  // Instantiate graph
  soc_interface_InstantiateGraph();

  // Initialize graph
  // 1. Setup const nodes as tensor.
  for (const GraphTransferInfo::ConstNodeInfo& params :
       graph_transfer_info.const_node_info()) {
    const int max_dim_num = 4;
    uint32 shape[max_dim_num] = { 0 };

    CHECK(params.shape_size() <= max_dim_num);
    for(int i = 0; i < params.shape_size(); i++) {
        shape[i] = params.shape[i];
    }

    soc_interface_AppendConstTensor(
            params.name().c_str(), params.node_id(),
            shape, params.shape_size(),
            params.data().data(), params.data().length());
  }

  // Setup tensors
  // Graph input tensor
  // Graph output tensor
  // For each node output.
  // soc_interface_AppendTensor(tensor_name);

  // 2. Setup op nodes
  void * ovxnode;
  for (const GraphTransferInfo::NodeInfo& params :
       graph_transfer_info.node_info()) {
    const int node_id = params.node_id();
    const int op_id = params.soc_op_id();

    /* TODO: Parse param */
    ovxnode = soc_interface_AppendNode(
                             params.name().c_str(), node_id,
                             op_id, input_ptr, input_count,
                             output_ptr, output_count
                             );
    // Find all param inputs.
    // For each param noodes
    // soc_interface_SetNodeParam( ovxnode, param );
  }

  // Connent nodes
  // Set node input tensor.
  // For each node input
  // soc_interface_SetNodeInput(tensor_name, port)
  // Set node output tensor.
  // For each node output
  // soc_interface_SetNodeOutput(tensor_name, port)

  LOG(INFO) << "Setup graph completed";

  // construct graph
  return soc_interface_ConstructGraph();

}

bool OvxControlWrapper::ExecuteGraph() {
  return soc_interface_ExecuteGraph();
}

bool OvxControlWrapper::TeardownGraph() {
  soc_interface_ReleaseNodeInputAndNodeOutputArray();
  return soc_interface_TeardownGraph();
}

bool OvxControlWrapper::FillInputNode(const string& node_name,
                                          const ConstByteArray bytes) {
  uint64 byte_size;
  const int x = 1;
  const int y = 28;
  const int z = 28;
  const int d = 1;
  CHECK(std::get<2>(bytes) == DT_FLOAT);
  byte_size = std::get<1>(bytes);
  dummy_input_float_.resize(byte_size / sizeof(float));
  std::memcpy(dummy_input_float_.data(), std::get<0>(bytes), byte_size);
  return soc_interface_FillInputNodeFloat(
      x, y, z, d, reinterpret_cast<uint8*>(dummy_input_float_.data()),
      byte_size);
}

bool OvxControlWrapper::ReadOutputNode(
    const string& node_name, TensorAllocatorFunc tensor_allocator) {
  CHECK_NE(execute_info_, nullptr);
  TensorShape output_shape;
  // TODO(satok): Switch shape corresponding to input shape
  for (int i = 0; i < execute_info_->graph_output_node_name_size(); ++i) {
    if (execute_info_->graph_output_node_name(i) == node_name) {
      for (const TensorShapeProto::Dim& dim :
           execute_info_->default_graph_output_tensor_shape(i).shape().dim()) {
        output_shape.AddDim(dim.size());
      }
      break;
    }
  }
  std::vector<IRemoteFusedGraphExecutor::ByteArray> outputs;
  ReadOutputNode(node_name, &outputs);
  Tensor* output = tensor_allocator(output_shape);
  CHECK(output->TotalBytes() >= std::get<1>(outputs[0]));
  // TODO: Avoid specifying float
  std::memcpy(output->flat<float>().data(), std::get<0>(outputs[0]),
              std::get<1>(outputs[0]));
}

bool OvxControlWrapper::ReadOutputNode(
    const string& node_name, std::vector<ByteArray>* const outputs) {
  CHECK(outputs != nullptr);
  ByteArray output;
  soc_interface_ReadOutputNodeFloat(node_name.c_str(), &std::get<0>(output),
                                    &std::get<1>(output));
  // TODO: Accept all results
  std::get<2>(output) = DT_FLOAT;
  outputs->emplace_back(output);
  return true;
}

bool OvxControlWrapper::FillInputNode(const string& node_name,
                                          const Tensor& tensor) {
  StringPiece tensor_data = tensor.tensor_data();
  const ConstByteArray ba =
      ConstByteArray(reinterpret_cast<const uint8*>(tensor_data.data()),
                     tensor_data.size(), tensor.dtype());
  if (DBG_DUMP_INPUT_TENSOR_AS_FLOAT_DATA) {
    LOG(INFO) << "Input tensor data: element size = " << tensor.NumElements()
              << ", byte syze = " << tensor.TotalBytes();
    std::stringstream line;
    for (int i = 0; i < tensor.NumElements(); ++i) {
      line << tensor.flat<float>().data()[i] << ", ";
      if ((i - 2) % 3 == 0 || i == tensor.NumElements() - 1) {
        LOG(INFO) << "(" << ((i - 2) / 3) << ") " << line.str();
        line.str("");
        line.clear();
      }
    }
  }
  FillInputNode(node_name, ba);
  return true;
}

#else
int OvxControlWrapper::GetVersion() { return -1; }
bool OvxControlWrapper::Init(const RemoteFusedGraphExecuteInfo&) {
  return false;
}
bool OvxControlWrapper::Finalize() { return false; }
bool OvxControlWrapper::SetupGraph() { return false; }
bool OvxControlWrapper::ExecuteGraph() { return false; }
bool OvxControlWrapper::TeardownGraph() { return false; }
bool OvxControlWrapper::FillInputNode(const string&, const ConstByteArray) {
  return false;
}
bool OvxControlWrapper::FillInputNode(const string&, const Tensor&) {
  return false;
}
bool OvxControlWrapper::ReadOutputNode(
    const string& node_name, TensorAllocatorFunc tensor_allocator) {
  return false;
}
bool OvxControlWrapper::ReadOutputNode(const string&,
                                           std::vector<ByteArray>* const) {
  return false;
}
#endif

}  // namespace tensorflow
