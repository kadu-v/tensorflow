#ifndef TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_REDUCEMEAN_LAYER_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_REDUCEMEAN_LAYER_BUILDER_H_
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/delegates/coreml/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace coreml {
class ReduceMeanLayerBuilder : public OpBuilder {
 public:
  explicit ReduceMeanLayerBuilder(GraphBuilder* graph_builder)
      : OpBuilder(graph_builder) {}

  CoreML::Specification::NeuralNetworkLayer* Build() override;

  TfLiteStatus RegisterInputs(const TfLiteIntArray* inputs,
                              TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

  const std::string& DebugName() override;

 private:
  bool keepdims_ = true;
  std::vector<int> axis_;
};

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_REDUCEMEAN_LAYER_BUILDER_H_