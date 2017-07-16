#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct SoftMaxLayerDefinitionParameters
{
std::string inputBlobName;
std::string layerName;
std::string trainingExpectedOutputBlobName;
std::string testExpectedOutputBlobName;
};

/**
This class is a straight forward implementation of the "Softmax"/"SoftmaxWithLoss" operators (depending on mode).  See ComputeModuleDefinition for the function meanings.
*/
class SoftMaxLayerDefinition : public ComputeModuleDefinition
{
public:
SoftMaxLayerDefinition(const SoftMaxLayerDefinitionParameters& inputParameters);

virtual std::string Name() const override;

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetGradientOperators() const override;

std::string GetSoftMaxOutputBlobName() const;

std::string GetTrainingLossOutputBlobName() const;

std::string GetTestLossOutputBlobName() const;

protected:
std::string mode;
std::string inputBlobName;
std::string layerName;
std::string trainingExpectedOutputBlobName;
std::string testExpectedOutputBlobName;
};




























}
