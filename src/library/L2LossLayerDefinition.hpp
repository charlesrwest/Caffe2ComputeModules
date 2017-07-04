#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct L2LossLayerDefinitionParameters
{
std::string inputBlobName;
std::string layerName;
std::string trainingExpectedOutputBlobName;
std::string testExpectedOutputBlobName;
};

class L2LossLayerDefinition : public ComputeModuleDefinition
{
public:
L2LossLayerDefinition(const L2LossLayerDefinitionParameters& inputParameters);

virtual std::string Name() const override;

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

std::string GetL2LossOutputBlobName() const;

protected:
std::string inputBlobName;
std::string layerName;
std::string trainingExpectedOutputBlobName;
std::string testExpectedOutputBlobName;
};






}
