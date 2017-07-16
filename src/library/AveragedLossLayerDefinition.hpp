#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct AveragedLossLayerDefinitionParameters
{
std::string inputBlobName;
std::string layerName;
};

/**
This class is a straight forward implementation of the AveragedLoss operator.  See ComputeModuleDefinition for the function meanings.
*/
class AveragedLossLayerDefinition : public ComputeModuleDefinition
{
public:
AveragedLossLayerDefinition(const AveragedLossLayerDefinitionParameters& inputParameters);

virtual std::string Name() const override;

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetGradientOperators() const override;

std::string GetAveragedLossOutputBlobName() const;

protected:
std::string inputBlobName;
std::string layerName;
std::string mode;
};




























} 
