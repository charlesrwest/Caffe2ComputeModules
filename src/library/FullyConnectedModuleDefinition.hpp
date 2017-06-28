#pragma once

#include "ComputeModuleDefinition.hpp"
#include "FullyConnectedLayerDefinition.hpp"

namespace GoodBot
{

struct FullyConnectedModuleDefinitionParameters
{
std::string inputBlobName;
int64_t numberOfInputs = 0;
std::vector<int64_t> numberOfNodesInLayers;
std::string moduleName;
std::string weightFillType = "XavierFill";
std::string biasFillType = "ConstantFill";
std::string activationType = "Sigmoid";
};

class FullyConnectedModuleDefinition : public ComputeModuleDefinition
{
public:
FullyConnectedModuleDefinition(const FullyConnectedModuleDefinitionParameters& inputParameters);

virtual const std::string& Type() const override;

virtual const std::string& Name() const override;

virtual std::vector<std::string> GetDeployInputBlobNames() const override;

virtual std::vector<std::string> GetDeployOutputBlobNames() const override;

virtual std::vector<std::string> GetTrainingGradientBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetDeployNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetTrainingNetworkInitializationOperators() const override;

protected:
std::string moduleName;
std::vector<FullyConnectedLayerDefinition> layers;
};































} 
