#pragma once

#include "CompositeComputeModuleDefinition.hpp"

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

class FullyConnectedModuleDefinition : public CompositeComputeModuleDefinition
{
public:
FullyConnectedModuleDefinition(const FullyConnectedModuleDefinitionParameters& inputParameters);

virtual std::string Name() const override;

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

protected:
std::string moduleName;
};































} 
