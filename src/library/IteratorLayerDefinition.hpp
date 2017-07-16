#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct IteratorLayerDefinitionParameters
{
std::string layerName;
int64_t initialValue = 0;
};

/**
This class is a straight forward implementation of the "Iter" operator.  See ComputeModuleDefinition for the function meanings.
*/
class IteratorLayerDefinition : public ComputeModuleDefinition
{
public:
IteratorLayerDefinition(const IteratorLayerDefinitionParameters& inputParameters);

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const override;

std::string IteratorBlobName() const;

protected:
std::string layerName;
int64_t initialValue;
};






















} 
