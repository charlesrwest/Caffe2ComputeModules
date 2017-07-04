#pragma once

#include "ComputeModuleDefinition.hpp"

namespace GoodBot
{

struct FullyConnectedLayerDefinitionParameters
{
std::string inputBlobName;
int64_t numberOfInputs = 0;
int64_t numberOfNodes = 0;
std::string layerName;
std::string weightFillType = "XavierFill";
std::string biasFillType = "ConstantFill";
std::string activationType = "Sigmoid";
};

class FullyConnectedLayerDefinition : public ComputeModuleDefinition
{
public:
FullyConnectedLayerDefinition(const FullyConnectedLayerDefinitionParameters& inputParameters);

virtual std::string Name() const override;

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<std::string> GetTrainableBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const override;

std::string GetWeightsBlobName() const;

std::string GetBiasesBlobName() const;

std::string GetFullyConnectedOutputBlobName() const;

std::string GetOutputBlobName() const;

int64_t GetNumberOfNodes() const;

protected:
std::string inputBlobName;
int64_t numberOfInputs;
int64_t numberOfNodes;
std::string layerName;
std::string weightFillType;
std::string biasFillType;
std::string activationType;
std::string lossType;
};



















} 
