#pragma once

#include "LayerDefinition.hpp"

namespace GoodBot
{

struct FullyConnectedLayerDefinitionParameters
{
std::string inputBlobName;
int64_t numberOfInputs = 0;
int64_t numberOfNodes = 0;
std::string outputBlobName;
std::string layerName;
std::string weightFillType = "XavierFill";
std::string biasFillType = "ConstantFill";
std::string activationType = "Sigmoid";
std::string lossType = "SoftmaxWithLoss";
};

class FullyConnectedLayerDefinition : public LayerDefinition
{
public:
FullyConnectedLayerDefinition(const FullyConnectedLayerDefinitionParameters& inputParameters);

virtual const std::string& Type() const override;

virtual const std::string& Name() const override;

virtual std::vector<std::string> GetInputBlobNames() const override;

virtual std::vector<std::string> GetOutputBlobNames() const override;

virtual std::vector<std::string> GetGradientBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const  override;

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetGradientOperators() const  override;

std::string GetWeightsBlobName() const;

std::string GetBiasesBlobName() const;

std::string GetFullyConnectedOutputBlobName() const;

int64_t GetNumberOfNodes() const;

protected:
std::string inputBlobName;
int64_t numberOfInputs;
int64_t numberOfNodes;
std::string outputBlobName;
std::string layerName;
std::string weightFillType;
std::string biasFillType;
std::string activationType;
std::string lossType;
};



















} 
