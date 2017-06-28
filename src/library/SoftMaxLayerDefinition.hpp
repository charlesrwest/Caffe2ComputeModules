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

class SoftMaxLayerDefinition : public ComputeModuleDefinition
{
public:
SoftMaxLayerDefinition(const SoftMaxLayerDefinitionParameters& inputParameters);

virtual const std::string& Type() const override;

virtual const std::string& Name() const override;

virtual std::vector<std::string> GetDeployInputBlobNames() const override;

virtual std::vector<std::string> GetTrainingInputBlobNames() const override;

virtual std::vector<std::string> GetTestInputBlobNames() const override;

virtual std::vector<std::string> GetDeployOutputBlobNames() const override;

virtual std::vector<std::string> GetTrainingOutputBlobNames() const override;

virtual std::vector<std::string> GetTrainingGradientBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetDeployNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetTrainingNetworkOperators() const  override;

virtual std::vector<caffe2::OperatorDef> GetTestNetworkOperators() const  override;

virtual std::vector<caffe2::OperatorDef> GetTrainingNetworkInitializationOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetTrainingGradientOperators() const override;

std::string GetSoftMaxOutputBlobName() const;

std::string GetTrainingLossOutputBlobName() const;

std::string GetTestLossOutputBlobName() const;

protected:
std::string inputBlobName;
std::string layerName;
std::string trainingExpectedOutputBlobName;
std::string testExpectedOutputBlobName;
};




























}
