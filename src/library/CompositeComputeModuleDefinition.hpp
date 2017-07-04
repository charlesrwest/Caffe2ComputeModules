#pragma once

#include "ComputeModuleDefinition.hpp"
#include<memory>

namespace GoodBot
{

class CompositeComputeModuleDefinition : public ComputeModuleDefinition
{
public:
//Takes ownership of module and deletes on destruction
virtual void AddModule(ComputeModuleDefinition& inputModule);

virtual bool SetMode(const std::string& inputMode) override;

virtual std::vector<std::string> GetTrainableBlobNames() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const override;

virtual std::vector<caffe2::OperatorDef> GetGradientOperators() const override;

protected:
std::vector<std::unique_ptr<ComputeModuleDefinition>> modules;
};

























}
