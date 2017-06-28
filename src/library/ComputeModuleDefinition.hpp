#pragma once

#include<string>
#include<vector>
#include "caffe2/proto/caffe2.pb.h"

//Gradient blobs have the same name as the normal network names with "_gradient" post appended
namespace GoodBot
{



class ComputeModuleDefinition
{
public:
virtual const std::string& Type() const = 0;

virtual const std::string& Name() const = 0;

virtual std::vector<std::string> GetDeployInputBlobNames() const = 0;

virtual std::vector<std::string> GetTrainingInputBlobNames() const;

virtual std::vector<std::string> GetTestInputBlobNames() const;

virtual std::vector<std::string> GetDeployOutputBlobNames() const = 0;

virtual std::vector<std::string> GetTrainingOutputBlobNames() const;

virtual std::vector<std::string> GetTrainingGradientBlobNames() const = 0;

virtual std::vector<caffe2::OperatorDef> GetDeployNetworkOperators() const = 0;

virtual std::vector<caffe2::OperatorDef> GetTrainingNetworkOperators() const;

virtual std::vector<caffe2::OperatorDef> GetTestNetworkOperators() const;

virtual std::vector<caffe2::OperatorDef> GetTrainingNetworkInitializationOperators() const = 0;

virtual std::vector<caffe2::OperatorDef> GetTrainingGradientOperators() const;

~ComputeModuleDefinition()
{
}
};

















} 
