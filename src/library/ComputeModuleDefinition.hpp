#pragma once

#include<string>
#include<vector>
#include "caffe2/proto/caffe2.pb.h"

//Gradient blobs have the same name as the normal network names with "_gradient" post appended
namespace GoodBot
{
/**
Currently supported modes:
"TRAIN", "TEST", "DEPLOY"
*/
class ComputeModuleDefinition
{
public:
virtual void SetName(const std::string& inputName);

virtual std::string Name() const;

virtual bool SetMode(const std::string& inputMode);

virtual std::string Mode() const;

virtual std::vector<std::string> GetInputBlobNames() const;

virtual std::vector<std::string> GetOutputBlobNames() const;

//List of bobs which will have gradients generated and should be updated when training is done.  Defaults to none.
virtual std::vector<std::string> GetTrainableBlobNames() const;

virtual std::vector<std::string> GetGradientBlobNames() const;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const;

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const;

virtual std::vector<caffe2::OperatorDef> GetGradientOperators() const;

virtual caffe2::NetDef GetInitializationNetwork() const;

virtual caffe2::NetDef GetNetwork() const;

~ComputeModuleDefinition()
{
}

protected:
std::string name;
std::string mode;
};

















} 
