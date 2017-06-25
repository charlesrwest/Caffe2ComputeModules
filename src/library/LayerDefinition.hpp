#pragma once

#include<string>
#include<vector>
#include "caffe2/proto/caffe2.pb.h"

//Gradient blobs have the same name as the normal network names with "_gradient" post appended
namespace GoodBot
{



class LayerDefinition
{
public:
virtual const std::string& Type() const = 0;

virtual const std::string& Name() const = 0;

virtual std::vector<std::string> GetInputBlobNames() const = 0;

virtual std::vector<std::string> GetOutputBlobNames() const = 0;

virtual std::vector<std::string> GetGradientBlobNames() const = 0;

virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const = 0;

virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const = 0;

virtual std::vector<caffe2::OperatorDef> GetGradientOperators() const = 0;

~LayerDefinition()
{
}
};






















} 
