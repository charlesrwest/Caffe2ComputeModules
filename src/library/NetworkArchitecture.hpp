#pragma once

#include "caffe2/proto/caffe2.pb.h"
#include<string>

namespace GoodBot
{


/**
This class is suppose to represent a single network architecture and the alternate definitions that are required to train it (such as initialization parameters).  It can be used to generate different versions of the same underlying network for different uses (initialization, training, etc).
*/
class NetworkArchitecture
{
public:

virtual const std::string& Name() const = 0;

virtual caffe2::NetDef GetTrainingInitializationNetwork() const = 0;

virtual caffe2::NetDef GetTrainingNetwork() const = 0;

virtual caffe2::NetDef GetTestInitializationNetwork() const = 0;

virtual caffe2::NetDef GetTestNetwork() const = 0;

virtual caffe2::NetDef GetDeployNetwork() const = 0;

inline virtual std::string GetTrainingInitializationNetworkName() const;

inline virtual std::string GetTrainingNetworkName() const;

inline virtual std::string GetTestInitializationNetworkName() const;

inline virtual std::string GetTestNetworkName() const;

inline virtual std::string GetDeployNetworkName() const;

virtual ~NetworkArchitecture()
{
}
};

std::string NetworkArchitecture::GetTrainingInitializationNetworkName() const
{
return Name() + "_train_init";
}

std::string NetworkArchitecture::GetTrainingNetworkName() const
{
return Name() + "_train";
}

std::string NetworkArchitecture::GetTestInitializationNetworkName() const
{
return Name() + "_test_init";
}

std::string NetworkArchitecture::GetTestNetworkName() const
{
return Name() + "_test";
}

std::string NetworkArchitecture::GetDeployNetworkName() const
{
return Name() + "_deploy";
}


















} 
