#pragma once

#include "NetworkArchitecture.hpp"
#include<memory>
#include<string>
#include<vector>
#include "FullyConnectedNetworkArchitecture.hpp"

namespace GoodBot
{

class NetworkFactory
{
public:
NetworkFactory(); //Sets up unique base network name

//Create fully connected network architecture
std::unique_ptr<NetworkArchitecture> CreateFullyConnectedNetwork(const FullyConnectedNetworkParameters& inputParameters) const;
 
protected:
std::string baseName;
};












} 
