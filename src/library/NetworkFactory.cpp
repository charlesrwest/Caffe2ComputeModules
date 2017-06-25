#include "NetworkFactory.hpp"

#include "UtilityFunctions.hpp"

using namespace GoodBot;


NetworkFactory::NetworkFactory() : baseName(GenerateRandomHexString(5))
{
} 

std::unique_ptr<NetworkArchitecture> NetworkFactory::CreateFullyConnectedNetwork(const FullyConnectedNetworkParameters& inputParameters) const
{

}
