#include "ComputeModuleDefinition.hpp"
#include "UtilityFunctions.hpp"
#include<algorithm>
#include<iostream>

using namespace GoodBot;

void ComputeModuleDefinition::SetName(const std::string& inputName)
{
name = inputName;
}

std::string ComputeModuleDefinition::Name() const
{
return name;
}

bool ComputeModuleDefinition::SetMode(const std::string& inputMode)
{
mode = inputMode;

return true;
}

std::string ComputeModuleDefinition::Mode() const
{
return mode;
}

std::vector<std::string> ComputeModuleDefinition::GetInputBlobNames() const
{
return {};
}

std::vector<std::string> ComputeModuleDefinition::GetOutputBlobNames() const
{
return {};
}

std::vector<std::string> ComputeModuleDefinition::GetTrainableBlobNames() const
{
return {};
}

std::vector<std::vector<int64_t>> ComputeModuleDefinition::GetTrainableBlobShapes() const
{
return {};
}

std::vector<std::string> ComputeModuleDefinition::GetGradientBlobNames() const
{
std::vector<caffe2::OperatorDef> gradientOperators = GetGradientOperators();

std::vector<std::string> results;
for(const caffe2::OperatorDef& gradientOperator : gradientOperators)
{
for(int64_t outputIndex = 0; outputIndex < gradientOperator.output_size(); outputIndex++)
{
results.emplace_back(gradientOperator.output(outputIndex));
}
}

//remove any redundant blob names
std::sort(results.begin(), results.end());
results.erase(std::unique(results.begin(), results.end()));

return results;
}

std::vector<caffe2::OperatorDef> ComputeModuleDefinition::GetNetworkOperators() const
{
return {};
}

std::vector<caffe2::OperatorDef> ComputeModuleDefinition::GetNetworkInitializationOperators() const
{
return {};
}

std::vector<caffe2::OperatorDef> ComputeModuleDefinition::GetGradientOperators() const
{
std::vector<caffe2::OperatorDef> result;

std::vector<caffe2::OperatorDef> networkOperators = GetNetworkOperators();

//Reverse the order and make a gradient operator for each network operator
for(std::vector<caffe2::OperatorDef>::reverse_iterator iter = networkOperators.rbegin(); iter != networkOperators.rend(); iter++ )
{
std::vector<caffe2::OperatorDef> gradientOperators = GetGradientOperatorsFromOperator(*iter);

result.insert(result.end(), gradientOperators.begin(), gradientOperators.end());
}

return result;
}

caffe2::NetDef ComputeModuleDefinition::GetInitializationNetwork() const
{
caffe2::NetDef network;
network.set_name(Name() + "_" + Mode() + "_init");

std::vector<caffe2::OperatorDef> initOperators = GetNetworkInitializationOperators();

for(const caffe2::OperatorDef& operatorDefinition : initOperators)
{
*network.add_op() = operatorDefinition; //Add to network
}

return network;
}

caffe2::NetDef ComputeModuleDefinition::GetNetwork(const std::vector<std::string>& inputPreviouslyExistingBlobNames) const
{
caffe2::NetDef network;
network.set_name(Name() + "_" + Mode());

std::vector<caffe2::OperatorDef> networkOperators = GetNetworkOperators();

if(mode == "TRAIN")
{
//Add any gradient calculations to the network too
std::vector<caffe2::OperatorDef> gradientOperators = GetGradientOperators();

networkOperators.insert(networkOperators.end(), gradientOperators.begin(), gradientOperators.end());
}

networkOperators = ReorderOperatorsToResolveDependencies(networkOperators, inputPreviouslyExistingBlobNames);

for(const caffe2::OperatorDef& operatorDefinition : networkOperators)
{
*network.add_op() = operatorDefinition; //Add to network
}

return network;
}

