#include "ComputeModuleDefinition.hpp"
#include "UtilityFunctions.hpp"

using namespace GoodBot;

std::vector<std::string> ComputeModuleDefinition::GetTrainingInputBlobNames() const
{
return GetDeployInputBlobNames(); //Default to deploy
}

std::vector<std::string> ComputeModuleDefinition::GetTestInputBlobNames() const
{
return GetDeployInputBlobNames(); //Default to deploy
}

std::vector<std::string> ComputeModuleDefinition::GetTrainingOutputBlobNames() const
{
return GetDeployOutputBlobNames(); //Default to deploy
}

std::vector<caffe2::OperatorDef> ComputeModuleDefinition::GetTrainingNetworkOperators() const
{
return GetDeployNetworkOperators(); //Default to deploy unless overriden
}

std::vector<caffe2::OperatorDef> ComputeModuleDefinition::GetTestNetworkOperators() const
{
return GetDeployNetworkOperators(); //Default to deploy unless overriden
} 

std::vector<caffe2::OperatorDef> ComputeModuleDefinition::GetTrainingGradientOperators() const
{
std::vector<caffe2::OperatorDef> result;

std::vector<caffe2::OperatorDef> networkOperators = GetTrainingNetworkOperators();

//Reverse the order and make a gradient operator for each network operator
for(std::vector<caffe2::OperatorDef>::reverse_iterator iter = networkOperators.rbegin(); iter != networkOperators.rend(); iter++ )
{
std::vector<caffe2::OperatorDef> gradientOperators = GetGradientOperatorsFromOperator(*iter);

result.insert(result.end(), gradientOperators.begin(), gradientOperators.end());
}

return result;
}
