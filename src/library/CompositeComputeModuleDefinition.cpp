#include "CompositeComputeModuleDefinition.hpp"
#include<iostream>

using namespace GoodBot;

void CompositeComputeModuleDefinition::AddModule(ComputeModuleDefinition& inputModule)
{
modules.emplace_back(std::unique_ptr<ComputeModuleDefinition>(&inputModule));
}

bool CompositeComputeModuleDefinition::SetMode(const std::string& inputMode)
{
bool ModeWasSetSuccessfully = true;

for(const std::unique_ptr<ComputeModuleDefinition>& module : modules)
{
if(!module->SetMode(inputMode))
{
ModeWasSetSuccessfully = false;
}
}

if(!ComputeModuleDefinition::SetMode(inputMode))
{
ModeWasSetSuccessfully = false;
}

return ModeWasSetSuccessfully;
}

std::vector<std::string> CompositeComputeModuleDefinition::GetTrainableBlobNames() const
{
std::vector<std::string> trainableBlobNames;

for(const std::unique_ptr<ComputeModuleDefinition>& module : modules)
{
std::vector<std::string> moduleTrainableBlobNames = module->GetTrainableBlobNames();

trainableBlobNames.insert(trainableBlobNames.end(), moduleTrainableBlobNames.begin(), moduleTrainableBlobNames.end());
}

return trainableBlobNames;
}

std::vector<std::vector<int64_t>> CompositeComputeModuleDefinition::GetTrainableBlobShapes() const
{
std::vector<std::vector<int64_t>> trainableBlobShapes;

for(const std::unique_ptr<ComputeModuleDefinition>& module : modules)
{
std::vector<std::vector<int64_t>> moduleTrainableBlobShapes = module->GetTrainableBlobShapes();

trainableBlobShapes.insert(trainableBlobShapes.end(), moduleTrainableBlobShapes.begin(), moduleTrainableBlobShapes.end());
}

return trainableBlobShapes;
}

std::vector<caffe2::OperatorDef> CompositeComputeModuleDefinition::GetNetworkOperators() const
{
std::vector<caffe2::OperatorDef> results;

for(const std::unique_ptr<ComputeModuleDefinition>& module : modules)
{
std::vector<caffe2::OperatorDef> moduleNetworkOperators = module->GetNetworkOperators();

results.insert(results.end(), moduleNetworkOperators.begin(), moduleNetworkOperators.end());
}

return results;
}

std::vector<caffe2::OperatorDef> CompositeComputeModuleDefinition::GetNetworkInitializationOperators() const
{
std::vector<caffe2::OperatorDef> results;

for(const std::unique_ptr<ComputeModuleDefinition>& module : modules)
{
std::vector<caffe2::OperatorDef> moduleOperators = module->GetNetworkInitializationOperators();

std::cout << module->Name() << " has " << moduleOperators.size() << " init operators" << std::endl;

results.insert(results.end(), moduleOperators.begin(), moduleOperators.end());
}

return results;
}

std::vector<caffe2::OperatorDef> CompositeComputeModuleDefinition::GetGradientOperators() const
{
std::vector<caffe2::OperatorDef> results;

for(std::vector<std::unique_ptr<ComputeModuleDefinition>>::const_reverse_iterator iter = modules.rbegin(); iter != modules.rend(); iter++)
{
std::vector<caffe2::OperatorDef> gradientOperators = (*iter)->GetGradientOperators();

results.insert(results.end(), gradientOperators.begin(), gradientOperators.end());
}

return results;
}
