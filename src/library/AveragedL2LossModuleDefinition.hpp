#pragma once

#include "CompositeComputeModuleDefinition.hpp"

namespace GoodBot
{

struct AveragedL2LossModuleDefinitionParameters
{
std::string inputBlobName;
std::string moduleName;
std::string trainingExpectedOutputBlobName;
std::string testExpectedOutputBlobName;
};

class AveragedL2LossModuleDefinition : public CompositeComputeModuleDefinition
{
public:
AveragedL2LossModuleDefinition(const AveragedL2LossModuleDefinitionParameters& inputParameters);
};






























}
