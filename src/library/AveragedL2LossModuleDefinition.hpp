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

/**
This class combines the L2LossLayer with the AveragedLoss layer to make a single module which computes the L2 loss and then reduces it to a scalar.
*/
class AveragedL2LossModuleDefinition : public CompositeComputeModuleDefinition
{
public:
AveragedL2LossModuleDefinition(const AveragedL2LossModuleDefinitionParameters& inputParameters);
};






























}
