#pragma once

#include "ComputeModuleDefinition.hpp"
#include<memory>

namespace GoodBot
{

/**
The purpose of this class is to allow easy construction of compute module definitions which are made of several sub-compute modules lined together.  A good example would be resnet's basic computation modules or GoogLeNet's modules.  It provides several defaults which make creating composite modules much easier.
*/
class CompositeComputeModuleDefinition : public ComputeModuleDefinition
{
public:
/**
This function takes ownership of the given module (including memory, deleting it when this object goes out of scope) and makes the members of this funciton affect it.
@param inputModule: The module to add to the composite.
*/
virtual void AddModule(ComputeModuleDefinition& inputModule);

/**
This function sets the mode for the composite top level and propogates it to all sub-modules.  It returns true if all submodules support the mode.
@param inputMode: The string indicating the mode the module should be in
@return: True if the provided mode is supported.
*/
virtual bool SetMode(const std::string& inputMode) override;

/**
The default implementation of this function returns a list of all trainable blobs provided by its component submodules.
@return: The trainable blobs associated with this module (defaults to return all submodule trainable blobs)
*/
virtual std::vector<std::string> GetTrainableBlobNames() const override;

/**
The default implementation of this function returns a list of all trainable blob shapes provided by its component submodules.
@return: The trainable blob shapes associated with this module (defaults to return all submodule trainable blob shapes)
*/
virtual std::vector<std::vector<int64_t>> GetTrainableBlobShapes() const override;

/**
The default implementation of this function returns an ordered list of all network operators provided by its submodules.
@return: A list of all network operation needed for normal operation in the current mode (returns submodule operators by default)
*/
virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const override;

/**
The default implementation of this function returns an ordered list of all network initialization operators provided by its submodules.
@return: A list of all network operators needed for normal operation in the current mode (returns submodule operators by default)
*/
virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const override;

/**
The default implementation of this function returns an ordered list of all network operators provided by its submodules.
@return: A list of all network operation needed for normal operation in the current mode (returns submodule operators by default)
*/
virtual std::vector<caffe2::OperatorDef> GetGradientOperators() const override;

protected:
std::vector<std::unique_ptr<ComputeModuleDefinition>> modules;
};

























}
