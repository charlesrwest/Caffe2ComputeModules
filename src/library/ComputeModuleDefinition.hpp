#pragma once

#include<string>
#include<vector>
#include "caffe2/proto/caffe2.pb.h"

//Gradient blobs have the same name as the normal network names with "_gradient" post appended
namespace GoodBot
{
/**
This class represents a one or more operators which may have different configurations depending on the currently set "Mode".  A compute module definition is capable of generating lists of operators and/or net definitions which can be used to define actual networks in caffe2.  It also supports some meta information which can be useful to combine operators.  Its most important 

Currently supported modes:
"TRAIN", "TEST", "DEPLOY"
*/
class ComputeModuleDefinition
{
public:
/**
This function sets name member variable, which is returned by default from the Name() function.
@param inputName: The name to assign to this module 
*/
virtual void SetName(const std::string& inputName);

/**
This function returns the name associated with this module (should return the same if called multiple times).  Can be overloaded but returns the internal "name" variable value by default.
@return: The name associated with this module (returns "" if none has been set by default).
*/
virtual std::string Name() const;

/**
This function allow the mode of the module to be set, which can change its input/output blobs and the network operators that it returns.  By default, the "TRAIN", "TEST", "DEPLOY" modes are generally expected to be supported.  The default implementation just sets the mode string and returns true.
@param inputMode: The string indicating the mode the module should be in
@return: True if the provided mode is supported.
*/
virtual bool SetMode(const std::string& inputMode);

/**
This function returns the current mode of the module.  The default implementation returns the contents of the mode member string (starts "").
@return: The current mode the module is set to
*/
virtual std::string Mode() const;

/**
An ordered list of the names of the input blobs that the operator is currently set to use in the current mode.
@return: An ordered list of the input blobs being used. (returns empty list by default)
*/
virtual std::vector<std::string> GetInputBlobNames() const;

/**
An ordered list of the names of the output blobs that the operator is currently set to use in the current mode.  These are typically (but not always) made by the module's operators.  There are often other blobs that are made for internal layers that would not be in this list.
@return: An ordered list of the output blobs being used. (returns empty list by default)
*/
virtual std::vector<std::string> GetOutputBlobNames() const;

/**
This function returns an ordered list of the names of trainable blobs associated with this module.  This is typically used with solver modules and gradient generation to determine which fields need to be updated.
@return: The trainable blobs associated with this module (defaults to return an empty list)
*/
virtual std::vector<std::string> GetTrainableBlobNames() const;

/**
The shape vector associated with each of the trainable blobs returned by GetTrainableBlobNames().  This is typically known by a module without much difficulty because it is needed to create initialization operators for the trainable blobs.
@return: The shape of each blob whose name is returned by GetTrainableBlobNames() (returns empty list by default)
*/
virtual std::vector<std::vector<int64_t>> GetTrainableBlobShapes() const;

/**
This function returns a list of all the gradient blobs created by this module in the current mode.  By default, this is done by generating the gradient operators and then parsing their output blobs.  Typically, the default implementation does not need to be changed.
@return: A list of the names of all the gradient blobs created by this module.
*/
virtual std::vector<std::string> GetGradientBlobNames() const;

/**
This function returns a list of operator definitions to implement the functionality represented by this module.
@return: A list of all network operators needed for normal operation in the current mode (returns empty by default)
*/
virtual std::vector<caffe2::OperatorDef> GetNetworkOperators() const;

/**
This function returns a list of operator definitions to initialize the blobs needed by this module in the current mode.
@return: A list of all network operators needed for module blob initialization in the current mode (returns empty by default)
*/
virtual std::vector<caffe2::OperatorDef> GetNetworkInitializationOperators() const;

/**
This function returns a list of operator definitions to create gradients for the network operators in the current network.  The default implementation generates the gradient operators from the operators returned by GetNetworkOperators() and typically doesn't need to be overriden.
@return: A list of all network operators needed to create gradients for all of the module's operators in the current mode
*/
virtual std::vector<caffe2::OperatorDef> GetGradientOperators() const;

/**
This function autogenerates a network name based on the current module name, creates a blank network and adds the operators from GetNetworkInitializationOperators().  In short, it makes an initialization network for the current mode (typically just used with the highest level composite module).  It does not typically need to be overriden.
@return: A network definition that initializes the network's blobs for the current mode
*/
virtual caffe2::NetDef GetInitializationNetwork() const;

/**
This function autogenerates a network name based on the current module name, creates a blank network and adds the operators from GetNetworkOperators() (default implementation also adds operators from GetGradientOperators() and reorders as needed if in "TRAIN" mode).  In short, it makes the network definition for the current mode .
@return: A network definition for the current mode
*/
virtual caffe2::NetDef GetNetwork(const std::vector<std::string>& inputPreviouslyExistingBlobNames = {}) const;

~ComputeModuleDefinition()
{
}

protected:
std::string name;
std::string mode;
};

















} 
