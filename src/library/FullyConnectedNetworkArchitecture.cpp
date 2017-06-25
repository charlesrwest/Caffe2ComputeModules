#include "FullyConnectedNetworkArchitecture.hpp"

using namespace GoodBot;

FullyConnectedNetworkArchitecture::FullyConnectedNetworkArchitecture(const std::string& inputArchitectureName, const FullyConnectedNetworkParameters& inputParameters) : name(inputArchitectureName), inputBlobName(inputParameters.inputBlobName), expectedOutputBlobName(inputParameters.expectedOutputBlobName), nodesPerLayer(inputParameters.nodesPerLayer), weightFillType(weightFillType), biasFillType(inputParameters.biasFillType), numberOfInputs(inputParameters.numberOfInputs), numberOfOutputs(inputParameters.numberOfOutputs), activationType(inputParameters.activationType), lossType(inputParameters.lossType)
{
std::vector<std::unique_ptr<LayerDefinition>> layers;

for(int64_t layerIndex = 0; layerIndex < nodesPerLayer.size(); layerIndex++)
{
std::string layerName = "fully_connected_" + std::to_string(layerIndex);

if(layerIndex == 0)
{
layers.emplace_back(new FullyConnectedLayerDefinition({inputBlobName, numberOfInputs, nodesPerLayer[layerIndex], layerName, weightFillType, biasFillType, activationType, lossType}));
}
else
{
const FullyConnectedLayerDefinition& previousLayer = (const FullyConnectedLayerDefinition&) *layers[layerIndex-1];

layers.emplace_back(new FullyConnectedLayerDefinition({previousLayer.GetOutputBlobNames()[0], previousLayer.GetNumberOfNodes(), nodesPerLayer[layerIndex], layerName, weightFillType, biasFillType, activationType, lossType}));
}
}

}

const std::string& FullyConnectedNetworkArchitecture::Name() const
{
return name;
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetTrainingInitializationNetwork() const
{
caffe2::NetDef initializationNetwork;

initializationNetwork.set_name(GetTrainingInitializationNetworkName());

for(int64_t layerIndex = 0; layerIndex < nodesPerLayer.size(); layerIndex++)
{
int64_t inputSize = 0

if(layerIndex == 0)
{
inputSize = numberOfInputs;
}
else
{
inputSize = nodesPerLayer[layerIndex-1]; //Size of previous layer
}

//Setup weights/biases
caffe2::OperatorDef& weightOperator = *initializationNetwork.add_op();
weightOperator.set_type(weightFillType);
caffe2::Argument& weightShape = weightOperator.add_arg();
weightShape.set_name("shape");
weightShape.add_ints(nodesPerLayer[layerIndex]); //Number of nodes in this layer
weightShape.add_ints(inputSize); //Number of inputs to this layer
weightOperator.add_output(GetLayerWeightsName(layerIndex));

caffe2::OperatorDef& biasOperator = *initializationNetwork.add_op();
biasOperator.set_type("XavierFill");
caffe2::Argument& biasShape = biasOperator.add_arg();
biasShape.set_name("shape");
biasShape.add_ints(nodesPerLayer[layerIndex]); //Number of nodes in this layer
weightOperator.add_output(GetLayerBiasesName(layerIndex));
}

return initializationNetwork;
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetTrainingNetwork() const
{
caffe2::NetDef trainingNetwork;
trainingNetwork.set_name(GetTrainingNetworkName());

//Add network body
for(int64_t layerIndex = 0; layerIndex < nodesPerLayer.size(); layerIndex++)
{

//Add fully connected portion
std::string layerInputBlobName;
if(layerIndex == 0) //Either input to network or output of previous layer
{
layerInputBlobName = inputBlobName
}
else
{
layerInputBlobName = GetLayerActivationOutputName(layerIndex-1);
}

caffe2::OperatorDef& fullyConnectedOperator = *trainingNetwork.add_op();
fullyConnectedOperator.set_type("FC");
fullyConnectedOperator.add_input(layerInputBlobName);
fullyConnectedOperator.add_input(GetLayerWeightsName(layerIndex));
fullyConnectedOperator.add_input(GetLayerBiasesName(layerIndex));
fullyConnectedOperator.add_output(GetLayerFullyConnectedOutputName(layerIndex));

//Add activation layer
caffe2::OperatorDef& activationOperator = *trainingNetwork.add_op();
activationOperator.set_type(activationType);
activationOperator.add_input(GetLayerFullyConnectedOutputName(layerIndex));
activationOperator.add_output(GetLayerActivationOutputName(layerIndex));
}

//Add loss
caffe2::OperatorDef& lossOperator = *trainingNetwork.add_op();
activationOperator.set_type(lossType);
activationOperator.add_input(GetLayerActivationOutputName(GetNumberOfFullyConnectedLayers()-1)); //Connect to last activation layer
activationOperator.add_input(expectedOutputBlobName)); //Connect to last activation layer

//Hard code for now
activationOperator.add_output("softmax");
activationOperator.add_output("loss");

//Add gradient operators
caffe2::OperatorDef& lossGradientOperator = *trainingNetwork.add_op();
lossGradientOperator.set_type("ConstantFill");
caffe2::Argument& lossGradientArgument = lossGradientOperator.add_arg();
lossGradientArgument.set_name("value");
lossGradientArgument.set_f(1.0);

//Hard code for now
lossGradientOperator.add_input("loss");
lossGradientOperator.add_output("loss_grad")
lossGradientOperator.set_is_gradient_op(true);

//Collected
//FC operators
//activation operators
//Loss operators

//Adds a new constant fill operator with singular value of 1.0, input of the loss and output of "loss_grad", then sets set_is_gradient_op to true

//Then reverses the list of previously collected operators

//Basically, it makes a second computation network that extends out from the loss in reverse order from the original, updating the weights

}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetTestInitializationNetwork() const
{
return caffe2::NetDef();
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetTestNetwork() const
{
return caffe2::NetDef();
}

caffe2::NetDef FullyConnectedNetworkArchitecture::GetDeployNetwork() const
{
return caffe2::NetDef();
}

int64_t FullyConnectedNetworkArchitecture::GetNumberOfFullyConnectedLayers() const
{
return nodesPerLayer.size();
}

std::string FullyConnectedNetworkArchitecture::GetLayerWeightsName(int64_t inputLayerIndex) const
{
return "fullyConnected_weights_" + std::to_string(inputLayerIndex);
}

std::string FullyConnectedNetworkArchitecture::GetLayerBiasesName(int64_t inputLayerIndex) const
{
return "fullyConnected_biases_" + std::to_string(inputLayerIndex);
}

std::string FullyConnectedNetworkArchitecture::GetLayerFullyConnectedOutputName(int64_t inputLayerIndex) const
{
return "fullyConnected_operator_" + std::to_string(inputLayerIndex);
}

std::string FullyConnectedNetworkArchitecture::GetLayerActivationOutputName(int64_t inputLayerIndex) const
{
return "fullyConnected_activation_" + std::to_string(inputLayerIndex);
}

