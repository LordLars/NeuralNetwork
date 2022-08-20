package AI;

import javax.xml.crypto.Data;

public class NNLayer {

    private final float[] neurons;
    private final int neuronCount;

    private final int preNeuronCount;
    private float[] preNeurons;
    private float[] nextChain;

    private float[] chain;
    private final float lernRate;

    private final ActivationFunction activationFunction;

    private final NN nn;
    private final int layerIndex;

    public NNLayer(int preNeuronCount, int neuronCount, ActivationFunction activationFunction, float nLernRate, NN nn, int layerIndex) {
        this.preNeuronCount = preNeuronCount;
        this.neuronCount = neuronCount;
        this.activationFunction = activationFunction;
        this.nn = nn;
        this.layerIndex = layerIndex;

        neurons = new float[neuronCount];
        lernRate = nLernRate;
        buildLayer();
    }

    //region NN LAYER BUILDER
    /**
     * Builds a new Layer with random weights and biases
     */
    public void buildLayer() {
        nn.weights[layerIndex] = new float[neuronCount][preNeuronCount];
        nn.biases[layerIndex] = new float[neuronCount];

        for(int i = 0; i < neuronCount; i++) {
            for(int j = 0; j < preNeuronCount; j++) {
                nn.weights[layerIndex][i][j] = (float) (Math.random() * 2-1);
            }
        }
        for(int i = 0; i < neuronCount; i++) {
            nn.biases[layerIndex][i] = (float) (Math.random() * 2 -1);
        }
    }
    //endregion

    //region PROPAGATION

    /**
     * Propagation method
     */
    public float[] propagation(float[] input) {
        preNeurons = input;
        for(int i = 0; i < neuronCount; i++) neurons[i] = activationFunction.activation(add(i) + nn.biases[layerIndex][i]);
        return neurons;
    }

    /**
     * Adds up all products of the weights and previous neurons
     * @param index index of the Neuron
     * @return sum
     */
    private float add(int index) {
        float sum = 0;
        for(int j = 0; j < preNeurons.length; j++){
            sum += preNeurons[j] * nn.weights[layerIndex][index][j];
        }
        return sum;
    }

    //endregion

    //region BACK - PROPAGATION

    /**
     * Backpropagation method
     * @param targets aims of the AI
     */
    public void backPropagation(float[] targets) {
        updateChain(targets);
        updateWeights();
        updateBiases();
    }

    /**
     * Updates the weights with the backpropagation method
     */
    private void updateWeights(){
        for(int i = 0; i < neuronCount; i++) {
            for(int j = 0; j < preNeuronCount; j++) {
                nn.weightGradients[layerIndex][i][j] += chain[i] * preNeurons[j] * lernRate;
            }
        }
    }

    /**
     * Updates the biases with the backpropagation method
     */
    private void updateBiases(){
        for(int i = 0; i < neuronCount; i++) nn.biasGradients[layerIndex][i] += chain[i] * lernRate;
    }

    /**
     * Updates the Chain Rule
     * @param targets aims of the AI
     */
    private void updateChain(float[] targets) {
        chain = new float[neuronCount];
        nextChain = new float[preNeuronCount];
        for(int i = 0; i < neuronCount; i++) chain[i] = targets[i] * activationFunction.activationDerivative(add(i) + nn.biases[layerIndex][i]);
        for(int i = 0; i < preNeuronCount; i++) {
            float chainSum = 0;
            for(int j = 0; j < neuronCount; j++) chainSum += chain[j] * nn.weights[layerIndex][j][i];
            nextChain[i] = chainSum;
        }
    }

    //endregion

    //region GETTER & SETTER

    /**
     * @return Values of the next Chain
     */
    public float[] getNextChain() {
        return nextChain;
    }

    public float[] getNeurons(){
        return neurons;
    }

    //endregion

}
