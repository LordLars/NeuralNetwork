package AI;

public class NNLayer {

    private float[][] weights;
    private float[] biases;
    private final float[] neurons;
    private int neuronCount;

    private int preNeuronCount;
    private float[] preNeurons;
    private float[] nextChain;

    private float[] chain;
    private final float lernRate;

    private final ActivationFunction activationFunction;


    public NNLayer(int preNeuronCount, int neuronCount, ActivationFunction activationFunction, float nLernRate) {
        this.preNeuronCount = preNeuronCount;
        this.neuronCount = neuronCount;
        this.activationFunction = activationFunction;
        neurons = new float[neuronCount];
        lernRate = nLernRate;
        buildLayer();
    }

    //region NN LAYER BUILDER
    /**
     * Builds a new Layer with random weights and biases
     */
    private void buildLayer() {
        weights = new float[neuronCount][preNeuronCount];
        biases = new float[neuronCount];

        for(int i = 0; i < neuronCount; i++) {
            for(int j = 0; j < preNeuronCount; j++) {
                weights[i][j] = (float) (Math.random() * 2-1);
            }
        }
        for(int i = 0; i < neuronCount; i++) {
            biases[i] = (float) (Math.random() * 2 -1);
        }
    }
    //endregion

    //region PROPAGATION

    /**
     * Propagation method
     */
    public float[] propagation(float[] input) {
        preNeurons = input;
        for(int i = 0; i < neuronCount; i++) neurons[i] = activationFunction.activation(add(i) + biases[i]);
        return neurons;
    }

    /**
     * Adds up all products of the weights and previous neurons
     * @param index index of the Neuron
     * @return sum
     */
    private float add(int index) {
        float sum = 0;
        for(int j = 0; j < preNeurons.length; j++) sum += preNeurons[j] * weights[index][j];
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
                weights[i][j] = weights[i][j] - chain[i] * preNeurons[j] * lernRate;
            }
        }
    }

    /**
     * Updates the biases with the backpropagation method
     */
    private void updateBiases(){
        for(int i = 0; i < neuronCount; i++) biases[i] = biases[i] - chain[i] * lernRate;
    }

    /**
     * Updates the Chain Rule
     * @param targets aims of the AI
     */
    private void updateChain(float[] targets) {
        chain = new float[neuronCount];
        nextChain = new float[preNeuronCount];
        for(int i = 0; i < neuronCount; i++) chain[i] = targets[i] * activationFunction.activationDerivative(add(i) + biases[i]);
        for(int i = 0; i < preNeuronCount; i++) {
            float chainSum = 0;
            for(int j = 0; j < neuronCount; j++) chainSum += chain[j] * weights[j][i];
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

    //endregion

    //region SAVE & LOAD

    /**
     * Saves the values of the layer
     * @return Data of the layer
     */
    public NNLayerData save(){
        NNLayerData daten = new NNLayerData();
        daten.biases = biases;
        daten.weights = weights;
        daten.neuronCount = neuronCount;
        daten.preNeuronCount = preNeuronCount;
        return daten;
    }

    /**
     * Loads the saved Data
     * @param daten saved Data
     */
    public void load(NNLayerData daten){
        biases = daten.biases;
        weights = daten.weights;
        neuronCount = daten.neuronCount;
        preNeuronCount = daten.preNeuronCount;
    }

    //endregion
}
