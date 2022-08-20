package AI;

import java.io.*;
import java.util.Arrays;
import java.util.Queue;

public class NN {

    private NNLayer[][] layers;
    private final int[] layerSizes;
    private final int layerCount;

    private final ActivationFunction activationFunction;
    private final Action[] outPutActions;
    private float[][] outputs;


    private float accuracy;
    private int guesses;
    private int rightGuesses;

    private final float nLernRate;
    private final float momentum;
    private final int miniBatchLength;

    //Weights & Biases

    public float[][][] weights;
    public float[][] biases;

    public float[][][] weightGradients;
    public float[][] biasGradients;

    public float[][][] preWeightGradients;
    public float[][] preBiasGradients;


    public NN(int[] layerSizes, Action[] oOptions, ActivationType activation, float nLernRate, int miniBatchLength, float momentum) {
        this.layerSizes = layerSizes;
        this.outPutActions = oOptions;
        this.nLernRate = nLernRate;
        this.miniBatchLength = miniBatchLength;
        this.momentum = momentum;
        activationFunction = new ActivationFunction(activation);
        layerCount = layerSizes.length-1;
        buildLayers();
    }

    //region LAYER BUILDER
    /**
     * Builds Layers
     */
    public void buildLayers() {
        outputs = new float[miniBatchLength][outPutActions.length];
        layers = new NNLayer[miniBatchLength][layerCount];
        weights = new float[layerCount][][];
        biases = new float[layerCount][];
        clearGradients();

        preWeightGradients = new float[layerCount][][];
        preBiasGradients = new float[layerCount][];
        for(int l = 0; l < layerCount;l++){
            preWeightGradients[l] = new float[layerSizes[l+1]][layerSizes[l]];
            preBiasGradients[l] = new float[layerSizes[l+1]];
        }
        for(int b = 0; b < miniBatchLength;b++){
            for(int l = 0; l < layerCount; l++) {
                layers[b][l] = new NNLayer(layerSizes[l], layerSizes[l+1],activationFunction, nLernRate, this,l);
                if(b == 0) layers[b][l].buildLayer();
            }
        }
    }

    //endregion

    //region PROPAGATION
    /**
     * Propagation Method
     */
    public float[] propagation(float[][] inputs) {
        for(int j = 0;j < inputs.length;j++) {
            for (int i = 0; i < layerCount; i++) {
                inputs[j] = layers[j][i].propagation(inputs[j]);
            }
            outputs[j] = layers[j][layerCount-1].getNeurons();
        }
        return outputs[0];
    }
    //endregion

    //region BACKPROPAGATION
    /**
     * Backpropagation method
     * @param targets targets of the AI
     */
    public void backPropagation(Action[] targets) {
        for(int b = 0; b < miniBatchLength;b++){
            float[] outputTargets = new float[outputs[0].length];
            int maxIndex = 0;
            for(int i = 1; i < outputs[0].length; i++) if(outputs[b][i] > outputs[b][maxIndex]) maxIndex = i;
            updateAccuracy(targets[b],maxIndex);
            outputTargets[Arrays.asList(outPutActions).indexOf(targets[b])] = 1;

            for(int i = 0; i < outputs[b].length; i++) outputTargets[i] = 2*(outputs[b][i] - outputTargets[i]);

            layers[b][layerCount - 1].backPropagation(outputTargets);
            for(int i = layers[b].length - 2; i >= 0; i--) layers[b][i].backPropagation(layers[b][i + 1].getNextChain());
        }
        updateWeights();
        updateBiases();
        clearGradients();
    }

    /**
     * Clears the array of gradients and sets the previous gradients
     */
    private void clearGradients(){
        preWeightGradients = weightGradients;
        preBiasGradients = biasGradients;
        weightGradients = new float[layerCount][][];
        biasGradients = new float[layerCount][];
        for(int l = 0; l < layerCount;l++){
            weightGradients[l] = new float[layerSizes[l+1]][layerSizes[l]];
            biasGradients[l] = new float[layerSizes[l+1]];
        }
    }

    /**
     * Updates all weights by using the gradients from the miniBatches and momentum
     */
    public void updateWeights(){
        for(int i = 0; i < weights.length;i++){
            for(int j = 0; j < weights[i].length;j++){
                for(int x = 0; x < weights[i][j].length;x++){
                    if(momentum == 0) weights[i][j][x] -=  weightGradients[i][j][x] / miniBatchLength;
                    else weights[i][j][x] -= preWeightGradients[i][j][x] * momentum + (weightGradients[i][j][x] / miniBatchLength - preWeightGradients[i][j][x] / miniBatchLength);
                }
            }
        }
    }

    /**
     * Updates all biases by using the gradients from the miniBatches and momentum
     */
    public void updateBiases(){
        for(int i = 0; i < biases.length;i++){
            for(int j = 0; j < biases[i].length;j++){
                if(momentum == 0) biases[i][j] -= biasGradients[i][j] / miniBatchLength;
                else biases[i][j] -= preBiasGradients[i][j] * momentum + (biasGradients[i][j] / miniBatchLength - preBiasGradients[i][j] / miniBatchLength);
            }
        }
    }

    /**
     * Updates the Accuracy
     * @param target targets of the AI
     */
    private void updateAccuracy(Action target, int maxIndex){
        guesses += 1;
        if(outPutActions[maxIndex].equals(target)) rightGuesses += 1;
        accuracy = (float) rightGuesses / guesses * 100;
    }

    //endregion

    //region SAVE & LOAD

    /**
     * Saves the AI
     */
    public void save(String path) {
        if(accuracy > getSavedAccuracy(path)){
            try {
                FileOutputStream fos = new FileOutputStream(path);
                BufferedOutputStream bf = new BufferedOutputStream(fos);
                ObjectOutputStream obj = new ObjectOutputStream(bf);
                NNData data = new NNData();
                data.accuracy = accuracy;
                data.weights = weights;
                data.biases = biases;
                obj.writeObject(data);
                obj.close();
            } catch (IOException ignored) {}
        }
    }

    /**
     * Loads the AI
     */
    public void load(String path) {
        try {
            FileInputStream fis = new FileInputStream(path);
            BufferedInputStream bis = new BufferedInputStream(fis);
            ObjectInputStream ois = new ObjectInputStream(bis);
            NNData data = (NNData) ois.readObject();
            weights = data.weights;
            biases = data.biases;
        } catch (IOException | ClassNotFoundException ignored){}
    }

    //endregion

    //region GETTER & SETTER
    /**
     * @return Saved accuracy
     */
    public float getSavedAccuracy(String path){
        try {
            FileInputStream fis = new FileInputStream(path);
            BufferedInputStream bis = new BufferedInputStream(fis);
            ObjectInputStream ois = new ObjectInputStream(bis);

            NNData data = (NNData) ois.readObject();
            return data.accuracy;

        } catch (IOException | ClassNotFoundException ignored) {}
        return 0;
    }

    /**
     * @return Output with the highest value
     */
    public Action getOutput(){
        int maxIndex = 0;
        for(int i = 1; i < outputs[0].length;i++) if(outputs[0][i] > outputs[0][maxIndex]) maxIndex = i;
        return outPutActions[maxIndex];
    }

    /**
     * @return Current accuracy
     */
    public float getCurrentAccuracy(){
        return accuracy;
    }

    //endregion
}
