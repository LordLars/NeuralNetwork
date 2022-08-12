package AI;

public class ActivationFunction {

    private final ActivationType activationType;

    public ActivationFunction(ActivationType activation){
        this.activationType = activation;
    }

    /**
     * Selects the correct activation function
     * @param neuron neuron
     * @return activation
     */
    public float activation(float neuron){
        return switch (activationType) {
            case SigMoid -> sigMoid(neuron);
            case ReLU -> reLU(neuron);
        };
    }

    /**
     * Selects the correct activation function
     * @param neuron neuron
     * @return activation
     */
    public float activationDerivative(float neuron){
        return switch (activationType) {
            case SigMoid -> SigmoidDerivative(neuron);
            case ReLU -> reLUDerivative(neuron);
        };
    }

    //region SigMoid

    public float sigMoid(float neuron){
        return (float) (1 / (1 + Math.exp(-neuron)));
    }

    public float SigmoidDerivative(float neuron){
        return (sigMoid(neuron) * (1- sigMoid(neuron)));
    }
    //endregion

    //region ReLU

    public float reLU(float neuron){
        return Math.max(neuron,neuron*.1f);
    }

    public float reLUDerivative(float neuron){
        if(neuron > 0) return 1;
        else return .1f;
    }

    //endregion

}
