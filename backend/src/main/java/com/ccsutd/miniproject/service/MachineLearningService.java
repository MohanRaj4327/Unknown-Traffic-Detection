package com.ccsutd.miniproject.service;

import com.ccsutd.miniproject.dto.ConfidenceResult;
import org.springframework.stereotype.Service;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

@Service
public class MachineLearningService {

    private RandomForest h2Classifier;

    /**
     * Trains the H2 Multi-class Classifier (Random Forest) on KnownC data.
     * @param knownTrainingData The dataset containing ONLY known classes.
     * @throws Exception if training fails
     */
    public void trainH2Classifier(Instances knownTrainingData) throws Exception {
        h2Classifier = new RandomForest();
        // Set number of trees (can be made configurable later)
        h2Classifier.setNumIterations(100); 
        h2Classifier.buildClassifier(knownTrainingData);
    }

    /**
     * Evaluates a single flow instance through H2 and calculates confidence metrics.
     * @param instance The network flow instance to evaluate
     * @param dataHeader The Instances header needed to resolve class names
     * @return ConfidenceResult containing Cfmax, Cfmin, and CfDmax
     * @throws Exception if prediction fails
     */
    public ConfidenceResult getH2Confidence(Instance instance, Instances dataHeader) throws Exception {
        if (h2Classifier == null) {
            throw new IllegalStateException("H2 Classifier is not trained yet.");
        }

        // distributionForInstance returns probabilities for each class
        double[] probabilities = h2Classifier.distributionForInstance(instance);
        
        double cfMax = -1.0;
        double cfMin = 2.0; // Initialized above max possible probability (1.0)
        int maxIndex = -1;
        
        Map<String, Double> classConfidences = new HashMap<>();

        // Calculate Cfmax and Cfmin exclusively across the known classes 
        // that the model was trained on.
        for (int i = 0; i < probabilities.length; i++) {
            // Weka's probability distributions correspond to the class values 
            // defined in the ARFF header.
            double prob = probabilities[i];
            String className = dataHeader.classAttribute().value(i);
            classConfidences.put(className, prob);

            if (prob > cfMax) {
                cfMax = prob;
                maxIndex = i;
            }
            if (prob < cfMin) {
                cfMin = prob;
            }
        }

        double cfDMax = cfMax - cfMin;
        String predictedClass = dataHeader.classAttribute().value(maxIndex);

        return new ConfidenceResult(predictedClass, classConfidences, cfMax, cfMin, cfDMax);
    }
}
