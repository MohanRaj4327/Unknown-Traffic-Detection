package com.ccsutd.miniproject.service;

import com.ccsutd.miniproject.dto.ConfidenceResult;
import com.ccsutd.miniproject.dto.PredictionResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

@Service
public class MachineLearningService {

    @Value("${ccsutd.model.beta:0.9}")
    private double betaThreshold;

    private RandomForest h2Classifier;

    /**
     * Trains the H2 Multi-class Classifier (Random Forest) on KnownC data.
     */
    public void trainH2Classifier(Instances knownTrainingData) throws Exception {
        h2Classifier = new RandomForest();
        h2Classifier.setNumIterations(100); 
        h2Classifier.buildClassifier(knownTrainingData);
    }

    /**
     * Evaluates a single flow instance through H2 and calculates confidence metrics.
     */
    public ConfidenceResult getH2Confidence(Instance instance, Instances dataHeader) throws Exception {
        if (h2Classifier == null) {
            throw new IllegalStateException("H2 Classifier is not trained yet.");
        }

        double[] probabilities = h2Classifier.distributionForInstance(instance);
        
        double cfMax = -1.0;
        double cfMin = 2.0; 
        int maxIndex = -1;
        
        Map<String, Double> classConfidences = new HashMap<>();

        for (int i = 0; i < probabilities.length; i++) {
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

    /**
     * PHASE 5: Fixed-Threshold Baseline Cascade.
     * Evaluates a sample using the fixed threshold (Beta) to detect New Classes.
     * If CfDmax > beta -> KnownC. Otherwise -> NewC.
     */
    public PredictionResult predictFixedThreshold(Instance instance, Instances dataHeader) throws Exception {
        // H1 is not implemented yet, so we assume H1 passes everything to H2 for now
        String h1Result = "PASS_TO_H2 (Not Implemented)";

        // Get confidences from H2
        ConfidenceResult confidenceResult = getH2Confidence(instance, dataHeader);

        String classType;
        String finalPrediction;

        // Apply Fixed-Threshold logic
        if (confidenceResult.getCfDMax() > betaThreshold) {
            classType = "KNOWN";
            finalPrediction = confidenceResult.getPredictedClass();
        } else {
            classType = "NEW";
            finalPrediction = "NEW_CLASS"; // Generic label for detected unknown
        }

        return new PredictionResult(finalPrediction, classType, confidenceResult, betaThreshold, h1Result);
    }
    
    // Setter for beta threshold testing
    public void setBetaThreshold(double betaThreshold) {
        this.betaThreshold = betaThreshold;
    }
}
