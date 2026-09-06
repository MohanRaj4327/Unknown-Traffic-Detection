package com.ccsutd.miniproject.service;

import com.ccsutd.miniproject.dto.ConfidenceResult;
import com.ccsutd.miniproject.dto.PredictionResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

@Service
public class MachineLearningService {

    @Value("${ccsutd.model.beta:0.9}")
    private double betaThreshold;

    private RandomForest h2Classifier;
    
    private RandomForest h1Classifier;
    private Instances h1DatasetHeader;

    /**
     * Phase 8: Builds the binary dataset for H1 training by relabeling KnownC as "KNOWN" 
     * and pseudo-negatives as "NEW".
     */
    private Instances buildH1Dataset(Instances knownData, Instances pseudoNegatives) {
        // 1. Create new class attribute {KNOWN, NEW}
        ArrayList<String> classVals = new ArrayList<>();
        classVals.add("KNOWN");
        classVals.add("NEW");
        Attribute newClassAttr = new Attribute("h1_class", classVals);
        
        // 2. Create new dataset structure replacing the multi-class attribute
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < knownData.numAttributes() - 1; i++) {
            attributes.add(knownData.attribute(i));
        }
        attributes.add(newClassAttr);
        
        Instances h1Data = new Instances("H1_Binary_Dataset", attributes, knownData.numInstances() + pseudoNegatives.numInstances());
        h1Data.setClassIndex(h1Data.numAttributes() - 1);
        
        // 3. Add KnownC instances (Label = KNOWN)
        for (int i = 0; i < knownData.numInstances(); i++) {
            double[] values = new double[h1Data.numAttributes()];
            for (int j = 0; j < knownData.numAttributes() - 1; j++) {
                values[j] = knownData.instance(i).value(j);
            }
            values[h1Data.classIndex()] = h1Data.classAttribute().indexOfValue("KNOWN");
            h1Data.add(new DenseInstance(1.0, values));
        }
        
        // 4. Add Pseudo-negative instances (Label = NEW)
        for (int i = 0; i < pseudoNegatives.numInstances(); i++) {
            double[] values = new double[h1Data.numAttributes()];
            for (int j = 0; j < pseudoNegatives.numAttributes() - 1; j++) {
                values[j] = pseudoNegatives.instance(i).value(j);
            }
            values[h1Data.classIndex()] = h1Data.classAttribute().indexOfValue("NEW");
            h1Data.add(new DenseInstance(1.0, values));
        }
        
        return h1Data;
    }

    /**
     * Phase 8: Trains the H1 Binary Classifier (Random Forest) to detect difficult NewC early.
     */
    public void trainH1Classifier(Instances knownTrainingData, Instances pseudoNegatives) throws Exception {
        Instances h1TrainingData = buildH1Dataset(knownTrainingData, pseudoNegatives);
        
        h1Classifier = new RandomForest();
        h1Classifier.setNumIterations(100);
        h1Classifier.buildClassifier(h1TrainingData);
        
        h1DatasetHeader = new Instances(h1TrainingData, 0);
    }

    /**
     * Phase 4/9: Trains the H2 Multi-class Classifier (Random Forest) on KnownC data.
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
     */
    public PredictionResult predictFixedThreshold(Instance instance, Instances dataHeader) throws Exception {
        String h1Result = "PASS_TO_H2 (Not Implemented in baseline)";

        ConfidenceResult confidenceResult = getH2Confidence(instance, dataHeader);

        String classType;
        String finalPrediction;

        if (confidenceResult.getCfDMax() > betaThreshold) {
            classType = "KNOWN";
            finalPrediction = confidenceResult.getPredictedClass();
        } else {
            classType = "NEW";
            finalPrediction = "NEW_CLASS";
        }

        return new PredictionResult(finalPrediction, classType, confidenceResult, betaThreshold, h1Result);
    }
    
    public void setBetaThreshold(double betaThreshold) {
        this.betaThreshold = betaThreshold;
    }
}
