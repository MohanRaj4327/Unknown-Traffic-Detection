package com.ccsutd.miniproject.service;

import com.ccsutd.miniproject.dto.ConfidenceResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Service
public class AtsService {

    @Value("${ccsutd.model.ats.S:15}")
    private int numberOfSubsetsS;

    @Value("${ccsutd.model.ats.percentile:10}")
    private int targetPercentile;

    /**
     * Phase 10: Adaptive Threshold Selection (ATS)
     * Calculates the dynamic alpha threshold by running progressive RF updates.
     */
    public double calculateAdaptiveThreshold(Instances knownTrainingData, Instances pseudoNegatives, Instances likelyKnowns) throws Exception {
        
        if (likelyKnowns.numInstances() == 0 || pseudoNegatives.numInstances() == 0) {
            // Fallback if data is too small to split
            return 0.9;
        }

        List<Double> percentileValues = new ArrayList<>();
        
        // Calculate chunk size for S subsets
        int chunkSize = likelyKnowns.numInstances() / numberOfSubsetsS;
        if (chunkSize == 0) {
            chunkSize = 1;
            numberOfSubsetsS = likelyKnowns.numInstances();
        }

        int currentIndex = 0;

        for (int s = 0; s < numberOfSubsetsS; s++) {
            
            // a. Create progressive training dataset (KnownC + subset of M2)
            Instances currentTraining = new Instances(knownTrainingData);
            
            int endIdx = Math.min(currentIndex + chunkSize, likelyKnowns.numInstances());
            // Add subset
            for (int i = currentIndex; i < endIdx; i++) {
                currentTraining.add(likelyKnowns.instance(i));
            }
            currentIndex = endIdx;

            // b. Train temporary progressive Random Forest
            RandomForest progressiveRf = new RandomForest();
            progressiveRf.setNumIterations(50); // Smaller iteration for speed during ATS loop
            progressiveRf.buildClassifier(currentTraining);

            // c. Calculate CfDmax for all pseudo-negatives using this progressive RF
            List<Double> currentCfdMaxList = new ArrayList<>();
            for (int i = 0; i < pseudoNegatives.numInstances(); i++) {
                Instance pn = pseudoNegatives.instance(i);
                double[] probs = progressiveRf.distributionForInstance(pn);
                
                double cfMax = -1.0;
                double cfMin = 2.0;
                for (double p : probs) {
                    if (p > cfMax) cfMax = p;
                    if (p < cfMin) cfMin = p;
                }
                currentCfdMaxList.add(cfMax - cfMin);
            }

            // d. Sort and find the 10th percentile
            Collections.sort(currentCfdMaxList);
            int pIndex = (int) Math.ceil((targetPercentile / 100.0) * currentCfdMaxList.size()) - 1;
            if (pIndex < 0) pIndex = 0;
            
            percentileValues.add(currentCfdMaxList.get(pIndex));
        }

        // e. Calculate mean and std of the percentile values
        double sum = 0.0;
        for (double p : percentileValues) {
            sum += p;
        }
        double mean = sum / percentileValues.size();

        double sqSum = 0.0;
        for (double p : percentileValues) {
            sqSum += Math.pow(p - mean, 2);
        }
        double std = Math.sqrt(sqSum / percentileValues.size());

        // f. alpha = mean - std
        double alpha = mean - std;
        
        // Ensure alpha stays within valid probability bounds (0.0 to 1.0)
        return Math.max(0.0, Math.min(1.0, alpha));
    }
}
