package com.ccsutd.miniproject.algorithms.algoflow1;

import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * =========================================================================================
 * ALGORITHM 1: ADAPTIVE THRESHOLD SELECTION (ATS)
 * =========================================================================================
 * 
 * Based on research paper:
 * "Unknown Network Traffic Detection via Two-Stage Cascade Classifier with Adaptive Threshold"
 * 
 * Exact Flow Implementation:
 * - Require: Size of memory S, RF_0 trained with known class samples, prediction memory, initial threshold alpha_0
 * - Input:   Unlabelled flow dataset U
 * - Output:  Adaptive threshold alpha
 * =========================================================================================
 */
public class AdaptiveThresholdSelection_Algorithm1 {

    // --- Hyperparameters and Settings ---
    private int S;                  // Size of memory / Number of subsets (default: 15)
    private double alpha_0;         // Initial threshold (default: 0.90)
    private int targetPercentile;   // Percentile cutoff (default: 10th percentile)

    public AdaptiveThresholdSelection_Algorithm1(int memorySizeS, double initialAlpha, int percentile) {
        this.S = memorySizeS;
        this.alpha_0 = initialAlpha;
        this.targetPercentile = percentile;
    }

    /**
     * Compute Confidence Difference (CfDmax) for a single instance m using a Random Forest model.
     * CfDmax = (Highest predicted probability) - (Lowest predicted probability)
     */
    public double computeCfDmax(RandomForest model, Instance m) throws Exception {
        double[] classProbabilities = model.distributionForInstance(m);
        
        double maxProb = -1.0;
        double minProb = 2.0;
        
        for (double p : classProbabilities) {
            if (p > maxProb) maxProb = p;
            if (p < minProb) minProb = p;
        }
        
        return (maxProb - minProb);
    }

    /**
     * Main Execution Flow for Algorithm 1: Adaptive Threshold Selection
     * 
     * @param RF_0             Initial Random Forest trained on known class samples
     * @param U                Unlabelled network flow dataset
     * @param M_2              Known flows obtained by Algorithm 2
     * @param M                Subset of unlabelled data with unknown classes whose CfDmax > alpha_0 obtained by Algorithm 2
     * @return alpha           The calculated adaptive decision threshold
     */
    public double executeAlgorithm1(RandomForest RF_0, Instances U, Instances M_2, Instances M) throws Exception {

        System.out.println(">>> [ALGORITHM 1] Starting Adaptive Threshold Selection (ATS)...");

        // ---------------------------------------------------------------------------------
        // Line 1: Obtain known flows M2 by Algorithm 2
        // (Passed as parameter M_2 from Algorithm 2)
        // ---------------------------------------------------------------------------------
        if (M_2 == null || M_2.numInstances() == 0) {
            System.out.println("[ALGORITHM 1 Warning] M_2 is empty. Returning initial threshold alpha_0.");
            return alpha_0;
        }

        // ---------------------------------------------------------------------------------
        // Line 2: Obtain subset M of unlabelled data with unknown classes whose CfDmax > alpha_0 by Algorithm 2
        // (Passed as parameter M from Algorithm 2)
        // ---------------------------------------------------------------------------------
        if (M == null || M.numInstances() == 0) {
            System.out.println("[ALGORITHM 1 Warning] M is empty. Returning initial threshold alpha_0.");
            return alpha_0;
        }

        // ---------------------------------------------------------------------------------
        // Line 3: # Update RF_0 to recalculate CfDmax for M
        // ---------------------------------------------------------------------------------
        System.out.println("Line 3: Initializing model collection and preparing subsets...");

        // ---------------------------------------------------------------------------------
        // Line 4: Divide M2 into S subsets: M2 = {Sub_1, Sub_2, ..., Sub_S}
        // ---------------------------------------------------------------------------------
        List<Instances> subsets = new ArrayList<>();
        int totalInstancesM2 = M_2.numInstances();
        int chunkSize = totalInstancesM2 / S;
        if (chunkSize == 0) chunkSize = 1;

        int startIndex = 0;
        for (int i = 0; i < S; i++) {
            int endIndex = Math.min(startIndex + chunkSize, totalInstancesM2);
            Instances sub = new Instances(M_2, startIndex, endIndex - startIndex);
            subsets.add(sub);
            startIndex = endIndex;
        }

        // ---------------------------------------------------------------------------------
        // Line 5: for i = 1 to S do
        // ---------------------------------------------------------------------------------
        List<RandomForest> trainedModelsRF = new ArrayList<>();
        RandomForest previousModel = RF_0;

        for (int i = 0; i < S; i++) {
            // Line 6: Define Training set Set_i = Sub_1 U Sub_2 U ... U Sub_i
            Instances Set_i = new Instances(subsets.get(0));
            for (int k = 1; k <= i; k++) {
                for (int n = 0; n < subsets.get(k).numInstances(); n++) {
                    Set_i.add(subsets.get(k).instance(n));
                }
            }

            // Line 7: Retrain RF_i using RF_{i-1} as the initial model and Training set Set_i
            RandomForest RF_i = new RandomForest();
            RF_i.setNumIterations(50); // Progressive retraining parameters
            RF_i.buildClassifier(Set_i);

            // Line 8: Obtain and store updated model RF_i
            trainedModelsRF.add(RF_i);
            previousModel = RF_i;

            // Line 9: end for
        }

        // ---------------------------------------------------------------------------------
        // Line 10: Initialize a temporary array: temp
        // ---------------------------------------------------------------------------------
        List<Double> predictionMemory = new ArrayList<>(); // Memory storing 10th percentile from each iteration

        // ---------------------------------------------------------------------------------
        // Line 11: for j = 1 to S do
        // ---------------------------------------------------------------------------------
        for (int j = 0; j < S; j++) {
            RandomForest RF_j = trainedModelsRF.get(j);
            List<Double> temp = new ArrayList<>(); // Line 10 re-initialized per subset iteration j

            // Line 12: for each m in M do
            for (int idx = 0; idx < M.numInstances(); idx++) {
                Instance m = M.instance(idx);

                // Line 13: Compute CfDmax-m <- calculation using RF_j on m
                double CfDmax_m = computeCfDmax(RF_j, m);

                // Line 14: Store CfDmax-m in temp
                temp.add(CfDmax_m);

                // Line 15: end for
            }

            // Line 16: Calculate the 10th percentile of temp and store it in memory
            Collections.sort(temp);
            int percentileIndex = (int) Math.ceil((targetPercentile / 100.0) * temp.size()) - 1;
            if (percentileIndex < 0) percentileIndex = 0;
            if (percentileIndex >= temp.size()) percentileIndex = temp.size() - 1;

            double tenthPercentileValue = temp.get(percentileIndex);
            predictionMemory.add(tenthPercentileValue);

            // Line 17: end for
        }

        // ---------------------------------------------------------------------------------
        // Line 18: # Calculate mean and standard deviation in memory to obtain adaptive threshold alpha
        // ---------------------------------------------------------------------------------

        // Line 19: Calculate the mean in memory: mean
        double sum = 0.0;
        for (double val : predictionMemory) {
            sum += val;
        }
        double mean = sum / predictionMemory.size();

        // Line 20: Calculate the standard deviation in memory: std
        double squaredSum = 0.0;
        for (double val : predictionMemory) {
            squaredSum += Math.pow(val - mean, 2);
        }
        double std = Math.sqrt(squaredSum / predictionMemory.size());

        // Line 21: Adaptive threshold alpha = mean - std
        double alpha = mean - std;

        // Ensure boundary constraints between [0.0, 1.0]
        alpha = Math.max(0.0, Math.min(1.0, alpha));

        System.out.printf("Line 21: Mean = %.4f | Std = %.4f | Computed Alpha = %.4f%n", mean, std, alpha);

        // ---------------------------------------------------------------------------------
        // Line 22: Return alpha
        // ---------------------------------------------------------------------------------
        return alpha;
    }
}
