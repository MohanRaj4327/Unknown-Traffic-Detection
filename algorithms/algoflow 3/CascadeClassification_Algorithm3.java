package com.ccsutd.miniproject.algorithms.algoflow3;

import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * =========================================================================================
 * ALGORITHM 3: CASCADE CLASSIFICATION ALGORITHM
 * =========================================================================================
 * 
 * Based on research paper:
 * "Unknown Network Traffic Detection via Two-Stage Cascade Classifier with Adaptive Threshold"
 * 
 * Exact Flow Implementation:
 * - Require: Mixed flow dataset X; CfDmax threshold beta
 * - Output:  KnownC label z (z_1, z_2, ..., z_k); NewC label y
 * =========================================================================================
 */
public class CascadeClassification_Algorithm3 {

    /**
     * Data structure holding the final classification outputs from Algorithm 3.
     */
    public static class Algorithm3Result {
        // KnownC flows partitioned by specific class label (z_1, z_2, ..., z_k)
        public final Map<String, List<Instance>> knownClassPartitionsZ; 
        
        // All flows classified as Known classes: z = U z_j
        public final List<Instance> allKnownFlowsZ; 
        
        // Flows identified early by H1 as New Class: y_1
        public final List<Instance> h1EarlyNewClassY1; 
        
        // Flows rejected by H2 threshold as New Class: y_2
        public final List<Instance> h2ThresholdNewClassY2; 
        
        // Final complete set of New Class (Unknown/Zero-Day) flows: y = y_1 U y_2
        public final List<Instance> totalNewClassFlowsY; 

        public Algorithm3Result(Map<String, List<Instance>> knownClassPartitionsZ,
                                List<Instance> allKnownFlowsZ,
                                List<Instance> h1EarlyNewClassY1,
                                List<Instance> h2ThresholdNewClassY2,
                                List<Instance> totalNewClassFlowsY) {
            this.knownClassPartitionsZ = knownClassPartitionsZ;
            this.allKnownFlowsZ = allKnownFlowsZ;
            this.h1EarlyNewClassY1 = h1EarlyNewClassY1;
            this.h2ThresholdNewClassY2 = h2ThresholdNewClassY2;
            this.totalNewClassFlowsY = totalNewClassFlowsY;
        }
    }

    /**
     * Main Execution Flow for Algorithm 3: Cascade Classification
     * 
     * @param X                  Mixed network flow dataset (Require: Mixed flow dataset X)
     * @param beta               CfDmax threshold beta (Require: CfDmax threshold beta, calculated via ATS Algorithm 1)
     * @param H_1                Trained Binary Classifier H_1 (Distinguishes KNOWN vs NEW)
     * @param h1DatasetHeader    Weka header for H_1 input structure
     * @param H_2                Trained Multi-class Classifier H_2 (Classifies known classes z_1 ... z_k)
     * @param knownClassNames    Array of the k known class names {z_1, z_2, ..., z_k}
     * @return Algorithm3Result  Contains KnownC label partitions z and complete NewC label set y
     */
    public Algorithm3Result executeAlgorithm3(Instances X, 
                                              double beta, 
                                              RandomForest H_1, 
                                              Instances h1DatasetHeader, 
                                              RandomForest H_2, 
                                              List<String> knownClassNames) throws Exception {

        System.out.println(">>> [ALGORITHM 3] Starting Cascade Classification...");

        int k = knownClassNames.size();

        // ---------------------------------------------------------------------------------
        // Line 1: FE is applied to the dataset X to obtain the input sample x
        // (In our pipeline, Feature Extraction / Selection has transformed flows into 23 features)
        // ---------------------------------------------------------------------------------
        Instances x = new Instances(X);

        List<Instance> y_1 = new ArrayList<>(); // Line 2: Early NewC flows identified by H1
        List<Instance> x_0 = new ArrayList<>(); // Line 2: Flows that pass H1 to be evaluated by H2

        // ---------------------------------------------------------------------------------
        // Line 2: H_1 classifies x to obtain y_1 and x_0, where y_1 in y
        // ---------------------------------------------------------------------------------
        System.out.println("Line 2: Stage 1 (H_1) Binary Classification...");

        for (int idx = 0; idx < x.numInstances(); idx++) {
            Instance currentSample = x.instance(idx);

            // Prepare sample for H_1 binary format
            double[] h1Values = new double[h1DatasetHeader.numAttributes()];
            for (int f = 0; f < h1DatasetHeader.numAttributes() - 1; f++) {
                h1Values[f] = currentSample.value(f);
            }
            h1Values[h1DatasetHeader.classIndex()] = weka.core.Utils.missingValue();
            DenseInstance h1Instance = new DenseInstance(1.0, h1Values);
            h1Instance.setDataset(h1DatasetHeader);

            double h1Prediction = H_1.classifyInstance(h1Instance);
            String h1Label = h1DatasetHeader.classAttribute().value((int) h1Prediction);

            if ("NEW".equals(h1Label)) {
                // Caught early as New Class by H_1
                y_1.add(currentSample);
            } else {
                // Passed by H_1 as potential known traffic -> x_0
                x_0.add(currentSample);
            }
        }

        System.out.printf("Stage 1 Output: %d flows caught early by H_1 (y_1), %d flows passed to Stage 2 (x_0).%n", 
                          y_1.size(), x_0.size());

        // ---------------------------------------------------------------------------------
        // Outputs for Stage 2
        // ---------------------------------------------------------------------------------
        Map<String, List<Instance>> z_partitions = new HashMap<>();
        for (String className : knownClassNames) {
            z_partitions.put(className, new ArrayList<>());
        }
        List<Instance> allKnownZ = new ArrayList<>();
        List<Instance> y_2 = new ArrayList<>(); // Low confidence flows rejected by H_2 threshold

        // ---------------------------------------------------------------------------------
        // Line 3: H_2 outputs the confidence set lt = {t_1, t_2, ..., t_k} for each sample l in x_0
        // Line 4: Calculate the difference between maximum confidence and minimum confidence in lt:
        //         beta_l = t_max - t_min
        // Line 5: for l in x_0 do
        // ---------------------------------------------------------------------------------
        System.out.println("Lines 3-19: Stage 2 (H_2) Deep Multi-class Classification with ATS Threshold beta...");

        for (Instance l : x_0) {
            // Line 3: H_2 outputs class probabilities lt = {t_1, t_2, ..., t_k}
            double[] lt = H_2.distributionForInstance(l);

            // Line 4: Calculate beta_l = t_max - t_min (CfDmax)
            double t_max = -1.0;
            double t_min = 2.0;
            int maxIndexJ = -1;

            for (int col = 0; col < lt.length; col++) {
                double prob = lt[col];
                if (prob > t_max) {
                    t_max = prob;
                    maxIndexJ = col; // Line 8: index j of maximum confidence
                }
                if (prob < t_min) {
                    t_min = prob;
                }
            }

            double beta_l = t_max - t_min; // Confidence difference

            // Line 6: if beta_l > beta then
            if (beta_l > beta) {
                // Line 7: l in z (Sample confirmed as Known class)
                allKnownZ.add(l);

                // Line 8: Find the subscript index j of maximum confidence
                // Line 9 - Line 15: Map to specific known class z_j
                if (maxIndexJ >= 0 && maxIndexJ < k) {
                    String targetClass = knownClassNames.get(maxIndexJ);
                    z_partitions.get(targetClass).add(l);
                }
            } else {
                // Line 16: else
                // Line 17: l in y_2 (Confidence is too low -> Classified as New Class / Zero-Day)
                y_2.add(l);
                // Line 18: end if
            }
            // Line 19: end for
        }

        // ---------------------------------------------------------------------------------
        // Line 20: y = y_1 U y_2
        // Complete NewC predictions combining early H1 blocks and late H2 threshold blocks
        // ---------------------------------------------------------------------------------
        List<Instance> y = new ArrayList<>();
        y.addAll(y_1);
        y.addAll(y_2);

        System.out.printf("Line 20 Complete: Total Known flows z = %d | Total New Class flows y = %d (y_1: %d, y_2: %d)%n",
                          allKnownZ.size(), y.size(), y_1.size(), y_2.size());

        return new Algorithm3Result(z_partitions, allKnownZ, y_1, y_2, y);
    }
}
