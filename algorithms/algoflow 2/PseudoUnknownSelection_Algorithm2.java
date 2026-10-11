package com.ccsutd.miniproject.algorithms.algoflow2;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * =========================================================================================
 * ALGORITHM 2: SELECTION FOR PSEUDO UNKNOWN CLASS SAMPLES FROM UNLABELLED FLOW DATASETS
 * =========================================================================================
 * 
 * Based on research paper:
 * "Unknown Network Traffic Detection via Two-Stage Cascade Classifier with Adaptive Threshold"
 * 
 * Exact Flow Implementation:
 * - Require: KnownC_k (z_1, z_2, ..., z_k), CfDmax adaptive threshold alpha, RF_0 trained with known class samples
 * - Input:   Unlabelled flow dataset U
 * - Output:  Subset M of unlabelled data with unknown classes whose CfDmax > alpha,
 *            Subset M_2 of unlabelled data with known classes
 * =========================================================================================
 */
public class PseudoUnknownSelection_Algorithm2 {

    /**
     * Container class to return both output subsets (M and M_2) as specified in Line 26.
     */
    public static class Algorithm2Result {
        public final Instances M;   // Pseudo-unknown samples (CfDmax > alpha)
        public final Instances M_2; // Likely known samples

        public Algorithm2Result(Instances M, Instances M_2) {
            this.M = M;
            this.M_2 = M_2;
        }
    }

    /**
     * Helper to train a One-Class SVM on instances of a specific known class z_i.
     */
    private svm_model trainOneClassSVM(Instances classInstances) {
        int numFeatures = classInstances.numAttributes() - 1;
        svm_problem prob = new svm_problem();
        prob.l = classInstances.numInstances();
        prob.y = new double[prob.l];
        prob.x = new svm_node[prob.l][];

        for (int i = 0; i < prob.l; i++) {
            Instance inst = classInstances.instance(i);
            svm_node[] nodes = new svm_node[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                nodes[j] = new svm_node();
                nodes[j].index = j + 1;
                nodes[j].value = inst.value(j);
            }
            prob.x[i] = nodes;
            prob.y[i] = 1.0; // Inlier label
        }

        svm_parameter param = new svm_parameter();
        param.svm_type = svm_parameter.ONE_CLASS;
        param.kernel_type = svm_parameter.RBF;
        param.gamma = 1.0 / numFeatures;
        param.nu = 0.5;
        param.cache_size = 100;
        param.eps = 1e-3;

        return svm.svm_train(prob, param);
    }

    /**
     * Helper to evaluate an instance using a trained One-Class SVM model.
     * Returns +1 for inlier, -1 for outlier.
     */
    private double evaluateOCSVM(svm_model model, Instance inst, int numFeatures) {
        svm_node[] nodes = new svm_node[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            nodes[j] = new svm_node();
            nodes[j].index = j + 1;
            nodes[j].value = inst.value(j);
        }
        return svm.svm_predict(model, nodes);
    }

    /**
     * Main Execution Flow for Algorithm 2:
     * 
     * @param knownClassSubsets KnownC_k = {z_1, z_2, ..., z_k}: List of datasets for each known class
     * @param alpha             CfDmax adaptive threshold alpha
     * @param RF_0              Random Forest model trained on known class samples
     * @param U                 Unlabelled flow dataset
     * @return Algorithm2Result containing {M, M_2}
     */
    public Algorithm2Result executeAlgorithm2(List<Instances> knownClassSubsets, 
                                              double alpha, 
                                              RandomForest RF_0, 
                                              Instances U) throws Exception {

        System.out.println(">>> [ALGORITHM 2] Starting Pseudo-Unknown Sample Selection...");

        int k = knownClassSubsets.size();
        int numFeatures = U.numAttributes() - 1;

        // ---------------------------------------------------------------------------------
        // Line 1: for i = 1 to k do
        // Line 2:    Train with KnownC z_i to obtain OC-SVM_i
        // Line 3: end for
        // ---------------------------------------------------------------------------------
        System.out.println("Lines 1-3: Training One-Class SVM models for each of the k known classes...");
        List<svm_model> ocSvmModels = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            Instances z_i = knownClassSubsets.get(i);
            svm_model ocSvm_i = trainOneClassSVM(z_i);
            ocSvmModels.add(ocSvm_i);
        }

        // ---------------------------------------------------------------------------------
        // Line 4: for i = 1 to k do
        // Line 5:    Classify U with OC-SVM_i and obtain U_{i1} and U_{i-1}
        // Line 6: end for
        // ---------------------------------------------------------------------------------
        System.out.println("Lines 4-6: Classifying unlabelled flows U with OC-SVM models...");
        // U_inlier.get(i) stores U_{i1} (inliers, label +1)
        // U_outlier.get(i) stores U_{i-1} (outliers, label -1)
        List<Set<Integer>> U_inlier = new ArrayList<>();
        List<Set<Integer>> U_outlier = new ArrayList<>();

        for (int i = 0; i < k; i++) {
            U_inlier.add(new HashSet<>());
            U_outlier.add(new HashSet<>());
        }

        for (int mIndex = 0; mIndex < U.numInstances(); mIndex++) {
            Instance inst = U.instance(mIndex);
            for (int i = 0; i < k; i++) {
                svm_model ocSvm_i = ocSvmModels.get(i);
                double prediction = evaluateOCSVM(ocSvm_i, inst, numFeatures);
                if (prediction > 0) {
                    U_inlier.get(i).add(mIndex);   // Belongs to U_{i1}
                } else {
                    U_outlier.get(i).add(mIndex);  // Belongs to U_{i-1}
                }
            }
        }

        // ---------------------------------------------------------------------------------
        // Line 7: M_1 = U_{11} U U_{21} U U_{31} U ... U U_{k1}
        // ---------------------------------------------------------------------------------
        Set<Integer> M_1_indices = new HashSet<>();
        for (int i = 0; i < k; i++) {
            M_1_indices.addAll(U_inlier.get(i));
        }

        // ---------------------------------------------------------------------------------
        // Line 8:  for each m in M_1 do
        // Line 9:     if m only in U_{11} \lor U_{21} \lor U_{31} \lor ... \lor U_{k1} then
        // Line 10:       m in M_2
        // Line 11:    else
        // Line 12:       m in M_3
        // Line 13:    end if
        // Line 14: end for
        // ---------------------------------------------------------------------------------
        Set<Integer> M_2_indices = new HashSet<>(); // Known flows with unique class membership
        Set<Integer> M_3_indices = new HashSet<>(); // Ambiguous flows classified as inlier by > 1 OC-SVM

        for (int mIndex : M_1_indices) {
            int matchCount = 0;
            for (int i = 0; i < k; i++) {
                if (U_inlier.get(i).contains(mIndex)) {
                    matchCount++;
                }
            }

            // Line 9: "m only in U_11 OR U_21 ...": matches exactly ONE class boundary
            if (matchCount == 1) {
                // Line 10: m in M_2
                M_2_indices.add(mIndex);
            } else {
                // Line 12: m in M_3 (boundary overlap / conflict)
                M_3_indices.add(mIndex);
            }
        }

        // ---------------------------------------------------------------------------------
        // Line 15: M_4 = U_{1-1} \cap U_{2-1} \cap U_{3-1} \cap ... \cap U_{k-1}
        // (Outliers rejected by ALL k models)
        // ---------------------------------------------------------------------------------
        Set<Integer> M_4_indices = new HashSet<>();
        for (int mIndex = 0; mIndex < U.numInstances(); mIndex++) {
            boolean rejectedByAll = true;
            for (int i = 0; i < k; i++) {
                if (!U_outlier.get(i).contains(mIndex)) {
                    rejectedByAll = false;
                    break;
                }
            }
            if (rejectedByAll) {
                M_4_indices.add(mIndex);
            }
        }

        // ---------------------------------------------------------------------------------
        // Line 16: M_5 = M_3 U M_4
        // (Combine ambiguous boundary samples M_3 and unanimous outliers M_4)
        // ---------------------------------------------------------------------------------
        Set<Integer> M_5_indices = new HashSet<>();
        M_5_indices.addAll(M_3_indices);
        M_5_indices.addAll(M_4_indices);

        // ---------------------------------------------------------------------------------
        // Line 17: RF output confidence set m_t = {t_1, t_2, ..., t_k} for each sample m in M_5
        // Line 18: Calculate the difference between maximum confidence and minimum confidence in m_t:
        //          alpha_m = t_max - t_min
        // Line 19: for each m in M_5 do
        // Line 20:    if alpha_m > alpha then
        // Line 21:       m in M
        // Line 22:    else
        // Line 23:       Discard m
        // Line 24:    end if
        // Line 25: end for
        // ---------------------------------------------------------------------------------
        Instances M = new Instances(U, 0); // Candidate unknown class dataset
        Instances M_2 = new Instances(U, 0); // Known flows dataset

        for (int mIndex : M_2_indices) {
            M_2.add(U.instance(mIndex));
        }

        for (int mIndex : M_5_indices) {
            Instance m = U.instance(mIndex);

            // Line 17: Output confidence probabilities m_t = {t_1, t_2, ..., t_k} using RF_0
            double[] m_t = RF_0.distributionForInstance(m);

            // Line 18: Calculate alpha_m = t_max - t_min (CfDmax)
            double t_max = -1.0;
            double t_min = 2.0;
            for (double prob : m_t) {
                if (prob > t_max) t_max = prob;
                if (prob < t_min) t_min = prob;
            }
            double alpha_m = t_max - t_min;

            // Line 20: if alpha_m > alpha then
            if (alpha_m > alpha) {
                // Line 21: m in M
                M.add(m);
            } else {
                // Line 23: Discard m
                // (Ignored / not added to M)
            }
            // Line 25: end for
        }

        System.out.printf("Lines 19-25 complete: Extracted %d Pseudo-Unknowns (M) and %d Known Flows (M_2).%n", 
                          M.numInstances(), M_2.numInstances());

        // ---------------------------------------------------------------------------------
        // Line 26: Return M, M_2
        // ---------------------------------------------------------------------------------
        return new Algorithm2Result(M, M_2);
    }
}
