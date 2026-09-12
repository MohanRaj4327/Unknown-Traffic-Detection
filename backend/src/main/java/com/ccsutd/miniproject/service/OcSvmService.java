package com.ccsutd.miniproject.service;

import com.ccsutd.miniproject.dto.ConfidenceResult;
import com.ccsutd.miniproject.dto.OcSvmResult;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class OcSvmService {

    @Value("${ccsutd.model.alpha-0:0.9}")
    private double alphaThreshold;

    @Autowired
    private MachineLearningService mlService;

    // Store raw libsvm models
    private Map<String, svm_model> ocSvmModels = new HashMap<>();

    public OcSvmResult selectPseudoNegatives(Instances knownTrainingData, Instances unlabelledData) throws Exception {
        
        List<String> knownClasses = new ArrayList<>();
        for (int i = 0; i < knownTrainingData.numClasses(); i++) {
            String className = knownTrainingData.classAttribute().value(i);
            boolean isPresent = false;
            for (int j = 0; j < knownTrainingData.numInstances(); j++) {
                if (knownTrainingData.instance(j).stringValue(knownTrainingData.classIndex()).equals(className)) {
                    isPresent = true;
                    break;
                }
            }
            if (isPresent) {
                knownClasses.add(className);
            }
        }

        // Train one OC-SVM per Known Class using raw LibSVM
        for (String className : knownClasses) {
            
            // 1. Gather instances for this class
            List<Instance> classInstances = new ArrayList<>();
            for (int i = 0; i < knownTrainingData.numInstances(); i++) {
                if (knownTrainingData.instance(i).stringValue(knownTrainingData.classIndex()).equals(className)) {
                    classInstances.add(knownTrainingData.instance(i));
                }
            }
            
            // 2. Prepare svm_problem
            svm_problem prob = new svm_problem();
            prob.l = classInstances.size();
            prob.y = new double[prob.l];
            prob.x = new svm_node[prob.l][];
            
            int numFeatures = knownTrainingData.numAttributes() - 1;
            
            for (int i = 0; i < classInstances.size(); i++) {
                Instance inst = classInstances.get(i);
                svm_node[] nodes = new svm_node[numFeatures];
                for (int j = 0; j < numFeatures; j++) {
                    nodes[j] = new svm_node();
                    nodes[j].index = j + 1; // libsvm uses 1-based indexing for features
                    nodes[j].value = inst.value(j);
                }
                prob.x[i] = nodes;
                prob.y[i] = 1.0; // Target class label
            }
            
            // 3. Set parameters for One-Class SVM
            svm_parameter param = new svm_parameter();
            param.svm_type = svm_parameter.ONE_CLASS;
            param.kernel_type = svm_parameter.RBF;
            param.gamma = 1.0 / numFeatures;
            param.nu = 0.5; // typical default
            param.cache_size = 100;
            param.eps = 1e-3;
            
            // 4. Train model
            svm_model model = svm.svm_train(prob, param);
            ocSvmModels.put(className, model);
        }

        Instances pseudoNegatives = new Instances(unlabelledData, 0);
        Instances likelyKnowns = new Instances(unlabelledData, 0);
        int numFeatures = unlabelledData.numAttributes() - 1;

        // 5. Predict on Unlabelled Data
        for (int i = 0; i < unlabelledData.numInstances(); i++) {
            Instance sample = unlabelledData.instance(i);
            
            // Convert to libsvm format
            svm_node[] nodes = new svm_node[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                nodes[j] = new svm_node();
                nodes[j].index = j + 1;
                nodes[j].value = sample.value(j);
            }
            
            boolean rejectedByAll = true;
            for (Map.Entry<String, svm_model> entry : ocSvmModels.entrySet()) {
                svm_model model = entry.getValue();
                double prediction = svm.svm_predict(model, nodes);
                
                // One-class SVM predicts 1.0 for in-class, -1.0 for out-class
                if (prediction > 0) { 
                    rejectedByAll = false;
                    break;
                }
            }

            if (rejectedByAll) {
                ConfidenceResult conf = mlService.getH2Confidence(sample, knownTrainingData);
                double alpha_m = conf.getCfDMax();
                
                if (alpha_m > alphaThreshold) {
                    pseudoNegatives.add(sample);
                }
            } else {
                likelyKnowns.add(sample);
            }
        }

        return new OcSvmResult(pseudoNegatives, likelyKnowns);
    }
}
