package com.ccsutd.miniproject.service;

import com.ccsutd.miniproject.dto.ConfidenceResult;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import weka.classifiers.functions.LibSVM;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

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

    // Map of Class Name to its trained OC-SVM model
    private Map<String, LibSVM> ocSvmModels = new HashMap<>();

    /**
     * Phase 7: Train one OC-SVM per Known Class and extract pseudo-negative samples.
     * 
     * @param knownTrainingData The reduced training dataset containing only Known Classes
     * @param unlabelledData The reduced unlabelled dataset (contains mixed data)
     * @return Instances dataset containing the discovered pseudo-negative samples
     */
    public Instances selectPseudoNegatives(Instances knownTrainingData, Instances unlabelledData) throws Exception {
        
        // 1. Train one OC-SVM for each known class
        List<String> knownClasses = new ArrayList<>();
        for (int i = 0; i < knownTrainingData.numClasses(); i++) {
            String className = knownTrainingData.classAttribute().value(i);
            
            // Check if this class is actually present in the training data
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

        for (String className : knownClasses) {
            // Isolate data for this specific class
            RemoveWithValues filter = new RemoveWithValues();
            int classIndex = knownTrainingData.classIndex();
            filter.setAttributeIndex(String.valueOf(classIndex + 1));
            
            // Weka nominal index is 1-based in RemoveWithValues, matching the string value index + 1
            int nominalIndex = knownTrainingData.classAttribute().indexOfValue(className) + 1;
            filter.setNominalIndices(String.valueOf(nominalIndex));
            filter.setInvertSelection(true); // Keep only this class
            filter.setInputFormat(knownTrainingData);
            
            Instances singleClassData = Filter.useFilter(knownTrainingData, filter);

            // Train LibSVM as One-Class SVM (SVMType = 2)
            LibSVM svm = new LibSVM();
            svm.setSVMType(new weka.core.SelectedTag(2, LibSVM.TAGS_SVMTYPE));
            // RBF Kernel is default (KernelType = 2)
            svm.buildClassifier(singleClassData);
            
            ocSvmModels.put(className, svm);
        }

        // 2. Identify Candidates and extract Pseudo-Negatives
        Instances pseudoNegatives = new Instances(unlabelledData, 0);

        for (int i = 0; i < unlabelledData.numInstances(); i++) {
            Instance sample = unlabelledData.instance(i);
            
            // Check if ALL OC-SVMs reject it (classify as outlier)
            boolean rejectedByAll = true;
            for (Map.Entry<String, LibSVM> entry : ocSvmModels.entrySet()) {
                LibSVM svm = entry.getValue();
                // In Weka LibSVM One-Class: class 0 is usually target, class 1 is outlier
                // However, predict returns a double. We evaluate it.
                double prediction = svm.classifyInstance(sample);
                if (prediction == 0.0) { // Assuming 0.0 is target/accepted
                    rejectedByAll = false;
                    break;
                }
            }

            // Construct candidate group: those rejected by all known-class OC-SVMs
            if (rejectedByAll) {
                // 3. Use RF (H2) to obtain confidence values
                ConfidenceResult conf = mlService.getH2Confidence(sample, knownTrainingData);
                
                // 4. Calculate alpha_m (CfDmax)
                double alpha_m = conf.getCfDMax();
                
                // 5. If alpha_m > alpha: select as pseudo-negative
                if (alpha_m > alphaThreshold) {
                    pseudoNegatives.add(sample);
                }
            }
        }

        return pseudoNegatives;
    }
}
