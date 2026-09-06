package com.ccsutd.miniproject.service;

import com.ccsutd.miniproject.dto.ConfidenceResult;
import com.ccsutd.miniproject.dto.OcSvmResult;
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

    private Map<String, LibSVM> ocSvmModels = new HashMap<>();

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

        for (String className : knownClasses) {
            RemoveWithValues filter = new RemoveWithValues();
            int classIndex = knownTrainingData.classIndex();
            filter.setAttributeIndex(String.valueOf(classIndex + 1));
            
            int nominalIndex = knownTrainingData.classAttribute().indexOfValue(className) + 1;
            filter.setNominalIndices(String.valueOf(nominalIndex));
            filter.setInvertSelection(true); 
            filter.setInputFormat(knownTrainingData);
            
            Instances singleClassData = Filter.useFilter(knownTrainingData, filter);

            LibSVM svm = new LibSVM();
            svm.setSVMType(new weka.core.SelectedTag(2, LibSVM.TAGS_SVMTYPE));
            svm.buildClassifier(singleClassData);
            
            ocSvmModels.put(className, svm);
        }

        Instances pseudoNegatives = new Instances(unlabelledData, 0);
        Instances likelyKnowns = new Instances(unlabelledData, 0);

        for (int i = 0; i < unlabelledData.numInstances(); i++) {
            Instance sample = unlabelledData.instance(i);
            
            boolean rejectedByAll = true;
            for (Map.Entry<String, LibSVM> entry : ocSvmModels.entrySet()) {
                LibSVM svm = entry.getValue();
                double prediction = svm.classifyInstance(sample);
                if (prediction == 0.0) { 
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
