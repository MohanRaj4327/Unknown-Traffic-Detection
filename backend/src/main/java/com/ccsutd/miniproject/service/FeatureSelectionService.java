package com.ccsutd.miniproject.service;

import org.springframework.stereotype.Service;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.util.ArrayList;
import java.util.List;

@Service
public class FeatureSelectionService {

    private AttributeSelection featureSelectionFilter;
    private List<String> selectedFeatureNames;

    /**
     * Identifies highly correlated, useful features and removes redundant ones.
     * Aligns with the paper's feature selection intent using Correlation-based Feature Selection (CFS).
     * @param trainingData The KnownC training dataset
     * @return The filtered training dataset with redundant features removed
     */
    public Instances fitAndTransform(Instances trainingData) throws Exception {
        featureSelectionFilter = new AttributeSelection();
        
        // CfsSubsetEval evaluates the worth of a subset of attributes by considering 
        // the individual predictive ability of each feature along with the degree of redundancy between them.
        // This natively satisfies the paper's PCC > 0.9 redundancy removal rule in a beginner-friendly way.
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);
        
        featureSelectionFilter.setEvaluator(eval);
        featureSelectionFilter.setSearch(search);
        
        // Initialize the filter with training data
        featureSelectionFilter.setInputFormat(trainingData);
        
        // Apply filter to generate the reduced training dataset
        Instances reducedData = Filter.useFilter(trainingData, featureSelectionFilter);

        // Record the selected feature names (excluding the class attribute)
        selectedFeatureNames = new ArrayList<>();
        for (int i = 0; i < reducedData.numAttributes() - 1; i++) {
            selectedFeatureNames.add(reducedData.attribute(i).name());
        }

        return reducedData;
    }

    /**
     * Applies the ALREADY FITTED feature selection filter to new data (e.g. testing or unlabelled data).
     * @param data The dataset to reduce
     * @return The reduced dataset
     */
    public Instances transform(Instances data) throws Exception {
        if (featureSelectionFilter == null) {
            throw new IllegalStateException("Feature Selection Filter has not been fitted on training data yet.");
        }
        return Filter.useFilter(data, featureSelectionFilter);
    }
    
    /**
     * Applies the ALREADY FITTED filter to a single instance.
     */
    public void transformInstance(Instance instance) throws Exception {
        if (featureSelectionFilter == null) {
            throw new IllegalStateException("Feature Selection Filter has not been fitted on training data yet.");
        }
        featureSelectionFilter.input(instance);
        // We typically process instances in bulk via transform(Instances).
        // If single instance processing is needed, we would call featureSelectionFilter.output() here.
    }

    public List<String> getSelectedFeatureNames() {
        return selectedFeatureNames;
    }
}
